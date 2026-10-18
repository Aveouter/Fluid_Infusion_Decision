"""Candidate curation and hash/ZIP sealing under the confirmed data contract."""
from pathlib import Path
import argparse
import csv
import json
import shutil
import zipfile
import pandas as pd
from .prepare import digest
from .vendor.inventory_event_sources import setting_source_events
from .vendor.inventory_event_availability import resolve_past_doctor_settings,resolve_past_setting_events


def curate(package):
    package=Path(package)
    assert not (package/'BUILD_INCOMPLETE').exists()
    previous=package/'provenance/final_curation.json'
    if previous.exists():
        existing=json.loads(previous.read_text(encoding='utf-8'))
        if existing['code_sha256']!=digest(Path(__file__)):raise ValueError('Curation code changed; use a new package version')
        return existing
    path=package/'hourly/oral_liquid.csv';df=pd.read_csv(path,dtype={'patient_id':str})
    mask=df.full_inpatient_hour&~df.recorded_oral_event_present&df.oral_liquid_estimate_ml.isna()
    df.loc[mask,'oral_liquid_estimate_ml']=0.
    df.to_csv(path,index=False,encoding='utf-8-sig')
    # Restore explicit clock precision and retiming linkage to the normalized
    # reference curve. The underlying resolver preserves values/clocks but its
    # compact output intentionally omits source precision metadata.
    cohort=pd.read_csv(package/'cohort/patients.csv',dtype={'patient_id':str})
    # Only416 patients have the retained boundary/split extraction record.
    # Do not mislabel an excluded patient's unavailable historical split as dev.
    cohort['historical_holdout']=cohort['historical_holdout'].astype('boolean')
    cohort.loc[~cohort.eligible,'historical_holdout']=pd.NA
    cohort.to_csv(package/'cohort/patients.csv',index=False,encoding='utf-8-sig')
    doctors=pd.read_csv(package/'fluid_events/doctor_observations.csv',dtype={'patient_id':str})
    notes=pd.read_csv(package/'fluid_events/nursing_events.csv',dtype={'patient_id':str})
    dg=dict(tuple(doctors.groupby('patient_id')));ng=dict(tuple(notes.groupby('patient_id')))
    ledger=[]
    for patient in cohort[cohort.eligible].itertuples():
        dd=dg.get(patient.patient_id,doctors.iloc[:0]);nn=ng.get(patient.patient_id,notes.iloc[:0])
        d,n=setting_source_events(dd,nn,patient.injury_time_local)
        end=max([73.]+[e['available_hour']+1 for e in n]+[e['hour']+1 for e in n])
        retimed={e['source_id']:e for e in resolve_past_doctor_settings(d,n,end)}
        doc={e.rate_event_id:e for e in dd.itertuples()};nurse={e.rate_event_id:e for e in nn.itertuples()}
        for event in resolve_past_setting_events(d,n,end):
            if not 0<=event['hour']<72:continue
            sid=event['source_id'];raw=doc.get(sid,nurse.get(sid));link=retimed.get(sid,{}).get('timing_narrative_source_id','')
            timing=nurse.get(link,raw)
            ledger.append(dict(event,patient_id=patient.patient_id,original_event_hour=float(raw.hour_from_injury),
                original_time_precision=raw.time_precision,time_precision=timing.time_precision,
                timing_narrative_source_id=link))
    reference=pd.DataFrame(ledger);reference.to_csv(package/'fluid_events/reference_setting_events.csv',index=False,encoding='utf-8-sig')
    raw_path=package/'provenance/raw_intake_events.csv'
    original_intakes=pd.read_csv(raw_path,dtype=str,keep_default_na=False,low_memory=False)
    intake_hours=pd.to_numeric(original_intakes.hour_from_injury,errors='coerce')
    in_window=intake_hours.ge(0)&intake_hours.lt(72)
    original_intakes.loc[in_window].to_csv(raw_path,index=False,encoding='utf-8-sig')
    precision={(r.patient_id,r.source_id):r.time_precision for r in reference.itertuples()}
    path=package/'fluid_segments/device_settings.csv';segments=pd.read_csv(path,dtype={'patient_id':str},keep_default_na=False)
    segments['time_precision']=[precision.get((r.patient_id,r.source_id),'unknown_initial') for r in segments.itertuples()]
    segments.to_csv(path,index=False,encoding='utf-8-sig')
    report=dict(policy='Complete inpatient hour and no recorded oral event: recorded-no-input zero; recorded missing quantity remains unknown.',
                oral_no_record_zeros_added=int(mask.sum()),other_numeric_fields_changed=False,
                normalized_reference_setting_events=len(reference),explicit_precision_restored=True,
                structured_intake_records_retained_in72h=int(in_window.sum()),
                structured_intake_records_outside72h_omitted=int((~in_window).sum()),
                reference_setting_values_and_event_clocks_changed=False,
                code_sha256=digest(Path(__file__)))
    (package/'provenance/final_curation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def seal(package,destination,archive,public):
    package=Path(package).resolve();destination=Path(destination).resolve();archive=Path(archive).resolve();public=Path(public).resolve()
    assert not destination.exists() and not archive.exists()
    checks=json.loads((package/'validation/aggregate.json').read_text(encoding='utf-8'))
    assert checks['status']=='passed'
    assert (package/'dictionary/fields.csv').exists()
    assert (package/'provenance/final_curation.json').exists()
    assert not (package/'BUILD_INCOMPLETE').exists()
    # Copy documentation and the completed extraction tools into the sealed
    # package, while retaining the exact earlier build snapshot separately.
    for name in ['README.md','METHODS.md','DATA_REPORT.md','REPRODUCIBILITY.json']:shutil.copy2(public/name,package/name)
    text=(package/'README.md').read_text(encoding='utf-8').replace('](fields.csv)','](dictionary/fields.csv)').replace('](clinical_fields.csv)','](dictionary/clinical_fields.csv)')
    (package/'README.md').write_text(text,encoding='utf-8')
    shutil.copytree(public/'examples',package/'examples')
    tools=Path(__file__).resolve().parent
    shutil.copytree(tools,package/'provenance/release_tools',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    meta=json.loads((package/'manifest.json').read_text(encoding='utf-8'))
    assert meta['status']=='candidate_awaiting_content_review','Only a completed clean build can be sealed'
    meta['tables_before_curation']=meta['tables']
    meta['tables']={p.relative_to(package).as_posix():len(pd.read_csv(p,usecols=[0],low_memory=False)) for p in package.rglob('*.csv')
                    if not p.relative_to(package).as_posix().startswith(('provenance/build_inputs/','provenance/final_builder/','provenance/release_tools/','dictionary/','examples/'))}
    meta.update(status='frozen_research_data_version',final_scope='Version/rules frozen; estimates and source uncertainty retained; independent clinical review not completed',
        content_authorization='2026-10-19 user confirmed all three package details (全部同意), then requested resuming the model goal after data organization; freeze uses that agreed scope and uncertainty policy',
        validation_sha256=digest(package/'validation/aggregate.json'),dictionary_sha256=digest(package/'dictionary/fields.csv'))
    (package/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    files=[]
    for p in sorted(package.rglob('*')):
        if p.is_file() and p.name!='file_checksums.csv':files.append(dict(path=p.relative_to(package).as_posix(),bytes=p.stat().st_size,sha256=digest(p)))
    with (package/'file_checksums.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(files)
    # Atomic rename on the same volume, with no recursive link traversal.
    destination.parent.mkdir(parents=True,exist_ok=True);package.rename(destination)
    archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(destination.rglob('*')):
            if p.is_file():z.write(p,arcname='final_v1/'+p.relative_to(destination).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    release=dict(version=meta['version'],status=meta['status'],package_manifest_sha256=digest(destination/'manifest.json'),
        checksum_index_sha256=digest(destination/'file_checksums.csv'),archive_sha256=digest(archive),
        archive_bytes=archive.stat().st_size,package_file_bytes=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()),
        source_patients=449,eligible_patients=416,development_patients=333,historical_holdout_patients=83,
        excluded_missing_admission=33,independent_clinical_review=False,model_scores_computed=False,
        validation=checks)
    (public/'RELEASE.json').write_text(json.dumps(release,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return release


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('curate');a.add_argument('--package',type=Path,required=True)
    a=sub.add_parser('seal');a.add_argument('--package',type=Path,required=True);a.add_argument('--destination',type=Path,required=True);a.add_argument('--archive',type=Path,required=True);a.add_argument('--public',type=Path,required=True)
    args=p.parse_args()
    result=curate(args.package) if args.command=='curate' else seal(args.package,args.destination,args.archive,args.public)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
