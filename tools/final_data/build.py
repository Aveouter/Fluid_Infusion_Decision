"""Build a model-independent, local-only Final candidate from versioned evidence."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
import platform
import shutil
import sys
import time
import numpy as np
import pandas as pd
from .prepare import prepare,digest
from .extract import clinical_records,settings_and_loads
from .quality import reference_rows
from .vendor.fluid_reconstruction_sources import DrugReference,extract_intake,doctor_rates
from .vendor.volume_speed_features import baseline_available_hour,baseline_consensus,BASELINE
from .vendor.inventory_event_sources import setting_source_events
from .vendor.inventory_primary_source_loads import primary_source_load_candidates
from .vendor.inventory_event_availability import resolve_past_setting_events
from .vendor.inventory_event_state import past_event_state

COMPONENTS=('crystalloid','colloid','water')


def save_csv(rows,path,columns=None):
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows,columns=columns)
    path.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(path,index=False,encoding='utf-8-sig')
    return len(frame)


def read(path):return pd.read_csv(path,dtype={'patient_id':str,'form_id':str,'inner_seq':str},low_memory=False)


def build(inputs,raw_root,out):
    started=time.time();inputs=Path(inputs).resolve();raw_root=Path(raw_root).resolve();out=Path(out).resolve()
    code_root=Path(__file__).resolve().parent
    used=set()
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.is_relative_to(code_root) and p.suffix=='.py':used.add(p)
    used.update((code_root/'vendor').glob('*.json'))
    initial_code={str(p.relative_to(code_root)):digest(p) for p in sorted(used)}
    if out.exists():raise FileExistsError('Use a new candidate version; no overwrite')
    manifest=json.loads((inputs/'input_manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:assert digest(inputs/item['path'])==item['sha256'],item['path']
    out.mkdir(parents=True)
    (out/'execution_code_sha256.json').write_text(json.dumps(initial_code,indent=2)+'\n',encoding='utf-8')
    shutil.copytree(inputs,out/'provenance/build_inputs')
    (out/'BUILD_INCOMPLETE').write_text('Candidate incomplete until validation succeeds.\n')
    source=read(inputs/'raw_source_manifest.csv').set_index('patient_id')
    patients=read(inputs/'patient_index.csv').set_index('patient_id')
    boundary=read(inputs/'boundary_reference.csv').set_index('patient_id')
    baseline=pd.read_excel(raw_root/'baseline/baseline.xlsx',dtype={'门诊编号':str})
    ref=DrugReference(raw_root/'reference_tables/input_classifier.xlsx')
    cohort=[];baselines=[]
    for pid,row in patients.iterrows():
        admission=baseline_available_hour(baseline,pid,row.injury_time_local)
        eligible=np.isfinite(admission)
        if eligible:
            b=boundary.loc[pid];assert abs(admission-b.admission_hour)<1e-8
        record=dict(patient_id=pid,injury_time_local=row.injury_time_local,
            admission_hour=admission if eligible else np.nan,eligible=bool(eligible),
            exclusion_reason='' if eligible else 'missing_baseline_admission',
            historical_holdout=bool(b.historical_holdout) if eligible else False,
            historical_development_fold=int(b.historical_development_fold) if eligible else -1,
            observation_end_lower_hour=float(b.observation_end_lower_hour) if eligible else np.nan,
            observation_end_upper_hour=float(b.observation_end_upper_hour) if eligible else np.nan,
            observation_end_status=str(b.observation_end_status) if eligible else 'excluded',
            cross_patient_volume_prior_dependency=bool(b.cross_patient_volume_prior_dependency) if eligible else False,
            raw_source_sha256=source.loc[pid,'sha256'])
        cohort.append(record)
        if eligible:
            vals=baseline_consensus(baseline,pid,row.injury_time_local)
            actual=baseline[baseline['门诊编号'].astype(str).str.replace(r'\.0$','',regex=True).eq(pid)]
            for name,value in zip(BASELINE,vals):
                baselines.append(dict(patient_id=pid,field=name,value=value,
                    available_at_hour=admission,source='baseline.xlsx',source_excel_rows=';'.join(str(i+2) for i in actual.index),
                    method='unique_nonmissing_consensus_then_declared_unit_and_range_rules'))
    assert len(cohort)==449 and sum(r['eligible'] for r in cohort)==416
    assert sum(r['historical_holdout'] for r in cohort)==83
    bypatient={r['patient_id']:r for r in cohort}
    save_csv(cohort,out/'cohort/patients.csv');save_csv(baselines,out/'baseline/values.csv')
    injuries={pid:pd.Timestamp(row.injury_time_local) for pid,row in patients.iterrows()}
    doctor,doctor_manifest=doctor_rates(raw_root/'doctor_speed/原始补液速度.zip',injuries)
    doctors=pd.DataFrame(doctor);save_csv(doctors,out/'fluid_events/doctor_observations.csv')
    save_csv(doctor_manifest,out/'provenance/doctor_members.csv')
    groups={};templates={}
    for filename in ['final_v2_event_decisions.csv','final_v2_hourly.csv','final_v2_integration_segments.csv',
                     'final_v2_lot_accounting.csv','final_v2_volume_constraints.csv']:
        table=read(inputs/filename);templates[filename]=table.iloc[:0]
        groups[filename]=dict(tuple(table.groupby('patient_id',sort=False)))
    allobs=[];allnotes=[];allunresolved=[];allroles=[];allreceipts=[];allloads=[];allaux=[]
    allsettings=[];allhourly=[];allevidence=[];allcausal=[];raw_intakes=[];source_checks=[]
    ddg=dict(tuple(doctors.groupby('patient_id')))
    for n,(pid,index) in enumerate(patients.iterrows(),1):
        info=source.loc[pid];path=raw_root/'patient_records'/f'{pid}.csv'
        assert digest(path)==info.sha256,pid
        raw=pd.read_csv(path,encoding=info.encoding,dtype=str,keep_default_na=False)
        fresh=extract_intake(raw,pid,pd.Timestamp(index.injury_time_local),ref)
        # Re-extract every original intake, including excluded patients. Stable
        # intake IDs must retain their raw source name and literal amount.
        own=groups['final_v2_event_decisions.csv'].get(pid,templates['final_v2_event_decisions.csv'])
        lookup={r.event_id:r for r in own.itertuples()}
        for e in fresh:
            if e['event_id'] in lookup:
                previous=lookup[e['event_id']]
                assert str(e['drug_name_raw'])==str(previous.drug_name_raw) or (not e['drug_name_raw'] and pd.isna(previous.drug_name_raw))
        raw_intakes.extend(fresh)
        source_checks.append(dict(patient_id=pid,source_sha256=info.sha256,raw_rows=len(raw),
            original_intake_events_reextracted=len(fresh),eligible=bypatient[pid]['eligible']))
        if not bypatient[pid]['eligible']:continue
        patient=bypatient[pid];obs=list(clinical_records(raw,pid,index.injury_time_local,info.sha256));allobs.extend(obs)
        notes,unresolved,roles,receipts=settings_and_loads(raw,pid,index.injury_time_local,own)
        allnotes.extend(notes);allunresolved.extend(unresolved);allroles.extend(roles);allreceipts.extend(receipts)
        frame=pd.DataFrame(notes)
        if frame.empty:frame=pd.DataFrame(columns=['eligible','hour_from_injury','record_time_local','event_kind','component','rate_ml_h','rate_event_id','unit_inferred','matched_text'])
        d,note_events=setting_source_events(ddg.get(pid,doctors.iloc[:0]),frame,index.injury_time_local)
        # Only finite original positive explicit rows can serve as auxiliary
        # quantity proofs. Unknown-volume renal clues remain uncertain evidence.
        role_proofs=[r for r in roles if r['scope'].startswith('auxiliary') and np.isfinite(r['source_volume_ml']) and r['source_volume_ml']>0]
        known,uncertain,aux=primary_source_load_candidates(own,index.injury_time_local,role_proofs)
        known_ids={e['source_id'] for e in known}
        excluded_ids={e['source_id'] for e in aux}
        receipts=[e for e in receipts if e['source_id'] not in known_ids|excluded_ids]
        known+=receipts;recovered={e['source_id'] for e in receipts};uncertain=[e for e in uncertain if e['source_id'] not in recovered]
        allloads += [dict(e,patient_id=pid,evidence_kind='known') for e in known]
        allloads += [dict(e,patient_id=pid,evidence_kind='unknown_quantity_cue') for e in uncertain]
        allaux += [dict(e,patient_id=pid) for e in aux]
        final_origin=max([73.]+[e['available_hour']+1 for e in note_events]+[e['hour']+1 for e in note_events])
        retrospective=resolve_past_setting_events(d,note_events,final_origin)
        for origin in range(73):
            settings=resolve_past_setting_events(d,note_events,origin)
            for component in COMPONENTS:
                state=past_event_state([e for e in known+settings if e['component']==component],origin)
                assert state['remaining_upper_ml']<=state['recorded_capacity_ml']+1e-8
                allcausal.append(dict(patient_id=pid,origin_hour=origin,component=component,
                    unknown_quantity_cue_count=sum(e['component']==component and 0<=e['hour']<origin and e['available_hour']<origin for e in uncertain),**state))
        original=groups['final_v2_hourly.csv'][pid].set_index('hour_start')
        metadata={str(e.rate_event_id):dict(hour=float(e.hour_from_injury),time_precision=e.time_precision) for e in ddg.get(pid,doctors.iloc[:0]).itertuples()}
        row,setting_pieces,evidence=reference_rows(pid,COMPONENTS,retrospective,known,
            groups['final_v2_integration_segments.csv'].get(pid,templates['final_v2_integration_segments.csv']),
            groups['final_v2_volume_constraints.csv'].get(pid,templates['final_v2_volume_constraints.csv']),
            groups['final_v2_lot_accounting.csv'].get(pid,templates['final_v2_lot_accounting.csv']),original,metadata,patient)
        allhourly.extend(row);allsettings.extend(setting_pieces);allevidence.extend(evidence)
        if n%20==0:print(json.dumps(dict(patients_done=n,observations=len(allobs),seconds=round(time.time()-started,1))),flush=True)
    outputs={
      'observations/records.csv':allobs,'fluid_events/nursing_events.csv':allnotes,
      'evidence/unparsed_nursing_events.csv':allunresolved,'provenance/source_roles.csv':allroles,
      'provenance/text_quantity_receipts.csv':allreceipts,'fluid_events/source_loads.csv':allloads,
      'fluid_events/auxiliary_loads.csv':allaux,'fluid_segments/device_settings.csv':allsettings,
      'hourly/iv_fluid.csv':allhourly,'evidence/reference_intervals.csv':allevidence,
      'causal_history/states.csv':allcausal,'provenance/raw_intake_events.csv':raw_intakes,
      'validation/source_checks.csv':source_checks}
    counts={}
    for name,rows in outputs.items():counts[name]=save_csv(rows,out/name)
    obs=pd.DataFrame(allobs)
    good=obs[obs.eligible].copy()
    # Canonical explicit observations supersede a literal spelling repair in a
    # bin. Unit and qualifier are grouping keys; never mix different scales.
    good['priority']=np.where(good.method.eq('literal_unit_alias'),1,0)
    group=['patient_id','hour','field','unit','qualifier']
    best=good.groupby(group,dropna=False).priority.transform('min');good=good[good.priority.eq(best)]
    bins=good.groupby(group,dropna=False).agg(value=('value','median'),source_records=('observation_id',lambda x:';'.join(x)),record_count=('value','size')).reset_index()
    # Urine rows are collection amounts. Sum within matching units; no division
    # by record gaps and no label that this is physiologic hourly production.
    urine=good.field.str.startswith('urine_recorded_')
    sums=good[urine].groupby(group,dropna=False).value.sum()
    for i,r in bins[bins.field.str.startswith('urine_recorded_')].iterrows():bins.loc[i,'value']=sums.loc[tuple(r[k] for k in group)]
    bins['available_at_hour']=bins.hour+1;counts['observations/hourly.csv']=save_csv(bins,out/'observations/hourly.csv')
    eligible={r['patient_id'] for r in cohort if r['eligible']}
    for old,new in [('final_v2_lot_accounting.csv','fluid_lots/accounting.csv'),
                    ('final_v2_integration_segments.csv','fluid_segments/delivered.csv'),
                    ('final_v2_volume_constraints.csv','evidence/completion_constraints.csv'),
                    ('final_v2_case_decisions.csv','provenance/case_decisions.csv'),
                    ('final_v2_text_case_decisions.csv','provenance/text_decisions.csv'),
                    ('final_v2_oral_decisions.csv','hourly/oral_events.csv'),
                    ('final_v2_event_decisions.csv','fluid_events/adjudicated_events.csv')]:
        df=read(inputs/old);counts[new]=save_csv(df[df.patient_id.isin(eligible)],out/new)
    oral=[]
    for pid in sorted(eligible):
        p=bypatient[pid]
        for r in groups['final_v2_hourly.csv'][pid].itertuples():
            full=r.hour_start>=p['admission_hour'] and r.hour_start+1<=p['observation_end_lower_hour']
            oral.append(dict(patient_id=pid,hour_start=r.hour_start,full_inpatient_hour=full,
                oral_liquid_estimate_ml=r.oral_recorded_liquid_estimate_ml if full else np.nan,
                recorded_oral_event_present=r.oral_has_record,oral_is_imputed=r.oral_is_imputed,
                missing_quantity_events=r.oral_missing_quantity_events,oral_methods=r.oral_methods))
    counts['hourly/oral_liquid.csv']=save_csv(oral,out/'hourly/oral_liquid.csv')
    # Private provenance-only snapshots, never imported as dependencies.
    for name,sha in initial_code.items():assert digest(code_root/name)==sha,'Builder changed during execution: '+name
    shutil.copytree(code_root,out/'provenance/final_builder',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    raw_assets=[]
    for p in [raw_root/'baseline/baseline.xlsx',raw_root/'reference_tables/input_classifier.xlsx',raw_root/'doctor_speed/原始补液速度.zip']:
        raw_assets.append(dict(path=p.relative_to(raw_root).as_posix(),sha256=digest(p)))
    result=dict(version='final_data_v1_20261019',status='candidate_awaiting_content_review',
      source_patients=449,eligible_patients=416,excluded_missing_admission=33,development_patients=333,historical_holdout_patients=83,
      historical_holdout_processed_with_same_data_rules=True,model_scores_computed=False,model_goal_resumed=False,
      independent_clinical_review=False,source_record_clock_is_business_proxy=True,
      raw_assets=raw_assets,input_manifest_sha256=digest(inputs/'input_manifest.json'),
      builder_code_sha256=initial_code,
      tables=counts,quantity_reference_eligible=int(pd.DataFrame(allhourly).quantity_reference_eligible.sum()),
      exact_joint_eligible=int(pd.DataFrame(allhourly).exact_joint_eligible.sum()),
      environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
      elapsed_seconds=round(time.time()-started,2))
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'BUILD_INCOMPLETE').unlink()
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--repository',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('build');a.add_argument('--inputs',type=Path,required=True);a.add_argument('--raw-root',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.repository,a.out)
    else:build(a.inputs,a.raw_root,a.out)


if __name__=='__main__':main()
