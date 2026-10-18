"""Independent table arithmetic and source checks, never a clinical certification."""
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import io
import json
import zipfile
import numpy as np
import pandas as pd
from .prepare import digest
from .vendor.inventory_event_availability import resolve_past_setting_events
from .vendor.inventory_event_state import past_event_state
from .vendor.inventory_event_sources import setting_source_events


def read(p):return pd.read_csv(p,dtype={'patient_id':str,'form_id':str},low_memory=False)


def validate(package,raw_root,*,preflight=False):
    package=Path(package).resolve();raw_root=Path(raw_root).resolve()
    if not preflight:
        assert not (package/'BUILD_INCOMPLETE').exists()
        meta=json.loads((package/'manifest.json').read_text(encoding='utf-8'))
        assert meta['status']=='candidate_awaiting_content_review'
    cohort=read(package/'cohort/patients.csv');eligible=cohort[cohort.eligible]
    assert len(cohort)==449 and len(eligible)==416 and len(cohort)-len(eligible)==33
    assert eligible.historical_holdout.sum()==83
    hourly=read(package/'hourly/iv_fluid.csv');assert len(hourly)==416*72*3
    assert set(hourly.patient_id)==set(eligible.patient_id)
    assert not hourly.duplicated(['patient_id','hour_start','component']).any()
    assert hourly.groupby(['patient_id','component']).hour_start.nunique().eq(72).all()
    assert hourly.loc[~hourly.full_inpatient_hour,'input_ml'].isna().all()
    assert hourly.loc[~hourly.full_inpatient_hour,'average_actual_input_rate_ml_h'].isna().all()
    assert hourly.input_ml.dropna().ge(-1e-8).all()
    assert not (hourly.exact_joint_eligible&~hourly.quantity_reference_eligible).any()
    assert not (hourly.exact_joint_eligible&~hourly.full_setting_known).any()
    assert not (hourly.exact_joint_eligible&hourly.uncertainty_reasons.notna()).any()
    assert not (hourly.quantity_reference_eligible&hourly.cross_patient_prior_dependency).any()
    np.testing.assert_allclose(hourly.average_actual_input_rate_ml_h,hourly.input_ml,equal_nan=True)
    upstream=read(package/'provenance/build_inputs/final_v2_hourly.csv').set_index(['patient_id','hour_start'])
    # The preserved numeric reference must be identical to the frozen source.
    for c,g in hourly.groupby('component'):
        keys=pd.MultiIndex.from_frame(g[['patient_id','hour_start']])
        np.testing.assert_allclose(g.reference_reconstructed_input_ml,upstream.loc[keys,c+'_input_ml'],equal_nan=True)
    pieces=read(package/'fluid_segments/delivered.csv')
    assert pieces.end_hour.gt(pieces.start_hour).all() and pieces.input_ml.ge(0).all()
    byhour=defaultdict(float)
    for s in pieces.itertuples():
        for h in range(max(0,int(np.floor(s.start_hour))),min(72,int(np.ceil(s.end_hour)))):
            overlap=max(0.,min(s.end_hour,h+1)-max(s.start_hour,h))
            byhour[s.patient_id,h,s.component]+=s.input_ml*overlap/(s.end_hour-s.start_hour)
    differences=[]
    for row in hourly[hourly.reference_reconstructed_input_ml.notna()].itertuples():
        differences.append(abs(row.reference_reconstructed_input_ml-byhour[row.patient_id,row.hour_start,row.component]))
    segment_diff=max(differences,default=0.)
    assert segment_diff<1e-5,segment_diff
    lots=read(package/'fluid_lots/accounting.csv')
    ledger_diff=float(np.max(np.abs(lots.original_volume_ml-lots.cancelled_ml-lots.net_ml)))
    balance_diff=float(np.max(np.abs(lots.net_ml-lots.modeled_consumed_ml-lots.remaining_at_episode_end_ml)))
    assert max(ledger_diff,balance_diff)<1e-5
    observations=read(package/'observations/records.csv')
    assert not observations.observation_id.duplicated().any()
    assert observations.hour.between(0,71).all()
    assert observations.available_at_hour.eq(observations.hour+1).all()
    assert not observations.loc[observations.reason.eq('unresolved_three_digit_K_source'),'eligible'].any()
    assert not observations.loc[observations.field.str.contains('residual_urine'),'field'].str.startswith('urine_recorded_').any()
    states=read(package/'causal_history/states.csv')
    assert len(states)==416*73*3
    assert not states.duplicated(['patient_id','origin_hour','component']).any()
    assert (states.remaining_lower_ml>=-1e-8).all()
    assert (states.remaining_upper_ml>=states.remaining_lower_ml-1e-8).all()
    assert (states.remaining_upper_ml<=states.recorded_capacity_ml+1e-8).all()
    assert not {'reference_reconstructed_input_ml','exact_joint_eligible','completion_total','net_ml'}&set(states.columns)
    original=read(package/'provenance/build_inputs/raw_source_manifest.csv')
    for r in original.itertuples():assert digest(raw_root/'patient_records'/f'{r.patient_id}.csv')==r.sha256
    doctors=read(package/'fluid_events/doctor_observations.csv');doctor_checked=0
    with zipfile.ZipFile(raw_root/'doctor_speed/原始补液速度.zip') as archive:
        for member,rows in doctors.groupby('source_member'):
            blob=archive.read(member)
            try:raw=pd.read_csv(io.BytesIO(blob),encoding='gb18030',dtype=str,keep_default_na=False)
            except UnicodeDecodeError:raw=pd.read_csv(io.BytesIO(blob),encoding='utf-8-sig',dtype=str,keep_default_na=False)
            names={'crystalloid':'晶体速度','colloid':'胶体速度','water':'水速度'}
            for r in rows.itertuples():
                value=pd.to_numeric(raw.iloc[int(r.source_record_number)-1][names[r.component]],errors='coerce')
                if r.eligible:assert abs(float(value)-r.rate_ml_h)<1e-8
                doctor_checked+=1
    notes=read(package/'fluid_events/nursing_events.csv');loads=read(package/'fluid_events/source_loads.csv')
    dg=dict(tuple(doctors.groupby('patient_id')));ng=dict(tuple(notes.groupby('patient_id')))
    sg=dict(tuple(states.groupby('patient_id')));lg=dict(tuple(loads.groupby('patient_id')))
    real_checks=0;future_checks=0
    # Fixed patient positions and origins, not selected by model or error.
    for patient in eligible.iloc[::31].itertuples():
        pid=patient.patient_id
        dd,nn=setting_source_events(dg.get(pid,doctors.iloc[:0]),ng.get(pid,notes.iloc[:0]),patient.injury_time_local)
        own=lg.get(pid,loads.iloc[:0]);known=own[own.evidence_kind.eq('known')].to_dict('records')
        for h in [1,24,48,71]:
            se=resolve_past_setting_events(dd,nn,h)
            for c in ['crystalloid','colloid','water']:
                ee=[e for e in known+se if e['component']==c]
                actual=past_event_state(ee,h)
                after=past_event_state(ee+[dict(hour=h+.2,available_hour=h+.3,kind='completion',value=-999)],h)
                assert actual==after;future_checks+=1
                saved=sg[pid];saved=saved[saved.origin_hour.eq(h)&saved.component.eq(c)].iloc[0]
                for field in ['remaining_lower_ml','remaining_upper_ml','remaining_midpoint_estimate_ml','recorded_capacity_ml']:
                    assert abs(actual[field]-saved[field])<1e-6,(field,pid,h,c)
                real_checks+=1
    summary=dict(status='preflight_passed_not_release' if preflight else 'passed',source_patients=449,eligible_patients=416,excluded_patients=33,
        hourly_iv_rows=len(hourly),causal_state_rows=len(states),clinical_source_observations=len(observations),
        source_raw_hashes_verified=449,doctor_observations_verified=doctor_checked,
        preserved_reference_max_difference_ml=0.,independent_segment_hour_integral_max_difference_ml=segment_diff,
        lot_original_net_max_difference_ml=ledger_diff,lot_net_consumed_remaining_max_difference_ml=balance_diff,
        real_past_state_checks=real_checks,future_completion_poison_checks=future_checks,
        no_model_training_or_scores=True,independent_clinical_review=False,
        support_by_component=hourly.groupby('component')[['full_inpatient_hour','quantity_reference_eligible','setting_reference_eligible','exact_joint_eligible']].sum().astype(int).to_dict('index'),
        input_value_status_counts=hourly.input_value_status.value_counts().to_dict(),
        clinical_field_counts=observations[observations.eligible].field.value_counts().to_dict())
    target='preflight_aggregate.json' if preflight else 'aggregate.json'
    (package/'validation'/target).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,dict)},ensure_ascii=False),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--preflight',action='store_true',help='Diagnostic tables only; never emits a release validation result')
    a=p.parse_args();validate(a.package,a.raw_root,preflight=a.preflight)


if __name__=='__main__':main()
