"""Unscaled table-to-array adapters with a predictor/target file boundary."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from .prepare import digest

COMPONENTS=('crystalloid','colloid','water')
BASELINE=('male','age_years','height_cm','weight_kg','bmi','tbsa_percent','third_degree_percent','deep_second_percent','shallow_second_percent','inhalation_injury')
STATE=('recorded_capacity_ml','remaining_lower_ml','remaining_upper_ml','remaining_midpoint_estimate_ml',
       'last_known_setting_ml_h','unknown_quantity_cue_count','unknown_flow_exposure_hours','last_setting_unit_inferred')


def predictors(package):
    """No hourly reference, completed lot, uncertainty flag or target is read."""
    p=Path(package)
    cohort=pd.read_csv(p/'cohort/patients.csv',dtype={'patient_id':str},usecols=['patient_id','eligible'])
    ids=cohort.loc[cohort.eligible,'patient_id'].tolist();index={v:i for i,v in enumerate(ids)}
    base=pd.read_csv(p/'baseline/values.csv',dtype={'patient_id':str},usecols=['patient_id','field','value','available_at_hour'])
    baseline=np.full((len(ids),len(BASELINE)),np.nan);available=np.full(len(ids),np.nan)
    for r in base.itertuples():
        baseline[index[r.patient_id],BASELINE.index(r.field)]=r.value
        available[index[r.patient_id]]=r.available_at_hour
    clinical=pd.read_csv(p/'observations/hourly.csv',dtype={'patient_id':str},usecols=['patient_id','hour','field','unit','qualifier','value','available_at_hour'])
    assert clinical.available_at_hour.eq(clinical.hour+1).all()
    names=sorted({(r.field,r.unit,r.qualifier) for r in clinical.itertuples()})
    columns={k:i for i,k in enumerate(names)};values=np.full((len(ids),72,len(names)),np.nan)
    for r in clinical.itertuples():values[index[r.patient_id],int(r.hour),columns[r.field,r.unit,r.qualifier]]=r.value
    state=pd.read_csv(p/'causal_history/states.csv',dtype={'patient_id':str},usecols=['patient_id','origin_hour','component',*STATE])
    states=np.full((len(ids),73,len(COMPONENTS)*len(STATE)),np.nan)
    for r in state.itertuples():
        c=COMPONENTS.index(r.component)
        states[index[r.patient_id],int(r.origin_hour),c*len(STATE):(c+1)*len(STATE)]=[getattr(r,k) for k in STATE]
    roles=[]
    for field,unit,qualifier in names:
        roles.append('treatment_documentation' if 'pump' in field else 'recorded_amount_unverified_unit' if 'urine' in field and 'unspecified' in field else 'clinical_recorded_observation')
    return dict(patient_ids=np.array(ids),baseline=baseline,baseline_names=np.array(BASELINE),baseline_available_hour=available,
        clinical_hourly=values,clinical_names=np.array(['|'.join(n) for n in names]),clinical_roles=np.array(roles),
        causal_state_at_origin=states,causal_state_names=np.array([c+'|'+k for c in COMPONENTS for k in STATE]))


def targets(package,ids):
    p=Path(package);index={str(v):i for i,v in enumerate(ids)}
    frame=pd.read_csv(p/'hourly/iv_fluid.csv',dtype={'patient_id':str})
    names=['input_ml','end_setting_ml_h','average_actual_input_rate_ml_h','quantity_reference_eligible','setting_reference_eligible','exact_joint_eligible']
    arrays={k:np.zeros((len(index),72,3),bool) if k.endswith('_eligible') else np.full((len(index),72,3),np.nan) for k in names}
    for r in frame.itertuples():
        pos=(index[r.patient_id],int(r.hour_start),COMPONENTS.index(r.component))
        for k in names:arrays[k][pos]=getattr(r,k)
    arrays['recorded_source_added_ml']=np.zeros((len(index),72,3))
    arrays['recorded_source_added_eligible']=np.zeros((len(index),72,3),bool)
    for r in frame.itertuples():
        pos=(index[r.patient_id],int(r.hour_start),COMPONENTS.index(r.component))
        arrays['recorded_source_added_eligible'][pos]=bool(r.full_inpatient_hour and 'interval_allocation_without_exact_flow' not in str(r.uncertainty_reasons))
    events=pd.read_csv(p/'fluid_events/source_loads.csv',dtype={'patient_id':str})
    for r in events.itertuples():
        if r.patient_id not in index or not 0<=r.hour<72:continue
        pos=(index[r.patient_id],int(r.hour),COMPONENTS.index(r.component))
        if r.evidence_kind=='known':arrays['recorded_source_added_ml'][pos]+=r.value
        else:arrays['recorded_source_added_eligible'][pos]=False
    return dict(patient_ids=np.array(ids),**arrays)


def sample(source,patient,origin,history=24):
    """Unfilled past bins and as-of state only; query does not need a target."""
    if not 0<=origin<=71 or history<1:raise ValueError('origin0..71 and positive history required')
    x=source['clinical_hourly'];result=np.full((history,x.shape[-1]),np.nan)
    fluid=np.full((history,source['causal_state_at_origin'].shape[-1]),np.nan)
    start=max(0,origin-history)
    if origin>start:
        result[-(origin-start):]=x[patient,start:origin]
        fluid[-(origin-start):]=source['causal_state_at_origin'][patient,start+1:origin+1]
    static=source['baseline'][patient].copy()
    if not np.isfinite(source['baseline_available_hour'][patient]) or origin<source['baseline_available_hour'][patient]:static[:]=np.nan
    return dict(clinical_history=result,fluid_history=fluid,baseline=static,causal_state=source['causal_state_at_origin'][patient,origin].copy())


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    source_manifest=json.loads((a.package/'manifest.json').read_text(encoding='utf-8'))
    if source_manifest['status']!='frozen_research_data_version':raise ValueError('CLI export requires the sealed Final package')
    assert not a.out.exists();a.out.mkdir(parents=True)
    x=predictors(a.package);y=targets(a.package,x['patient_ids'])
    np.savez_compressed(a.out/'predictors.npz',**x);np.savez_compressed(a.out/'targets.npz',**y)
    manifest=dict(source_package_manifest_sha256=digest(a.package/'manifest.json'),predictor_file_sha256=digest(a.out/'predictors.npz'),
      target_file_sha256=digest(a.out/'targets.npz'),patients=len(x['patient_ids']),clinical_channels=len(x['clinical_names']),
      causal_state_channels=len(x['causal_state_names']),normalization=False,imputation=False,model_fitted=False,
      predictor_target_files_separate=True,reference_masks_not_predictors=True)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print(json.dumps(manifest))


if __name__=='__main__':main()
