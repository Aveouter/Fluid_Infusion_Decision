"""Curation adds provenance and justified no-record zero, without changing rates."""
import pandas as pd
from tools.final_data.release import curate


def test_curation_restores_retiming_precision_and_no_oral_event_zero(tmp_path):
    for n in ['cohort','hourly','fluid_events','fluid_segments','provenance']:(tmp_path/n).mkdir()
    pd.DataFrame([dict(patient_id='synthetic',eligible=True,historical_holdout=False,injury_time_local='2020-01-01')]).to_csv(tmp_path/'cohort/patients.csv',index=False)
    pd.DataFrame([dict(patient_id='synthetic',hour_start=0,full_inpatient_hour=True,recorded_oral_event_present=False,oral_liquid_estimate_ml=None)]).to_csv(tmp_path/'hourly/oral_liquid.csv',index=False)
    pd.DataFrame([dict(patient_id='synthetic',eligible=True,hour_from_injury=0,rate_ml_h=100,component='crystalloid',rate_event_id='d',time_precision='hour_estimate')]).to_csv(tmp_path/'fluid_events/doctor_observations.csv',index=False)
    pd.DataFrame([dict(patient_id='synthetic',eligible=True,hour_from_injury=.5,record_time_local='2020-01-01 00:45',event_kind='rate_change',component='crystalloid',rate_ml_h=100,rate_event_id='n',unit_inferred=False,matched_text='调晶体速度100ml/h',time_precision='record_seconds')]).to_csv(tmp_path/'fluid_events/nursing_events.csv',index=False)
    pd.DataFrame([dict(patient_id='synthetic',component='crystalloid',start_hour=.5,end_hour=1,setting_ml_h=100,source_id='d',source='doctor_table',time_precision='')]).to_csv(tmp_path/'fluid_segments/device_settings.csv',index=False)
    pd.DataFrame([dict(patient_id='synthetic',hour_from_injury=0,amount_raw='000100'),dict(patient_id='synthetic',hour_from_injury=73,amount_raw='200')]).to_csv(tmp_path/'provenance/raw_intake_events.csv',index=False)
    first=curate(tmp_path);second=curate(tmp_path)
    assert first==second and first['oral_no_record_zeros_added']==1
    assert pd.read_csv(tmp_path/'hourly/oral_liquid.csv').oral_liquid_estimate_ml.iloc[0]==0
    ref=pd.read_csv(tmp_path/'fluid_events/reference_setting_events.csv')
    row=ref[ref.source_id.eq('d')].iloc[0]
    assert row.hour==.5 and row.original_event_hour==0
    assert row.time_precision=='record_seconds' and row.original_time_precision=='hour_estimate'
    assert row.timing_narrative_source_id=='n'
    assert pd.read_csv(tmp_path/'fluid_segments/device_settings.csv').setting_ml_h.iloc[0]==100
    raw=pd.read_csv(tmp_path/'provenance/raw_intake_events.csv',dtype=str)
    assert len(raw)==1 and raw.amount_raw.iloc[0]=='000100'
