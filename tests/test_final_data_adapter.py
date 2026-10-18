"""The optional consumer adapter must never open retrospective labels."""
import numpy as np
import pandas as pd
from tools.final_data.adapter import predictors,sample,BASELINE,STATE


def test_predictor_builder_works_without_any_reference_table(tmp_path):
    for folder in ['cohort','baseline','observations','causal_history']:(tmp_path/folder).mkdir()
    pd.DataFrame([dict(patient_id='synthetic',eligible=True)]).to_csv(tmp_path/'cohort/patients.csv',index=False)
    pd.DataFrame([dict(patient_id='synthetic',field=f,value=1.,available_at_hour=.5) for f in BASELINE]).to_csv(tmp_path/'baseline/values.csv',index=False)
    pd.DataFrame([dict(patient_id='synthetic',hour=0,field='heart_rate',unit='beats/min',qualifier='recorded',value=90.,available_at_hour=1)]).to_csv(tmp_path/'observations/hourly.csv',index=False)
    rows=[]
    for h in range(73):
        for c in ['crystalloid','colloid','water']:rows.append(dict(patient_id='synthetic',origin_hour=h,component=c,**{f:0. for f in STATE}))
    pd.DataFrame(rows).to_csv(tmp_path/'causal_history/states.csv',index=False)
    source=predictors(tmp_path)
    assert source['clinical_hourly'].shape==(1,72,1)
    assert sample(source,0,1,history=2)['clinical_history'][-1,0]==90.
    assert np.isnan(sample(source,0,0)['baseline']).all()


def test_sample_ignores_query_and_future_clinical_bins():
    source=dict(clinical_hourly=np.arange(72,dtype=float)[None,:,None],baseline=np.array([[1.,2.]]),
        baseline_available_hour=np.array([0.]),causal_state_at_origin=np.arange(73,dtype=float)[None,:,None])
    expected=sample(source,0,12,history=6)
    source['clinical_hourly'][:,12:]=1e12
    source['causal_state_at_origin'][:,13:]=-1e12
    actual=sample(source,0,12,history=6)
    for key in expected:np.testing.assert_equal(actual[key],expected[key])
