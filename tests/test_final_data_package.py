"""Synthetic extraction contracts, independent of all private patient data."""
import numpy as np
import pandas as pd
from tools.final_data.extract import clinical_records
from tools.final_data.vendor.inventory_event_state import past_event_state
from tools.final_data.vendor.inventory_event_clock import hour_setting_segments,summarize_setting_hour
from tools.final_data.vendor.inventory_event_availability import resolve_past_setting_events
from tools.final_data.vendor.inventory_text_quantity import resolve_record
from tools.final_data.vendor.fluid_reconstruction_sources import event_clock
from tools.final_data.quality import conditional_delivery


def test_bottle_addition_is_not_immediate_delivered_volume():
    events=[dict(hour=0,available_hour=0,kind='addition',value=500),
            dict(hour=0,available_hour=0,kind='setting',value=80)]
    state=past_event_state(events,2)
    assert state['recorded_capacity_ml']==500
    assert state['remaining_point_ml']==340


def test_stock_and_unknown_initial_setting_stay_interval_not_zero_rate():
    state=past_event_state([dict(hour=0,available_hour=0,kind='addition',value=500)],2)
    assert state['last_known_setting_ml_h'] is None
    assert (state['remaining_lower_ml'],state['remaining_upper_ml'])==(0,500)


def test_future_payload_and_late_record_do_not_enter_causal_state():
    events=[dict(hour=0,available_hour=0,kind='addition',value=500)]
    expected=past_event_state(events,1)
    events += [dict(hour=2,available_hour=2,kind='completion',value=-999),
               dict(hour=.5,available_hour=3,kind='addition',value=999)]
    assert past_event_state(events,1)==expected


def test_hour_boundary_event_belongs_to_following_hour_and_internal_change_integrates():
    changes=[dict(hour=0,rate=100),dict(hour=.5,rate=200),dict(hour=1,rate=500)]
    a=summarize_setting_hour(hour_setting_segments(changes,0))
    assert a['start_setting_ml_h']==100 and a['end_setting_ml_h']==200
    assert a['setting_integral_ml']==150
    assert summarize_setting_hour(hour_setting_segments(changes,1))['start_setting_ml_h']==500


def test_nursing_cannot_silently_override_known_doctor_numeric_setting():
    doctors=[dict(hour=0,available_hour=0,value=100,component='crystalloid',source_id='d')]
    notes=[dict(hour=.5,available_hour=.5,value=250,kind='setting',component='crystalloid',source_id='n',explicit_change=True)]
    assert resolve_past_setting_events(doctors,notes,1)[-1]['value']==100


def test_quantity_receipt_requires_exact_clock_and_single_family_peer():
    notes={('form',10):['予输入血浆200ml']}
    assert resolve_record('血浆','form',10,notes,1)['volume_ml']==200
    assert resolve_record('血浆','form',11,notes,1)['method']=='unresolved'
    assert resolve_record('血浆','form',10,notes,2)['conflict']


def test_urine_without_unit_is_not_promoted_to_ml_per_hour():
    raw=pd.DataFrame([
        dict(FORM_TIME='2020-01-01T00:30:00Z',FORM_ID='f',INNER_SEQ='1',组合项目名称='出量:名称',节点名称='',父节点名称='',NUMBER_VALUE='',STRING_VALUE='尿量'),
        dict(FORM_TIME='2020-01-01T00:30:00Z',FORM_ID='f',INNER_SEQ='1',组合项目名称='出量:量',节点名称='',父节点名称='',NUMBER_VALUE='300',STRING_VALUE='')])
    rows=list(clinical_records(raw,'synthetic','2020-01-01','synthetic-hash'))
    assert len(rows)==1
    assert rows[0]['field']=='urine_recorded_amount_unspecified_unit'
    assert rows[0]['value']==300 and rows[0]['unit']=='unspecified'


def test_three_digit_potassium_is_retained_but_not_eligible_and_bounds_stay_bounds():
    raw=pd.DataFrame([dict(FORM_TIME='2020-01-01T00:30:00Z',FORM_ID='f',INNER_SEQ='1',组合项目名称='病情观测及措施',节点名称='',父节点名称='',NUMBER_VALUE='',STRING_VALUE='血钾138mmol/L，乳酸>4mmol/L')])
    rows=list(clinical_records(raw,'synthetic','2020-01-01','synthetic-hash'))
    potassium=[r for r in rows if r['field']=='reported_potassium_mmol_l'][0]
    assert potassium['value']==138 and not potassium['eligible']
    bound=[r for r in rows if r['field']=='reported_lactate_lower_bound_mmol_l'][0]
    assert bound['value']==4 and bound['qualifier']=='lower'


def test_oxygen_saturation_text_does_not_become_a_clock():
    at,source,precision=event_clock('SO2:100%，晶体速度100ml/h',15,pd.Timestamp('2020-01-01 12:00'))
    assert at==pd.Timestamp('2020-01-01 12:00')


def test_residual_urine_is_not_new_hourly_production():
    raw=pd.DataFrame([
        dict(FORM_TIME='2020-01-01T00:30:00Z',FORM_ID='f',INNER_SEQ='1',组合项目名称='出量:名称',节点名称='',父节点名称='',NUMBER_VALUE='',STRING_VALUE='残余尿'),
        dict(FORM_TIME='2020-01-01T00:30:00Z',FORM_ID='f',INNER_SEQ='1',组合项目名称='出量:量',节点名称='',父节点名称='',NUMBER_VALUE='120',STRING_VALUE='')])
    rows=list(clinical_records(raw,'synthetic','2020-01-01','synthetic-hash'))
    assert rows[0]['field']=='residual_urine_recorded_amount_unspecified_unit'


def test_conditional_delivery_caps_at_stock_and_handles_later_arrival():
    pieces=[dict(start_hour=0,end_hour=1,setting_ml_h=100)]
    assert conditional_delivery(20,[(.5,30)],pieces)==50
    assert conditional_delivery(20,[],pieces)==20


def test_unknown_setting_has_no_conditional_point_delivery():
    assert conditional_delivery(500,[],[dict(start_hour=0,end_hour=1,setting_ml_h=None)]) is None
