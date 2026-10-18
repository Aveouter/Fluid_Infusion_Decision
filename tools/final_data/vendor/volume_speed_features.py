"""Causal recorded-source features for volume/device-setting experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENTS=('crystalloid','colloid','water')
VITALS=('temperature','heart_rate','pulse','respiratory_rate','systolic_bp','diastolic_bp','cvp','spo2','glucose','urine_recorded_ml')
BASELINE=('male','age_years','height_cm','weight_kg','bmi','tbsa_percent','third_degree_percent','deep_second_percent','shallow_second_percent','inhalation_injury')


def baseline_available_hour(baseline,pid,injury):
    rows=baseline[baseline['门诊编号'].astype(str).str.replace(r'\.0$','',regex=True).eq(str(pid))]
    admission=pd.to_datetime(rows['入院时间'],errors='coerce').dropna()
    if not len(admission):return np.inf
    return float((admission.min()-pd.Timestamp(injury)).total_seconds()/3600)


def online_stock_history(additions, doctor_settings):
    """Replay source additions/doctor observations, no later completion anchors.

    Missing initial rates remain NaN for modeled delivery; inventory is only
    a recorded-stock proxy, not an asserted actual physical stock observation.
    Array row h is known only after hour h and is used to forecast h+1.
    """
    rates=np.asarray(doctor_settings,float)
    inventory=0.;last_rate=np.nan;pending=[];result=np.zeros((72,6),float)
    for when,amount in additions:
        if 0<=when<72 and np.isfinite(amount) and amount>0:pending.append((float(when),float(amount)))
    pending.sort();idx=0;last_add=np.nan
    for h in range(72):
        if np.isfinite(rates[h]):last_rate=float(rates[h])
        time=float(h);given=0.;added=0.
        while idx<len(pending) and pending[idx][0]<h+1:
            event_time,amount=pending[idx]
            if np.isfinite(last_rate):
                dose=min(inventory,max(0.,last_rate)*(event_time-time));inventory-=dose;given+=dose
            inventory+=amount;added+=amount;last_add=event_time;time=event_time;idx+=1
        if np.isfinite(last_rate):
            dose=min(inventory,max(0.,last_rate)*(h+1-time));inventory-=dose;given+=dose
        result[h]=[added,inventory,given if np.isfinite(last_rate) else np.nan,
                   last_rate,(h+1-last_add) if np.isfinite(last_add) else np.nan,
                   min(inventory,max(0.,last_rate)) if np.isfinite(last_rate) else np.nan]
    return result


def raw_vitals(raw,injury):
    """Hour-bin ORIGINAL recorded values; bins are consumed strictly in past."""
    out=pd.DataFrame(index=range(72),columns=VITALS,dtype=float)
    clock=pd.to_datetime(raw.FORM_TIME.str.replace('Z','',regex=False),errors='coerce')
    hours=np.floor((clock-pd.Timestamp(injury)).dt.total_seconds()/3600)
    number=pd.to_numeric(raw.NUMBER_VALUE,errors='coerce')
    items=raw['组合项目名称'];nodes=raw['节点名称'];parents=raw['父节点名称']
    rules={
        'temperature':(nodes.eq('体温：'),25,45),
        'heart_rate':(nodes.eq('心率：'),15,300),
        'pulse':(nodes.eq('脉搏：'),15,300),
        'respiratory_rate':(nodes.eq('呼吸：'),3,100),
        'systolic_bp':(nodes.eq('血压：'),30,300),
        'diastolic_bp':(parents.eq('血压：'),10,220),
        'cvp':(items.eq('CVP')|nodes.eq('CVP'),-10,50),
        'spo2':(nodes.eq('氧饱和度：'),20,100),
        'glucose':(parents.eq('血糖：'),1,60)}
    for name,(mask,lo,hi) in rules.items():
        selected=mask&hours.between(0,71)&number.between(lo,hi)
        aggregated=number[selected].groupby(hours[selected]).median()
        out.loc[aggregated.index.astype(int),name]=aggregated.values
    # Name and amount must belong to same form/time/sequence; no cross-row mix.
    discharge=raw[items.isin(['出量:名称','出量:量'])].copy()
    for _,g in discharge.groupby(['FORM_ID','FORM_TIME','INNER_SEQ'],sort=False):
        names=g.loc[g['组合项目名称'].eq('出量:名称'),'STRING_VALUE']
        if not names.str.contains('尿',na=False).any():continue
        values=pd.to_numeric(g.loc[g['组合项目名称'].eq('出量:量'),'NUMBER_VALUE'],errors='coerce').dropna().unique()
        if len(values)!=1 or not 0<=values[0]<=5000:continue
        hour=hours.loc[g.index[0]]
        if pd.notna(hour) and 0<=hour<72:
            old=out.loc[int(hour),'urine_recorded_ml']
            out.loc[int(hour),'urine_recorded_ml']=(old if np.isfinite(old) else 0.)+values[0]
    return out


def baseline_consensus(baseline,pid,injury):
    rows=baseline[baseline['门诊编号'].astype(str).str.replace(r'\.0$','',regex=True).eq(str(pid))]
    result=np.full(len(BASELINE),np.nan)
    def unique_numeric(column):
        vals=pd.to_numeric(rows[column],errors='coerce').dropna().unique()
        return float(vals[0]) if len(vals)==1 else np.nan
    fields=['身高','体重','体重指数','烧伤总面积','合计Ⅲ°','合计深Ⅱ°','合计浅Ⅱ°','吸入性损伤']
    result[2:]=[unique_numeric(c) for c in fields]
    sex=rows['基线_性别'].dropna().astype(str).unique()
    if len(sex)==1 and sex[0] in ['男','女']:result[0]=float(sex[0]=='男')
    births=pd.to_datetime(rows['出生日期'],errors='coerce').dropna().unique()
    if len(births)==1:
        age=(pd.Timestamp(injury)-pd.Timestamp(births[0])).total_seconds()/(365.25*86400)
        if 0<=age<=120:result[1]=age
    if np.isfinite(result[2]) and result[2]<3:result[2]*=100
    if np.isfinite(result[3]) and result[3]>1000:result[3]/=1000
    ranges={2:(40,250),3:(2,250),4:(5,80),5:(0,100),6:(0,100),7:(0,100),8:(0,100),9:(0,1)}
    for j,(lo,hi) in ranges.items():
        if np.isfinite(result[j]) and not lo<=result[j]<=hi:result[j]=np.nan
    return result


def feature_window(sequence,origin,length):
    """Left-padded history; target row origin is NEVER read."""
    if not 0<=origin<=len(sequence) or length<1:raise ValueError('Invalid window')
    start=max(0,origin-length);window=np.full((length,sequence.shape[1]),np.nan)
    if origin>start:window[-(origin-start):]=sequence[start:origin]
    return window
