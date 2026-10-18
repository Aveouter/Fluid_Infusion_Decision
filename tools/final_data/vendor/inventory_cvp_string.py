"""Recover literal CVP scalar records only; never infer units or ranges."""
import re,unicodedata
import numpy as np
import pandas as pd
from tools.final_data.vendor.inventory_numeric_vitals import canonical_name


def literal_scalar(value):
    text=unicodedata.normalize('NFKC',str(value)).strip()
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)',text):return None
    number=float(text)
    return number if np.isfinite(number) else None


def cvp_string_records(raw,injury):
    named=raw['组合项目名称'].map(canonical_name).eq('CVP')|raw['节点名称'].map(canonical_name).eq('CVP')
    blank=raw.NUMBER_VALUE.str.strip().eq('')
    clock=pd.to_datetime(raw.FORM_TIME.str.replace('Z','',regex=False),errors='coerce')
    hour=(clock-pd.Timestamp(injury)).dt.total_seconds()/3600
    result=[]
    for i in np.flatnonzero(named&blank&hour.between(0,72,inclusive='left')):
        value=literal_scalar(raw.STRING_VALUE.iloc[i])
        if value is not None and -10<=value<=50:
            result.append(dict(source_record_number=int(i)+2,record_hour=float(hour.iloc[i]),
                hour=int(np.floor(hour.iloc[i])),value=value))
    return pd.DataFrame(result,columns=['source_record_number','record_hour','hour','value'])
