"""Fixed text canonicalization of the existing nine numeric vital rules."""
import unicodedata
import numpy as np
import pandas as pd
from tools.final_data.vendor.volume_speed_features import VITALS


def canonical_name(value):
    return unicodedata.normalize('NFKC',str(value)).strip().replace(':','：')


def normalized_numeric_vitals(raw,injury):
    names={k:raw[k].map(canonical_name) for k in ['组合项目名称','节点名称','父节点名称']}
    number=pd.to_numeric(raw.NUMBER_VALUE.map(lambda v:unicodedata.normalize('NFKC',str(v)).strip()),errors='coerce')
    clock=pd.to_datetime(raw.FORM_TIME.str.replace('Z','',regex=False),errors='coerce')
    hour=np.floor((clock-pd.Timestamp(injury)).dt.total_seconds()/3600)
    item,node,parent=[names[k] for k in ['组合项目名称','节点名称','父节点名称']]
    old_number=pd.to_numeric(raw.NUMBER_VALUE,errors='coerce')
    old_item,old_node,old_parent=[raw[k] for k in ['组合项目名称','节点名称','父节点名称']]
    old_fields=[old_node.eq('体温：'),old_node.eq('心率：'),old_node.eq('脉搏：'),old_node.eq('呼吸：'),
        old_node.eq('血压：'),old_parent.eq('血压：'),old_item.eq('CVP')|old_node.eq('CVP'),
        old_node.eq('氧饱和度：'),old_parent.eq('血糖：')]
    rules=[(node.eq('体温：'),25,45),(node.eq('心率：'),15,300),(node.eq('脉搏：'),15,300),
        (node.eq('呼吸：'),3,100),(node.eq('血压：'),30,300),(parent.eq('血压：'),10,220),
        (item.eq('CVP')|node.eq('CVP'),-10,50),(node.eq('氧饱和度：'),20,100),(parent.eq('血糖：'),1,60)]
    result=pd.DataFrame(index=range(72),columns=VITALS[:9],dtype=float)
    accepted=[]
    for c,(field,lo,hi) in enumerate(rules):
        valid=field&hour.between(0,71)&number.between(lo,hi)
        values=number[valid].groupby(hour[valid]).median()
        result.loc[values.index.astype(int),VITALS[c]]=values.values
        accepted.append(pd.DataFrame(dict(source_record_number=np.flatnonzero(valid)+2,
            hour=hour[valid].astype(int).to_numpy(),component=c,value=number[valid].to_numpy(),
            legacy_row_accepted=(old_fields[c]&old_number.between(lo,hi))[valid].to_numpy())))
    return result,pd.concat(accepted,ignore_index=True)
