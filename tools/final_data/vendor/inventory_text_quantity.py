"""Conservative current-record volume mentions, not adjudicated new stock."""
import re
import unicodedata
import numpy as np

FAMILIES={
    'red_cells':r'红细胞(?:混悬液|悬液|悬浮液)?',
    'plasma':r'(?<!代)血浆',
    'albumin':r'(?:人血)?白蛋白',
    'dextran':r'右旋糖酐|代血浆',
    'bicarbonate':r'碳酸氢钠',
    'saline':r'氯化钠',
    'ringer':r'林格(?:氏液)?|平衡(?:盐|液)',
    'glucose':r'葡萄糖(?!酸钙)|糖水',
}
VOLUME=re.compile(r'(?P<number>\d+(?:\.\d+)?)\s*(?:ml|毫升)(?!\s*(?:[/／]|每|[·*]?\s*h\b))',re.I)
DENY=re.compile(r'拟|计划|准备|备血|申请|待|建议|如果|若|目标|尚余|剩余|未予|未输|未开始|暂不|停止|取消|不予|不再|无需|禁止|勿|拒绝|累计|总计|昨日|昨天|昨晚|前日|此前|已输')
ACTUAL=re.compile(r'予|给予|开始|输入|输注')


def normalize(text):
    return unicodedata.normalize('NFKC',str(text or ''))


def family(name):
    text=normalize(name)
    for kind in ['dextran','albumin','red_cells','plasma','bicarbonate','saline','ringer','glucose']:
        if re.search(FAMILIES[kind],text):
            return kind
    return None


def name_volume(name):
    """Trailing clinical mL outside parentheses; only an uncertain source cue."""
    text=normalize(name)
    if re.search(r'余|剩|已输|输毕|输完|取消|停止',text):
        return None
    tail=re.sub(r'\([^)]*\)','',text)
    matches=list(VOLUME.finditer(tail))
    if len(matches)!=1 or tail[matches[0].end():].strip('★☆!* \t'):
        return None
    quantity=float(matches[0]['number'])
    return quantity if 0<quantity<=10000 else None


def actual_note_volumes(text,kind):
    """Require fluid-specific adjacency; never borrow urine, dose or rate units."""
    if kind not in FAMILIES:
        return []
    result=[]
    for clause in re.split(r'[，,。；;\n]',normalize(text)):
        if DENY.search(clause):
            continue
        for match in re.finditer(FAMILIES[kind],clause):
            if not ACTUAL.search(clause[max(0,match.start()-30):match.start()]):
                continue
            suffix=clause[match.end():]
            prefix=re.match(r'\s*(?:(?:约|共|总量|为|计|量为|:)\s*)?',suffix)
            quantity=VOLUME.match(suffix,prefix.end())
            if quantity:
                value=float(quantity['number'])
                if 0<value<=10000:
                    result.append(value)
    return result


def resolve(name,notes):
    """Caller must supply ONLY same-form, exact same-record-clock notes."""
    candidate=name_volume(name)
    values={v for text in notes for v in actual_note_volumes(text,family(name))}
    if len(values)>1:
        return dict(volume_ml=np.nan,method='conflicting_actual_mentions',conflict=True)
    if values:
        quantity=next(iter(values))
        return dict(volume_ml=quantity,method='same_clock_form_actual_mention',conflict=candidate is not None and candidate!=quantity)
    if candidate is not None:
        return dict(volume_ml=candidate,method='name_volume_candidate',conflict=False)
    return dict(volume_ml=np.nan,method='unresolved',conflict=False)


def resolve_record(name,form_id,clock_token,notes_by_key,peer_count=1):
    """Exact-clock binding avoids future edits and duplicated same-family rows."""
    if peer_count!=1:
        return dict(volume_ml=np.nan,method='ambiguous_same_family_peer',conflict=True)
    return resolve(name,notes_by_key.get((str(form_id),clock_token),()))


def features(records,patient_ids,hours,admissions):
    """39 past-only mention statistics; absence means no cue, not actual zero."""
    result=np.full((len(hours),39),np.nan)
    for i,(pid,hour,admission) in enumerate(zip(patient_ids,hours,admissions)):
        if not np.isfinite(admission) or hour<admission:
            continue
        for c in range(3):
            # columns: record clock, candidate volume, method(0unresolved/1name/2actual), conflict
            source=records.get((str(pid),c),np.empty((0,4)))
            past=source[(source[:,0]>=max(0.,admission))&(source[:,0]<hour)]
            valid=past[np.isfinite(past[:,1])]
            start=c*13
            result[i,start:start+3]=[(valid[:,0]>=hour-w).sum() for w in (1,6,24)]
            result[i,start+3:start+6]=[valid[valid[:,0]>=hour-w,1].sum() for w in (1,6,24)]
            result[i,start+8]=valid[:,1].sum()
            if len(valid):
                last_clock=valid[:,0].max()
                result[i,start+6:start+8]=[hour-last_clock,valid[valid[:,0]==last_clock,1].sum()]
            recent=past[past[:,0]>=hour-24]
            result[i,start+9:start+13]=[(recent[:,2]==2).sum(),(recent[:,2]==1).sum(),
                (~np.isfinite(recent[:,1])).sum(),recent[:,3].sum()]
    return result
