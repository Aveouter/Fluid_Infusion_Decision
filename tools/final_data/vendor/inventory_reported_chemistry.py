"""Explicit reported chemistry, preserving censoring instead of invented points."""
import re,unicodedata

MMOL=r'(?:mmol\s*/\s*l|毫摩尔\s*/\s*升)(?![A-Za-z0-9/])'
RULES={
 'glucose':(r'血糖(?:值)?|(?<![A-Za-z])Glu(?:cose)?(?![A-Za-z])',MMOL,'mmol/L'),
 'sodium':(r'血钠|钠离子|(?<![A-Za-z])Na\+*(?![A-Za-z])',MMOL,'mmol/L'),
 'potassium':(r'血钾|钾离子|(?<![A-Za-z])K\+*(?![A-Za-z])',MMOL,'mmol/L'),
 'calcium':(r'血钙|钙离子|(?<![A-Za-z])Ca\+*(?![A-Za-z])',MMOL,'mmol/L'),
 'hematocrit':(r'红细胞压积|红细胞比容|(?<![A-Za-z])Hct(?![A-Za-z])',r'(?:%|百分比)','%'),
 'total_hemoglobin':(r'总血红蛋白(?:\s*tHb)?|(?<![A-Za-z])tHb(?![A-Za-z])',r'g\s*/\s*l(?![A-Za-z0-9/])','g/L'),
 'lactate':(r'乳酸(?:值)?|(?<![A-Za-z])Lac(?:tate)?(?![A-Za-z])',MMOL,'mmol/L')}
EXTRA_NAMES=('reported_glucose_mmol_l','reported_sodium_mmol_l','reported_potassium_mmol_l',
    'reported_calcium_mmol_l','reported_hematocrit_percent','reported_total_hemoglobin_g_l',
    'reported_hematocrit_lower_bound_percent','reported_hematocrit_upper_bound_percent',
    'reported_lactate_lower_bound_mmol_l','reported_lactate_upper_bound_mmol_l')


def reported_chemistry(text):
    text=unicodedata.normalize('NFKC',str(text)).strip();out=[]
    for kind,(label,unit,canonical) in RULES.items():
        pattern=r'(?:'+label+r')\)?[\s:：=]*(?:为|测得|结果为|结果示)?\s*(?P<op>>=|<=|>|<|≥|≤)?\s*(?P<value>[+-]?\d+(?:\.\d+)?)\s*[xX*]?\s*'+unit
        for m in re.finditer(pattern,text,re.I):
            before=text[max(0,m.start()-80):m.start()];sentence=re.split(r'[。；;\n]',before)[-1];clause=re.split(r'[，,]',sentence)[-1]
            suffix=text[m.end():m.end()+8];reason='explicit_reported_value'
            if re.search(r'若|如果|假如',sentence) or re.search(r'目标|维持在|保持在|控制在|要求|预计|计划|建议|拟|(?:当|一旦)$',clause):reason='conditional_or_target'
            elif re.search(r'昨日|昨天|前日|既往|平素|上次|此前|入院时',clause):reason='historical_reference'
            elif re.search(r'家属|母亲|父亲|陪护|其母|其父|他人|同室',clause):reason='other_person_context'
            elif re.search(r'未测|未查|未监测|未见|参考|正常值|正常范围',clause):reason='not_observed_or_reference'
            elif re.search(r'配制|配液|溶液|浓度|输注|注射|添加',clause):reason='treatment_concentration_context'
            elif re.match(r'\s*(?:以下|以上|以内|以外|左右|上下|时|则|至|~|—)',suffix):reason='conditional_bound_or_approximation'
            value=float(m.group('value'));op=m.group('op') or '='
            if value<0 or (kind=='hematocrit' and value>100) or (kind=='glucose' and not 1<=value<=60):reason='outside_declared_domain'
            qualifier='lower' if op in ['>','>=','≥'] else 'upper' if op in ['<','<=','≤'] else 'exact'
            out.append(dict(field=kind,value=value,unit=canonical,operator=op,qualifier=qualifier,
                eligible=reason=='explicit_reported_value',reason=reason,raw_match=m.group(),span_start=m.start(),span_end=m.end()))
    return sorted(out,key=lambda r:(r['span_start'],r['field']))


def feature_name(record):
    field=record['field'];q=record['qualifier']
    if q=='exact':
        return dict(glucose=EXTRA_NAMES[0],sodium=EXTRA_NAMES[1],potassium=EXTRA_NAMES[2],calcium=EXTRA_NAMES[3],hematocrit=EXTRA_NAMES[4],total_hemoglobin=EXTRA_NAMES[5]).get(field)
    if field=='hematocrit':return EXTRA_NAMES[6 if q=='lower' else 7]
    if field=='lactate':return EXTRA_NAMES[8 if q=='lower' else 9]
    return None
