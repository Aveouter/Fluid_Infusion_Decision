"""Conservative text observations and sparse actions; no invented negatives."""
import re
import unicodedata

NUMBER=r'([+-]?\d+(?:\.\d+)?)'
MMOL=r'\s*(mmol\s*/\s*[lL]|毫摩尔\s*/\s*升)'
RULES={
    'lactate_mmol_l':(r'(?:乳酸(?:值)?|(?<![A-Za-z])lac(?:tate)?)[\s:：=]*'+NUMBER+MMOL,'mmol/L'),
    'base_excess_mmol_l':(r'(?<![A-Za-z])BE[\s:：=]*'+NUMBER+MMOL,'mmol/L'),
    'bicarbonate_mmol_l':(r'HCO3-?[\s:：=]*'+NUMBER+MMOL,'mmol/L'),
    'blood_ph':(r'(?<![A-Za-z])pH[\s:：=]*'+NUMBER,'unitless'),
    'paco2_mmhg':(r'(?<![A-Za-z])P(?:a)?CO2[\s:：=]*'+NUMBER+r'\s*(mmHg|毫米汞柱)','mmHg'),
    'pao2_mmhg':(r'(?<![A-Za-z])P(?:a)?O2[\s:：=]*'+NUMBER+r'\s*(mmHg|毫米汞柱)','mmHg'),
    'hemoglobin_g_l':(r'(?:血红蛋白|(?<![A-Za-z])H[bB])[\s:：=]*'+NUMBER+r'\s*(g\s*/\s*[lL]|克/升|g\s*/\s*d[lL])','g/L'),
    'creatinine_umol_l':(r'(?:肌酐|(?<![A-Za-z])Cr)[\s:：=]*'+NUMBER+r'\s*([uμµ]mol\s*/\s*[lL]|微摩尔/升)','umol/L'),
    'cardiac_index_l_min_m2':(r'(?:PCCI|(?<![A-Za-z])CI)[\s:：=]*'+NUMBER+r'\s*(L\s*/\s*min\s*/\s*m[²2]|升/分/平方米)','L/min/m2'),
    'gedi_ml_m2':(r'GEDI[\s:：=]*'+NUMBER+r'\s*(ml\s*/\s*m[²2]|毫升/平方米)','mL/m2'),
    'elwi_ml_kg':(r'ELWI[\s:：=]*'+NUMBER+r'\s*(ml\s*/\s*kg|毫升/公斤)','mL/kg'),
    'norepinephrine_pump_ml_h':(r'去甲肾上腺素[^。；;，,\n\d]{0,24}'+NUMBER+r'\s*(ml\s*/\s*h(?:r)?|毫升/小时)','mL/h'),
    'dopamine_pump_ml_h':(r'多巴胺[^。；;，,\n\d]{0,24}'+NUMBER+r'\s*(ml\s*/\s*h(?:r)?|毫升/小时)','mL/h'),
}
CONTEXT={
    'operative_mentions':r'手术|术中|术后|术前',
    'bleeding_mentions':r'出血|失血',
    'urine_review_mentions':r'尿量.{0,12}(?:报告|医生)|(?:报告|医生).{0,12}尿量',
    'norepinephrine_mentions':r'去甲肾上腺素',
    'renal_replacement_mentions':r'透析|血滤|血液滤过',
    'diuretic_mentions':r'呋塞米|速尿|托拉塞米|利尿剂',
}
FEATURE_NAMES=tuple(RULES)+tuple(CONTEXT)


def normalize(text):return unicodedata.normalize('NFKC',str(text)).strip()


def observations(text):
    text=normalize(text);out=[]
    for field,(pattern,unit) in RULES.items():
        for match in re.finditer(pattern,text,re.I):
            # Hypothetical thresholds are not measured values. Actual reported
            # values can accompany a subsequent order in the same note.
            prefix=text[max(0,match.start()-12):match.start()]
            if re.search(r'若|如果|目标|维持在|小于|大于|低于|高于|<|>|≤|≥',prefix):continue
            value=float(match.group(1));rawunit=match.group(2) if match.lastindex and match.lastindex>1 else ''
            if field=='hemoglobin_g_l' and re.match(r'g\s*/\s*dl',rawunit,re.I):value*=10
            out.append({'field':field,'value':value,'unit':unit,'raw_match':match.group(),
                        'span_start':match.start(),'span_end':match.end(),
                        'measurement_clock':'unknown; record clock is availability proxy only'})
    return out


def context_counts(text):
    text=normalize(text)
    return {name:len(re.findall(pattern,text,re.I)) for name,pattern in CONTEXT.items()}


def rate_scope(text,match_start,match_end,component):
    """Require an explicit category prefix; named bottles do not set all-category rates."""
    text=normalize(text);fragment=text[match_start:match_end]
    if re.search(r'口服|鼻饲|胃管|肠内|鼻肠',fragment):return 'enteral_or_oral_excluded'
    expected={'crystalloid':r'晶体(?:液|组)?','colloid':r'胶体(?:液|组)?','water':r'糖水(?:组)?|水组'}
    if component not in expected:return 'unresolved_component'
    if not re.search(expected[component],fragment):return 'named_fluid_or_unspecified_scope'
    if component=='water' and re.search(r'维生素|维他|胰岛素|氯化钾',fragment):return 'specific_mixed_group'
    return 'explicit_component'


def pause_scope(text,component):
    text=normalize(text)
    if re.search(r'口服|鼻饲|胃管|肠内|鼻肠',text):return 'route_requires_review'
    if re.search(r'(?:晶体|胶体|糖水|水组).{0,8}(?:三升袋|治疗组|治疗液体|此组|该组)',text):return 'specific_named_group'
    if component=='water' and re.search(r'维生素|维他|胰岛素|氯化钾',text):return 'specific_mixed_group'
    return 'explicit_component' if component in ('crystalloid','colloid','water') else 'unresolved_component'


def additional_controls(text):
    """Find common short-form pauses missed by the historical parser."""
    text=normalize(text);out=[]
    rx=r'(?:暂停|停止|停输|停用|停)(?:输入|输注)?(?P<subject>晶体(?:液|组)?|胶体(?:液|组)?|糖水(?:组)?|水组)'
    for match in re.finditer(rx,text):
        prefix=text[max(0,match.start()-15):match.start()]
        eligible=not bool(re.search(r'未|不|无需|若|如果|计划|拟|准备|建议|考虑',prefix))
        component='crystalloid' if '晶体' in match.group() else 'colloid' if '胶体' in match.group() else 'water'
        out.append({'event_kind':'pause','component':component,'match_start':match.start(),'match_end':match.end(),
                    'matched_text':match.group(),'eligible':eligible,'scope':pause_scope(text,component),
                    'duration_known':False,'resumption_time_known':False})
    return out


def unqualified_observations(text):
    """Retain potentially useful numbers without guessing missing/typo units."""
    text=normalize(text);accepted=observations(text);out=[]
    pattern=r'(乳酸(?:值)?|(?<![A-Za-z])lac(?:tate)?|(?<![A-Za-z])BE|HCO3-?|血红蛋白|肌酐|PCCI|GEDI|ELWI|SVRI)[\s:：=]*'+NUMBER
    for match in re.finditer(pattern,text,re.I):
        if any(o['span_start']<=match.start()<o['span_end'] for o in accepted):continue
        out.append({'field_raw':match.group(1),'number_raw':match.group(2),
                    'context':text[max(0,match.start()-15):min(len(text),match.end()+35)],
                    'reason':'not a qualified observation: verify unit, conditional context and clock; no inferred scale'})
    return out
