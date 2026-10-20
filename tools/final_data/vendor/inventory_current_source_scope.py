"""Conservative source-visible primary-fluid scope; no final adjudication read."""
import re
import math
from tools.final_data.vendor.inventory_text_quantity import normalize,VOLUME,DENY

DEDICATED=re.compile('去甲肾上腺素|多巴胺|右美托咪定|硝普钠|胰岛素|丙泊酚|瑞芬太尼')
PUMP=re.compile('泵入|泵注|微泵|持续泵')
MIX=re.compile(r'\+|加入|配制|配伍|混合|稀释|溶于|组')
RENAL=re.compile('血液透析|血透|血液净化|CRRT|透析',re.I)
SALINE=re.compile(r'氯化钠(?:注射液)?|生理盐水|(?<![A-Za-z])NS(?![A-Za-z])',re.I)
GLUCOSE=re.compile(r'葡萄糖(?!酸钙)|糖水')


def current_source_scope(name,input_kind,route,volume_ml,same_clock_notes,*,same_quantity_peers=1):
    """Only exact record/form notes may be supplied by caller.

    No blanket exclusion of drink-kind rows, pumps, or three-litre bags. A drug
    carrier needs explicit mixture+pump+same fluid quantity in one clause.
    """
    name,kind,route=map(normalize,[name,input_kind,route])
    if re.search('静脉注射|静推|静注|肌注',route):
        return dict(scope='excluded_injection',reason='original_injection_route',evidence=route)
    if re.search('口服|鼻饲|胃管|肠管',route):
        return dict(scope='oral_or_enteral',reason='original_enteral_route',evidence=route)
    notes=[normalize(note) for note in same_clock_notes]
    if RENAL.search(kind):
        evidence=next((note for note in notes if RENAL.search(note)),None)
        if evidence is not None:
            return dict(scope='auxiliary_renal',reason='original_renal_purpose_tag_and_same_record_nursing',evidence=evidence)
        return dict(scope='uncertain',reason='renal_tag_without_same_record_corroboration',evidence=kind)
    try:quantity=float(volume_ml)
    except (TypeError,ValueError):quantity=math.nan
    fluid=SALINE if SALINE.search(name) else GLUCOSE if GLUCOSE.search(name) else None
    if fluid is not None and math.isfinite(quantity) and quantity>0:
        for note in notes:
            for clause in re.split('[，,。；;\n]',note):
                if DENY.search(clause) or not DEDICATED.search(clause) or not MIX.search(clause):continue
                if not PUMP.search(clause) and '泵' not in route:continue
                for match in fluid.finditer(clause):
                    # Restrict the dose to the immediately named carrier.
                    suffix=clause[match.end():]
                    offset=re.match(r'\s*(?:液)?\s*(?:为|:)?\s*',suffix).end()
                    value=VOLUME.match(suffix,offset)
                    if value is not None and abs(float(value['number'])-quantity)<1e-8:
                        if same_quantity_peers!=1:
                            return dict(scope='uncertain',reason='ambiguous_same_clock_carrier_quantity_peer',evidence=clause)
                        return dict(scope='auxiliary_drug_carrier',reason='explicit_same_record_dedicated_mixture_and_quantity',evidence=clause)
    return dict(scope='primary_candidate',reason='no_source_visible_auxiliary_scope_proof',evidence='')
