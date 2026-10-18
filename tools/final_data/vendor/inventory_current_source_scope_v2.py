"""Purpose evidence differs from new-administration evidence.

Stopping an explicitly described drug mixture can identify its purpose without
asserting a new infusion at that time. Planning/nonadministration is abstained.
"""
import math,re
from tools.final_data.vendor.inventory_current_source_scope import current_source_scope as initial_scope
from tools.final_data.vendor.inventory_current_source_scope import SALINE,GLUCOSE,DEDICATED,PUMP,MIX,RENAL
from tools.final_data.vendor.inventory_text_quantity import normalize,VOLUME

ROLE_DENY=re.compile('拟|计划|准备|备血|申请|待|建议|如果|若|目标|尚余|剩余|未予|未输|未开始|暂不|取消|不予|不再|无需|禁止|勿|拒绝|累计|总计|昨日|昨天|昨晚|前日|此前|既往|上次')
FLUSH=re.compile('冲管|冲洗|润管')


def current_source_scope(name,input_kind,route,volume_ml,same_clock_notes,*,same_quantity_peers=1):
    notes=[normalize(v) for v in same_clock_notes]
    result=initial_scope(name,input_kind,route,volume_ml,notes,same_quantity_peers=same_quantity_peers)
    if result['scope']=='auxiliary_drug_carrier' and FLUSH.search(result['evidence']):
        return dict(scope='uncertain',reason='mixture_and_flush_scope_not_uniquely_separated',evidence=result['evidence'])
    if result['scope']!='primary_candidate':return result
    name,route=normalize(name),normalize(route)
    fluid=SALINE if SALINE.search(name) else GLUCOSE if GLUCOSE.search(name) else None
    try:quantity=float(volume_ml)
    except (TypeError,ValueError):quantity=math.nan
    if fluid is None or not math.isfinite(quantity) or quantity<=0:return result
    for note in notes:
        for clause in re.split('[，,。；;\n]',note):
            if ROLE_DENY.search(clause) or FLUSH.search(clause) or not DEDICATED.search(clause) or not MIX.search(clause):continue
            if not PUMP.search(clause) and '泵' not in route:continue
            for match in fluid.finditer(clause):
                tail=clause[match.end():];offset=re.match(r'\s*(?:液)?\s*(?:为|:)?\s*',tail).end()
                value=VOLUME.match(tail,offset)
                if value is None or abs(float(value['number'])-quantity)>1e-8:continue
                if same_quantity_peers!=1:
                    return dict(scope='uncertain',reason='ambiguous_same_clock_carrier_quantity_peer',evidence=clause)
                return dict(scope='auxiliary_drug_carrier',reason='explicit_same_record_dedicated_mixture_and_quantity',evidence=clause)
    return result
