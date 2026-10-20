"""Separate directly evidenced auxiliary sources without future-role leakage."""
import math
import re
import pandas as pd
from tools.final_data.vendor.inventory_event_sources import source_load_candidates

PROOFS={
    'auxiliary_renal':'original_renal_purpose_tag_and_same_record_nursing',
    'auxiliary_drug_carrier':'explicit_same_record_dedicated_mixture_and_quantity',
}


def primary_source_load_candidates(events,injury_time,role_records):
    """Return primary known/uncertain plus a retained auxiliary-source ledger.

    Role records must originate from exactsameform/currentrecord evidence.
    Later clinical adjudications are not accepted as an online role filter.
    """
    known,uncertain=source_load_candidates(events,injury_time)
    source={str(row.event_id):row for row in events.itertuples()}
    exclude={};seen=set()
    for role in role_records:
        event_id=str(role['event_id'])
        if event_id in seen:raise ValueError('Duplicate source-role evidence')
        seen.add(event_id)
        if role['scope'] not in PROOFS:continue
        if role['reason']!=PROOFS[role['scope']]:raise ValueError('Unsupported retrospective/unknown scope proof')
        if event_id not in source:raise ValueError('Source-role event not present')
        event=source[event_id]
        if pd.Timestamp(role['record_time_local'])!=pd.Timestamp(event.record_time_local):
            raise ValueError('Role evidence must have same original availability clock')
        if str(role['form_id'])!=str(event.form_id):raise ValueError('Role evidence must have same form')
        volume=float(event.volume_ml);evidenced=float(role['source_volume_ml'])
        if not all(math.isfinite(v) for v in [volume,evidenced]) or abs(volume-evidenced)>1e-8:
            raise ValueError('Auxiliary role source quantity mismatch')
        if not role.get('evidence'):raise ValueError('Explicit original scope evidence required')
        # Explicit flushes are retained as primary candidates; a compound
        # containing both mixture and flush language is not uniquely a carrier.
        if role['scope']=='auxiliary_drug_carrier' and re.search('冲管|冲洗|润管',str(role['evidence'])):continue
        exclude[event_id]=role
    auxiliary=[]
    for event in known+uncertain:
        if event['source_id'] in exclude:
            role=exclude[event['source_id']]
            auxiliary.append(dict(event,scope=role['scope'],scope_reason=role['reason'],scope_evidence=role['evidence']))
    return ([event for event in known if event['source_id'] not in exclude],
        [event for event in uncertain if event['source_id'] not in exclude],auxiliary)
