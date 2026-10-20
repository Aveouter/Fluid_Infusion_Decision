"""Source-ledger adapters shared by input and reference builders.

No effective volume, completion linkage, lot net quantity or model target is
read here. The caller must supply already selected patient/source rows.
"""
import pandas as pd


def setting_source_events(doctors,notes,injury_time):
    injury=pd.Timestamp(injury_time)
    dd=[dict(hour=float(d.hour_from_injury),available_hour=float(d.hour_from_injury),value=float(d.rate_ml_h),
        component=d.component,source_id=d.rate_event_id) for d in doctors[doctors.eligible].itertuples()]
    nn=[dict(hour=float(n.hour_from_injury),available_hour=(pd.Timestamp(n.record_time_local)-injury).total_seconds()/3600,
        kind='setting' if n.event_kind=='rate_change' else str(n.event_kind),value=None if pd.isna(n.rate_ml_h) else float(n.rate_ml_h),
        component=n.component,source_id=n.rate_event_id,unit_inferred=bool(n.unit_inferred),
        explicit_change=any(word in str(n.matched_text) for word in ['调','改','开始']))
        for n in notes[notes.eligible].itertuples() if pd.notna(n.hour_from_injury)]
    return dd,nn


def source_load_candidates(events,injury_time):
    """Recorded source amounts plus unknown-quantity cues; no guessed amounts.

    A name-only cue is not an assertion that a new physical bottle was loaded.
    Unknown/negative sources remain distinguishable; they are not zero loads.
    """
    injury=pd.Timestamp(injury_time)
    component=events.component_original.fillna(events.component)
    route=events.route_raw.fillna('')
    chosen=events[component.isin(['crystalloid','colloid','water'])&~route.str.contains('静脉注射|静推|静注|肌注|鼻饲|口服|胃管|肠管')].copy()
    chosen['_source_component']=component.loc[chosen.index]
    known=[];uncertain=[]
    for _,e in chosen.iterrows():
        if pd.isna(e.hour_from_injury):continue
        item=dict(hour=float(e.hour_from_injury),available_hour=(pd.Timestamp(e.record_time_local)-injury).total_seconds()/3600,
            component=e['_source_component'],source_id=e.event_id)
        if pd.notna(e.volume_ml) and e.volume_ml>0 and e.volume_method=='explicit_unit':
            known.append(dict(**item,kind='addition',value=float(e.volume_ml)))
        elif pd.isna(e.volume_ml) or (e.volume_ml>0 and e.volume_method!='explicit_unit'):
            uncertain.append(dict(**item,kind='unknown_quantity_cue'))
    return known,uncertain
