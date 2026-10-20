"""Resolve doctor event timing using only nursing evidence available at origin.

Full retrospective doctor/narrative links belong to target construction, never
this input path. Supplied availability clocks are business-record proxies until
validated against real entry logs.
"""
import math


def resolve_past_doctor_settings(doctors,nursing,origin_hour):
    if not math.isfinite(origin_hour):raise ValueError('finite origin required')
    def visible(event):
        at=float(event['hour']);known=float(event['available_hour'])
        if not math.isfinite(at) or not math.isfinite(known):raise ValueError('finite sourced clocks required')
        return 0<=at<origin_hour and known<origin_hour
    # No future kind, component, amount or text payload is used for matching.
    notes=[n for n in nursing if visible(n)]
    output=[]
    for d in doctors:
        if not visible(d):continue
        at=float(d['hour']);rate=float(d['value'])
        if not math.isfinite(rate) or rate<0:raise ValueError('finite nonnegative doctor setting required')
        matches=[n for n in notes if n['kind']=='setting' and n['component']==d['component']
                 and n['value'] is not None and abs(float(n['value'])-rate)<1e-6
                 and int(n['hour'])==int(at) and n['explicit_change']]
        times={float(n['hour']) for n in matches}
        match=matches[0] if len(times)==1 else None
        if match is not None:at=float(match['hour'])
        output.append(dict(hour=at,available_hour=max(float(d['available_hour']),at,
            float(match['available_hour']) if match is not None else float(d['available_hour'])),
            kind='setting',value=rate,component=d['component'],source_id=d['source_id'],
            timing_narrative_source_id='' if match is None else match['source_id']))
    return sorted(output,key=lambda x:x['hour'])


def resolve_past_setting_events(doctors,nursing,origin_hour):
    """Add explicit component pauses/resumes and pre-doctor initial settings.

    Numeric nursing conflicts after a doctor setting do not overwrite it.
    Resume requires a known prior positive setting; no population fallback.
    """
    doctor_events=resolve_past_doctor_settings(doctors,nursing,origin_hour)
    timeline=[(e['hour'],10,i,e,True) for i,e in enumerate(doctor_events)]
    for i,n in enumerate(nursing):
        at=float(n['hour']);known=float(n['available_hour'])
        if not math.isfinite(at) or not math.isfinite(known):raise ValueError('finite sourced clocks required')
        if not 0<=at<origin_hour or known>=origin_hour:continue
        if n['component'] not in ('crystalloid','colloid','water') or n['kind'] not in ('setting','pause','resume'):continue
        timeline.append((at,40 if n['kind']=='setting' else 30,i,n,False))
    timeline.sort(key=lambda item:item[:3])
    doctor_known=set();positive={};output=[]
    for at,_,_,event,is_doctor in timeline:
        c=event['component'];kind=event['kind']
        inferred=False if is_doctor else bool(event.get('unit_inferred',False))
        if is_doctor:
            doctor_known.add(c);rate=float(event['value'])
        elif kind=='setting':
            if c in doctor_known or event['value'] is None:continue
            rate=float(event['value'])
            if not math.isfinite(rate) or rate<0:raise ValueError('finite nonnegative nursing setting required')
        elif kind=='pause':rate=0.
        elif c in positive:rate,inferred=positive[c]
        else:continue
        if rate>0:positive[c]=(rate,inferred)
        output.append(dict(hour=at,available_hour=max(at,float(event['available_hour'])),kind='setting',value=rate,
            component=c,source_id=event['source_id'],source='doctor_table' if is_doctor else 'explicit_nursing_'+kind,
            unit_inferred=inferred))
    return output
