"""Past-only ledger state, conditional on supplied recorded additions.

`available_hour` is mandatory. It must be separately sourced; business-clock
availability is not proof of real-time entry. Completion events are unsupported.
"""
import math


def past_event_state(events,origin_hour,*,start_hour=0.,initial_stock_ml=0.):
    if not all(math.isfinite(x) for x in (origin_hour,start_hour,initial_stock_ml)) or origin_hour<start_hour or initial_stock_ml<0:
        raise ValueError('finite ordered window and nonnegative initial stock required')
    past=[]
    for e in events:
        # Inspect clocks before payload: invisible future payloads cannot affect state.
        at=float(e['hour']);available=float(e['available_hour'])
        if not math.isfinite(at) or not math.isfinite(available):raise ValueError('explicit finite event and availability clocks required')
        if at<start_hour or at>=origin_hour or available>=origin_hour:continue
        if e['kind'] not in ('addition','setting'):raise ValueError('only addition/setting ledger supported; do not read retrospective completions')
        value=float(e['value'])
        if not math.isfinite(value) or value<0:raise ValueError('past event quantity must be finite nonnegative')
        past.append((at,e))
    past.sort(key=lambda item:item[0])
    low=high=capacity=float(initial_stock_ml);rate=None;position=float(start_hour)
    rate_source=None;rate_inferred=None;inferred_history=False
    unknown_duration=0.
    def advance(end):
        nonlocal low,high,position,unknown_duration
        dt=end-position
        if rate is None:
            if high>0 and dt>0:low=0.;unknown_duration+=dt
        else:
            low=max(0.,low-rate*dt);high=max(0.,high-rate*dt)
        position=end
    for at,e in past:
        advance(at)
        if e['kind']=='addition':
            low+=e['value'];high+=e['value'];capacity+=e['value']
        else:
            rate=float(e['value']);rate_source=e.get('source_id')
            rate_inferred=bool(e.get('unit_inferred',False));inferred_history|=rate_inferred
    advance(float(origin_hour))
    return dict(remaining_lower_ml=low,remaining_upper_ml=high,
        remaining_point_ml=low if low==high else None,
        remaining_midpoint_estimate_ml=(low+high)/2.,recorded_capacity_ml=capacity,
        last_known_setting_ml_h=rate,unknown_flow_exposure_hours=unknown_duration,
        last_setting_source_id=rate_source,last_setting_unit_inferred=rate_inferred,
        has_unit_inferred_setting_history=inferred_history,
        included_events=len(past),conditional_on_recorded_additions=True)
