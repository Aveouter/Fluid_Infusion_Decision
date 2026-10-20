"""Reference-only quality flags. These are forbidden prediction inputs."""
import math
import numpy as np
from .vendor.inventory_event_clock import hour_setting_segments,summarize_setting_hour
from .vendor.inventory_reference_uncertainty import assess_anchor
from .vendor.inventory_event_state import past_event_state

TOLERANCE_ML=0.01


def conditional_delivery(stock,loads,pieces):
    """Known-setting FIFO under a conditional source ledger, not clinical truth."""
    if not all(s['setting_ml_h'] is not None for s in pieces):return None
    cuts=sorted({pieces[0]['start_hour'],pieces[-1]['end_hour'],*[s['start_hour'] for s in pieces],*[a for a,q in loads]})
    remaining=float(stock);total=0.
    for a,b in zip(cuts,cuts[1:]):
        remaining+=sum(q for at,q in loads if at==a)
        rate=next(s['setting_ml_h'] for s in pieces if s['start_hour']<=a<s['end_hour'])
        used=min(remaining,rate*(b-a));remaining-=used;total+=used
    return total


def reference_rows(pid,components,events,known,segments,constraints,lots,hours,metadata,patient):
    """Keep both reference streams; flag conflicts without changing a source value."""
    output=[];settings=[];evidence=[]
    for component in components:
        changes=[dict(e,rate=e['value']) for e in events if e['component']==component]
        part=segments[segments.component.eq(component)]
        anchors=constraints[constraints.component.eq(component)&constraints.resolution.eq('volume_endpoint_applied')]
        ownlots=lots[lots.component.eq(component)]
        unsafe=np.zeros(72,bool);why=[set() for _ in range(72)]

        def flag(a,b,reason,source_id):
            if not np.isfinite(a+b) or b<=a:return
            for h in range(72):
                if h<b and h+1>a:unsafe[h]=True;why[h].add(reason)
            evidence.append(dict(patient_id=pid,component=component,start_hour=float(a),end_hour=float(b),reason=reason,source_id=source_id))

        # Operative and prehospital interval estimates do not describe observed
        # within-hour flow. Preserve their totals without exact joint support.
        for s in part.itertuples():
            if 'operative' in str(s.method) or 'prehospital' in str(s.method):
                flag(s.start_hour,s.end_hour,'interval_allocation_without_exact_flow',str(s.anchor_source_id))
        for lot in ownlots.itertuples():
            if lot.remaining_status=='censored_at_operation_transition' and lot.remaining_at_episode_end_ml>1e-6:
                t=float(lot.episode_end)
                if 0<t<72 and abs(t-round(t))>1e-9:flag(math.floor(t),math.ceil(t),'operation_transition_coverage_uncertain',str(lot.lot_id))
        for a in anchors.itertuples():
            start,end=float(a.interval_start_hour),float(a.hour)
            if not 0<=start<end<=72:continue
            check=assess_anchor(changes,metadata,start,end,allocated_ml=float(a.allocated_ml))
            if check['robust_incompatible']:flag(start,end,'endpoint_exceeds_documented_clock_envelope',str(a.source_id))
            evidence.append(dict(patient_id=pid,component=component,start_hour=start,end_hour=end,
                reason='endpoint_evidence',source_id=str(a.source_id),allocated_ml=float(a.allocated_ml),
                nominal_setting_integral_ml=check['nominal']['integral_ml'],
                clock_envelope_upper_ml=check['envelope']['upper_integral_ml'],
                unresolved_clock_metadata=check['unresolved_precision_events']))
        summaries=[];conditional=[]
        for h in range(72):
            pieces=hour_setting_segments(changes,h);summaries.append(summarize_setting_hour(pieces))
            settings += [dict(patient_id=pid,component=component,**p) for p in pieces]
            ledger=[e for e in known+events if e['component']==component]
            state=past_event_state(ledger,h)
            arrivals=[(e['hour'],e['value']) for e in known if e['component']==component and h<=e['hour']<h+1]
            lo=conditional_delivery(state['remaining_lower_ml'],arrivals,pieces)
            hi=conditional_delivery(state['remaining_upper_ml'],arrivals,pieces)
            conditional.append((lo,hi))
            value=hours.loc[h,component+'_input_ml']
            if np.isfinite(value):
                check=assess_anchor(changes,metadata,h,h+1,allocated_ml=float(value))
                if check['robust_incompatible']:
                    contributing=part[part.start_hour.lt(h+1)&part.end_hour.gt(h)]
                    for s in contributing.itertuples():
                        matched=anchors[anchors.source_id.eq(str(s.anchor_source_id))]
                        if len(matched)==1:
                            a=matched.iloc[0];flag(float(a.interval_start_hour),float(a.hour),'hourly_allocation_exceeds_clock_envelope',str(a.source_id))
                    flag(h,h+1,'hourly_quantity_setting_conflict','')
                # No false assertion that the source clock/stock estimate is
                # bedside truth: a discrepancy is dual-evidence uncertainty.
                # It can reflect omitted pauses/returns/loads or interpolation.
                if lo is not None and value<lo-TOLERANCE_ML:
                    flag(h,h+1,'below_conditional_documented_flow','')
                    contributing=part[part.start_hour.lt(h+1)&part.end_hour.gt(h)]
                    for s in contributing.itertuples():
                        matched=anchors[anchors.source_id.eq(str(s.anchor_source_id))]
                        if len(matched)==1:
                            a=matched.iloc[0];flag(float(a.interval_start_hour),float(a.hour),'conditional_flow_interval_conflict',str(a.source_id))
        for h,summary in enumerate(summaries):
            source=hours.loc[h];value=source[component+'_input_ml']
            full=h>=patient['admission_hour'] and h+1<=patient['observation_end_lower_hour']
            raw=float(value) if np.isfinite(value) else np.nan
            unknown=str(source.get(component+'_unquantified_source_event_ids',''))
            unknown='' if unknown in ['nan','None'] else unknown
            # No generic fill of an unknown numeric record. Only absence of all
            # recorded additions up to this hour permits a recorded-no-input0.
            recorded_before=[e for e in known if e['component']==component and 0<=e['hour']<h+1]
            no_input_zero=bool(full and not np.isfinite(raw) and not recorded_before and not unknown)
            analysis=raw if full else np.nan
            if no_input_zero:analysis=0.
            prior=bool(patient['cross_patient_volume_prior_dependency'])
            endpoint=bool(full and np.isfinite(analysis) and not prior)
            exact=bool(endpoint and summary['full_setting_known'] and not unsafe[h] and not unknown)
            method=str(source.get(component+'_method',''))
            if no_input_zero:method='no_recorded_input_in_complete_inpatient_hour'
            output.append(dict(patient_id=pid,hour_start=h,component=component,
                hour_start_local=str(source.hour_start_local),full_inpatient_hour=bool(full),
                reference_reconstructed_input_ml=raw,input_ml=analysis,
                input_value_status='outside_or_partial_boundary' if not full else 'recorded_no_input_zero' if no_input_zero else 'estimated_reference' if np.isfinite(analysis) else 'unknown',
                reference_method=method,reference_confidence=source.get(component+'_confidence',''),
                reconstructed_is_imputed=source.get(component+'_is_imputed',True),
                start_setting_ml_h=summary['start_setting_ml_h'],end_setting_ml_h=summary['end_setting_ml_h'],
                setting_time_integral_ml=summary['setting_integral_ml'],full_setting_known=summary['full_setting_known'],
                conditional_delivered_lower_ml=conditional[h][0],conditional_delivered_upper_ml=conditional[h][1],
                setting_reference_eligible=bool(full and summary['end_setting_ml_h'] is not None),
                quantity_reference_eligible=endpoint,exact_joint_eligible=exact,
                average_actual_input_rate_ml_h=analysis if full else np.nan,
                uncertainty_reasons=';'.join(sorted(why[h])),unquantified_source_event_ids=unknown,
                cross_patient_prior_dependency=prior,
                missing_flush_effect_not_imputed=source.get('flush_missing_effect_not_imputed',True)))
    return output,settings,evidence
