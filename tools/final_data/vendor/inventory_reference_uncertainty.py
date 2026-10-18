"""Reference-only uncertainty; these flags must never become model inputs."""
import numpy as np
from tools.final_data.vendor.inventory_reference_feasibility import setting_interval,clock_envelope_upper


def assess_anchor(changes,doctor_metadata,start,end,tolerance_ml=.01,allocated_ml=None):
    nominal=setting_interval([dict(e,rate=e['value']) for e in changes],start,end)
    windows=[];unresolved=0
    for event in changes:
        at=float(event['hour']);latest=at
        if event['source']=='doctor_table':
            metadata=doctor_metadata.get(str(event['source_id']))
            if metadata is None:unresolved+=1;continue
            if metadata['time_precision']=='hour_estimate' and abs(float(metadata['hour'])-at)<1e-8:latest=at+1.
            elif metadata['time_precision'] not in ['hour_estimate','record_seconds']:unresolved+=1;continue
        elif not str(event['source']).startswith('explicit_nursing_'):unresolved+=1;continue
        windows.append(dict(earliest=at,latest=latest,rate=float(event['value'])))
    envelope=clock_envelope_upper(windows,start,end) if unresolved==0 else dict(upper_integral_ml=None,unknown_duration_h=None)
    upper=envelope['upper_integral_ml']
    robust=allocated_ml is not None and upper is not None and float(allocated_ml)>upper+tolerance_ml
    return dict(nominal=nominal,envelope=envelope,unresolved_precision_events=unresolved,robust_incompatible=bool(robust))


def uncertainty_masks(shape,intervals):
    """Whole overlapping volume windows and strictly interior state origins.

    The interval is uncertain; this does not claim each overlapping hour is
    individually wrong. At endpoints, cumulative constraints do not depend on
    the disputed within-interval allocation. Original values are not modified.
    """
    if len(shape)!=3 or shape[1:]!=(72,3):raise ValueError('Expected patient x72x3 reference grid')
    volume=np.zeros(shape,bool);state=np.zeros(shape,bool)
    hours=np.arange(72,dtype=float)
    for interval in intervals:
        p,c=int(interval['patient']),int(interval['component'])
        start,end=float(interval['start']),float(interval['end'])
        if not 0<=p<shape[0] or not 0<=c<3 or not 0<=start<end<=72:raise ValueError('Invalid reference interval')
        volume[p,:,c]|=(hours<end)&(hours+1>start)
        state[p,:,c]|=(hours>start+1e-9)&(hours<end-1e-9)
    return volume,state
