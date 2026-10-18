"""Reference-only rate-duration feasibility; never an online input builder."""
import math


def setting_interval(changes,start,end):
    """Integrate all documented changes on [start,end), keeping unknown duration.

    This is an attempted-flow integral, not delivered volume. A delivery above
    it is incompatible with treating these documented settings as actual flow;
    it does not establish which clinical source is inaccurate.
    """
    if not all(math.isfinite(float(t)) for t in [start,end]) or end<start:
        raise ValueError('Finite ordered interval required')
    previous=-math.inf
    for event in changes:
        at=float(event['hour']);rate=float(event['rate'])
        if not math.isfinite(at) or at<previous or not math.isfinite(rate) or rate<0:
            raise ValueError('Ordered finite nonnegative settings required')
        previous=at
    points=[float(start)]+sorted({float(e['hour']) for e in changes if start<float(e['hour'])<end})+[float(end)]
    cursor=0;selected=None;known=unknown=integral=0.;peak=None;pieces=[]
    for left,right in zip(points,points[1:]):
        if right==left:continue
        while cursor<len(changes) and float(changes[cursor]['hour'])<=left:
            selected=changes[cursor];cursor+=1
        rate=None if selected is None else float(selected['rate'])
        if rate is None:unknown+=right-left
        else:
            known+=right-left;integral+=rate*(right-left);peak=rate if peak is None else max(peak,rate)
        pieces.append(dict(start_hour=left,end_hour=right,setting_ml_h=rate,
            source_id='' if selected is None else str(selected.get('source_id',''))))
    return dict(integral_ml=integral if unknown==0 else None,known_duration_h=known,unknown_duration_h=unknown,
        peak_setting_ml_h=peak,segments=pieces)


def clock_envelope_upper(events,start,end):
    """Loose upper integral when event i may occur in [earliest,latest].

    Pointwise maxima can correspond to incompatible choices at different
    times, so this is an outer envelope, not an attainable optimal curve.
    Exact nursing clocks are fixed by supplying equal endpoints. An unobserved
    initial setting leaves the upper integral unknown, never zero.
    """
    if not all(math.isfinite(float(t)) for t in [start,end]) or end<start:raise ValueError('Invalid interval')
    for event in events:
        lo,hi,rate=[float(event[key]) for key in ['earliest','latest','rate']]
        if not all(math.isfinite(v) for v in [lo,hi,rate]) or hi<lo or rate<0:raise ValueError('Invalid event window')
    points=[float(start),float(end)]+[float(e[key]) for e in events for key in ['earliest','latest'] if start<float(e[key])<end]
    points=sorted(set(points));total=unknown=0.
    for left,right in zip(points,points[1:]):
        time=(left+right)/2
        mandatory=[e for e in events if e['latest']<=time]
        if not mandatory:unknown+=right-left;continue
        newest_mandatory_start=max(e['earliest'] for e in mandatory)
        possible=[e['rate'] for e in events if e['earliest']<=time and e['latest']>=newest_mandatory_start]
        total+=max(possible)*(right-left)
    return dict(upper_integral_ml=total if unknown==0 else None,unknown_duration_h=unknown)
