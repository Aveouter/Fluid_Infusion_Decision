"""Reference event-clock queries; no completion-derived actual flow is a setting.

Events must already have adjudicated timestamps and tie precedence. Queries are
reference-label operations; future events must never be passed as model inputs.
"""
import math


def hour_setting_segments(changes, hour):
    """Clip ordered setting events to [hour,hour+1), preserving unknown start.

    An event at hour is effective immediately; at hour+1 belongs to the next
    hour. Same-time events use supplied order (last wins). No backwards fill.
    """
    if not math.isfinite(hour):
        raise ValueError('hour must be finite')
    previous=-math.inf
    selected=None
    for event in changes:
        at=float(event['hour']);rate=float(event['rate'])
        if not math.isfinite(at) or not math.isfinite(rate) or rate<0 or at<previous:
            raise ValueError('events must be finite, nonnegative and time ordered')
        previous=at
    points=[float(hour)]+sorted({float(e['hour']) for e in changes if hour<float(e['hour'])<hour+1})+[float(hour+1)]
    output=[];position=0
    for start,end in zip(points,points[1:]):
        while position<len(changes) and float(changes[position]['hour'])<=start:
            selected=changes[position];position+=1
        output.append(dict(start_hour=start,end_hour=end,
            setting_ml_h=None if selected is None else float(selected['rate']),
            source_id='' if selected is None else str(selected.get('source_id','')),
            source='' if selected is None else str(selected.get('source','')),
            time_precision='' if selected is None else str(selected.get('time_precision',''))))
    return output


def summarize_setting_hour(segments):
    if not segments:raise ValueError('nonempty segments required')
    known=all(s['setting_ml_h'] is not None for s in segments)
    return dict(start_setting_ml_h=segments[0]['setting_ml_h'],
        end_setting_ml_h=segments[-1]['setting_ml_h'],
        setting_integral_ml=None if not known else sum((s['end_hour']-s['start_hour'])*s['setting_ml_h'] for s in segments),
        full_setting_known=known,
        interpretation='Setting integral is a rate-duration quantity, not delivered volume; finite stock and reconstruction assumptions remain separate.')
