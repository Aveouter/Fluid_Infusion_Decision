"""Conservative omitted-mL rate extraction, separate from frozen parser.

Requires a named single fluid component, explicit administered volume in mL,
and a fluid-setting /h statement in the same sentence. No drip conversion or
allocation of unspecified/combined rates. Output retains unit-inference flags.
"""
import re
from tools.final_data.vendor.fluid_reconstruction_sources import TYPE,component_for,norm,event_clock


def extract_omitted_ml_rates(text,record_time):
    text=norm(text);output=[]
    for sentence in re.finditer(r'[^。；;\n]+',text):
        clause=sentence.group()
        if re.search(r'计划|拟|建议|如果|若|待|曾|昨日|外院|入院前|转入前|家属|静脉注射|静推|静注',clause):continue
        matches=list(TYPE.finditer(clause))
        components={component_for(m.group()) for m in matches}
        if len(components)!=1:continue
        for m in re.finditer(r'(?:补液|输液|输注)速度\s*(?:为|调至|调为|[:：])?\s*(?P<rate>\d+(?:\.\d+)?)\s*/\s*(?:h(?:r)?|小时)(?![A-Za-z])',clause,re.I):
            prefix=clause[:m.start()]
            if not re.search(r'\d+(?:\.\d+)?\s*(?:ml|毫升)(?!\s*/)',prefix,re.I):continue
            if not re.search(r'静滴|输注|输入|泵入',prefix):continue
            if not any(subject.end()<=m.start() for subject in matches):continue
            if re.search(r'滴\s*/|加快|减慢|增加|减少|上调|下调|[-~～至+]\s*$',prefix):continue
            start=sentence.start()+m.start('rate');end=sentence.start()+m.end()
            at,source,precision=event_clock(text,start,record_time)
            output.append(dict(event_kind='rate_change',component=next(iter(components)),rate_ml_h=float(m['rate']),
                event_time_local=str(at),time_source=source,time_precision=precision,
                match_start=sentence.start()+m.start(),match_end=end,matched_text=clause[:m.end()],
                unit_inferred=True,quantity_confidence='inferred_ml_unit_from_explicit_fluid_volume_and_hour_denominator',
                eligible=True,issues='omitted_ml_unit_inferred',source_rule='named_fluid_explicit_ml_volume_and_fluid_rate_per_hour'))
    return output
