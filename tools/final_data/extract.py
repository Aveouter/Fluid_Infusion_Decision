"""Raw-record extraction with row provenance; no training-time preprocessing."""
import hashlib
import re
import unicodedata
import numpy as np
import pandas as pd
from .vendor.fluid_reconstruction_sources import extract_intake,extract_notes,local_time
from .vendor.inventory_numeric_vitals import normalized_numeric_vitals,canonical_name
from .vendor.inventory_cvp_string import cvp_string_records
from .vendor.fluid_text_evidence import observations,unqualified_observations,context_counts
from .vendor.clinical_unit_aliases import unit_alias_observations
from .vendor.inventory_reported_chemistry import reported_chemistry,feature_name
from .vendor.inventory_explicit_context_rates import extract_omitted_ml_rates
from .vendor.inventory_current_source_scope_v2 import current_source_scope,SALINE,GLUCOSE,RENAL
from .vendor.inventory_text_quantity import family,resolve_record,normalize
from .vendor.inventory_event_sources import source_load_candidates

VITAL_NAMES=('temperature','heart_rate','pulse','respiratory_rate','systolic_bp','diastolic_bp','cvp','spo2','glucose_recorded_value')
VITAL_UNITS=('degC','beats/min','beats/min','breaths/min','recorded_BP_scale','recorded_BP_scale','recorded_CVP_scale','percent','recorded_glucose_scale')


def stable_id(*parts):return hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:24]


def clinical_records(raw,pid,injury,source_sha):
    """Yield every accepted observation plus explicitly rejected text candidates."""
    clock=pd.to_datetime(raw.FORM_TIME.str.replace('Z','',regex=False),format='mixed',errors='coerce')
    hours=(clock-pd.Timestamp(injury)).dt.total_seconds()/3600
    _,accepted=normalized_numeric_vitals(raw,injury)
    for row in accepted.itertuples():
        i=int(row.source_record_number)-2;c=int(row.component)
        yield dict(patient_id=pid,observation_id=stable_id(pid,i,VITAL_NAMES[c],'typed'),
            source_csv_record=i+2,source_form_id=raw.FORM_ID.iloc[i],source_sha256=source_sha,
            record_time_local=str(clock.iloc[i]),record_hour=float(hours.iloc[i]),hour=int(row.hour),
            available_at_hour=int(row.hour)+1,field=VITAL_NAMES[c],value=float(row.value),
            unit=VITAL_UNITS[c],qualifier='exact_recorded_scalar',eligible=True,
            method='typed_NUMBER_VALUE_legacy_range',reason='accepted_on_recorded_scale',
            raw_match=raw.NUMBER_VALUE.iloc[i],source_field='NUMBER_VALUE',
            measurement_time_verified=False)
    cvp=cvp_string_records(raw,injury)
    for row in cvp.itertuples():
        i=int(row.source_record_number)-2
        yield dict(patient_id=pid,observation_id=stable_id(pid,i,'cvp','string'),source_csv_record=i+2,
            source_form_id=raw.FORM_ID.iloc[i],source_sha256=source_sha,record_time_local=str(clock.iloc[i]),
            record_hour=row.record_hour,hour=row.hour,available_at_hour=row.hour+1,field='cvp',value=row.value,
            unit='recorded_CVP_scale',qualifier='exact_recorded_scalar',eligible=True,
            method='typed_CVP_blank_NUMBER_bare_STRING',reason='no_unit_conversion',
            raw_match=raw.STRING_VALUE.iloc[i],source_field='STRING_VALUE',measurement_time_verified=False)
    # Keep rejected typed numeric candidates, rather than silently losing the
    # distinction between an absent measurement and a recorded out-of-domain one.
    named={k:raw[k].map(canonical_name) for k in ['组合项目名称','节点名称','父节点名称']}
    typed=[named['节点名称'].eq(x) for x in ['体温：','心率：','脉搏：','呼吸：','血压：']]
    typed += [named['父节点名称'].eq('血压：'),named['组合项目名称'].eq('CVP')|named['节点名称'].eq('CVP'),named['节点名称'].eq('氧饱和度：'),named['父节点名称'].eq('血糖：')]
    accepted_keys={(int(r.source_record_number)-2,int(r.component)) for r in accepted.itertuples()}
    numbers=pd.to_numeric(raw.NUMBER_VALUE.map(lambda v:unicodedata.normalize('NFKC',str(v)).strip()),errors='coerce')
    for c,mask in enumerate(typed):
        for i in np.flatnonzero(mask&hours.ge(0)&hours.lt(72)&numbers.notna()):
            if (int(i),c) in accepted_keys:continue
            yield dict(patient_id=pid,observation_id=stable_id(pid,i,VITAL_NAMES[c],'typed_rejected'),source_csv_record=int(i)+2,
                source_form_id=raw.FORM_ID.iloc[i],source_sha256=source_sha,record_time_local=str(clock.iloc[i]),
                record_hour=float(hours.iloc[i]),hour=int(hours.iloc[i]),available_at_hour=int(hours.iloc[i])+1,
                field=VITAL_NAMES[c],value=float(numbers.iloc[i]),unit=VITAL_UNITS[c],qualifier='recorded_scalar_outside_guard',
                eligible=False,method='typed_NUMBER_VALUE_legacy_range',reason='outside_declared_legacy_range',
                raw_match=raw.NUMBER_VALUE.iloc[i],source_field='NUMBER_VALUE',measurement_time_verified=False)
    # A discharge amount is a recorded collection amount, never an hourly rate.
    out=raw[raw['组合项目名称'].str.startswith('出量:',na=False)]
    for _,g in out.groupby(['FORM_ID','FORM_TIME','INNER_SEQ'],sort=False):
        names=g.loc[g['组合项目名称'].eq('出量:名称'),'STRING_VALUE']
        if not names.str.contains('尿',na=False).any():continue
        amount=g[g['组合项目名称'].eq('出量:量')]
        values=pd.to_numeric(amount.NUMBER_VALUE,errors='coerce').dropna().unique()
        if len(values)!=1:continue
        i=int(amount.index[0]);at=hours.iloc[i]
        if not 0<=at<72:continue
        units=g.loc[g['组合项目名称'].str.contains('单位'),'STRING_VALUE'].map(normalize).str.strip().unique()
        unit=units[0] if len(units)==1 and units[0] else 'unspecified'
        residual=names.str.contains('残余尿|残尿',regex=True,na=False).any()
        prefix='residual_urine' if residual else 'urine'
        field=prefix+'_recorded_ml' if unit.lower() in ['ml','毫升','cc'] else prefix+'_recorded_amount_unspecified_unit'
        good=0<=values[0]<=5000
        yield dict(patient_id=pid,observation_id=stable_id(pid,i,field),source_csv_record=i+2,
            source_form_id=raw.FORM_ID.iloc[i],source_sha256=source_sha,record_time_local=str(clock.iloc[i]),
            record_hour=float(at),hour=int(at),available_at_hour=int(at)+1,field=field,value=float(values[0]),
            unit=unit,qualifier='recorded_collection_amount',eligible=good,
            method='same_FORM_ID_FORM_TIME_INNER_SEQ_name_amount',reason=('residual_urine_not_new_production' if residual else 'collection_duration_unknown') if good else 'outside_legacy_0_5000_guard',
            raw_match=str(values[0]),source_field='NUMBER_VALUE',measurement_time_verified=False)
    # Chronological duplicate suppression is a declared source-record rule,
    # not a claim repeated paragraphs are independent new measurements.
    take=np.flatnonzero(hours.ge(0)&hours.lt(72)&~raw['组合项目名称'].str.startswith(('入量:','出量:'))&raw.STRING_VALUE.ne(''))
    take=take[np.argsort(hours.iloc[take].to_numpy(),kind='stable')];seen=set()
    for i in take:
        text=unicodedata.normalize('NFKC',raw.STRING_VALUE.iloc[i]).strip()
        copied=len(text)>=80 and text in seen
        if len(text)>=80:seen.add(text)
        parsed=[]
        for o in observations(text):parsed.append(dict(o,field='text_'+o['field'],qualifier='exact_reported_value',eligible=not copied,reason='repeated_long_note' if copied else 'explicit_text_value',method='explicit_unit_text'))
        for o in unit_alias_observations(text):parsed.append(dict(o,field='text_'+o['field'],qualifier='exact_reported_value',eligible=not copied,reason='repeated_long_note' if copied else o['unit_correction'],method='literal_unit_alias'))
        for o in reported_chemistry(text):
            feature=feature_name(o)
            if feature is None:continue
            why='repeated_long_note' if copied else o['reason']
            if o['field']=='potassium' and o['value']>=100:why='unresolved_three_digit_K_source'
            parsed.append(dict(o,field=feature,eligible=o['eligible'] and not copied and why!='unresolved_three_digit_K_source',reason=why,method='explicit_reported_chemistry'))
        for o in unqualified_observations(text):
            parsed.append(dict(field='unqualified_'+o['field_raw'],value=float(o['number_raw']),unit='unknown',qualifier='unqualified',eligible=False,reason=o['reason'],method='unqualified_text_candidate',raw_match=o['context']))
        for j,o in enumerate(parsed):
            yield dict(patient_id=pid,observation_id=stable_id(pid,i,o['field'],o.get('span_start',j),o['method']),
                source_csv_record=int(i)+2,source_form_id=raw.FORM_ID.iloc[i],source_sha256=source_sha,
                record_time_local=str(clock.iloc[i]),record_hour=float(hours.iloc[i]),hour=int(hours.iloc[i]),
                available_at_hour=int(hours.iloc[i])+1,field=o['field'],value=o['value'],unit=o['unit'],
                qualifier=o['qualifier'],eligible=o['eligible'],method=o['method'],reason=o['reason'],
                raw_match=o.get('raw_match',''),source_field='STRING_VALUE',measurement_time_verified=False)


def settings_and_loads(raw,pid,injury,events):
    notes,unresolved=extract_notes(raw,pid,pd.Timestamp(injury))
    clocks=pd.to_datetime(raw.FORM_TIME.str.replace('Z','',regex=False),format='mixed',errors='coerce')
    nursing=raw[~raw['组合项目名称'].str.startswith(('入量:','出量:'))&raw.STRING_VALUE.ne('')]
    bykey={}
    for i,row in nursing.iterrows():
        at=clocks.iloc[i]
        if pd.notna(at):bykey.setdefault((str(row.FORM_ID),at.value),[]).append(row.STRING_VALUE)
        if pd.isna(at) or not re.search(r'(?:补液|输液|输注)速度.*?/\s*(?:h|小时)',row.STRING_VALUE,re.I):continue
        for e in extract_omitted_ml_rates(row.STRING_VALUE,at):
            hour=(pd.Timestamp(e['event_time_local'])-pd.Timestamp(injury)).total_seconds()/3600
            if 0<=hour<72:notes.append(dict(patient_id=pid,source_file=f'raw/patient_records/{pid}.csv',
              source_record_number=int(i)+1,form_id=row.FORM_ID,record_time_local=str(at),available_hour=(at-pd.Timestamp(injury)).total_seconds()/3600,
              hour_from_injury=hour,rate_event_id=stable_id(pid,int(i)+1,'context-rate',e['match_start']),text_raw=row.STRING_VALUE,source='narrative',**e))
    # Source-only classification explicitly uses ORIGINAL component/quantity,
    # never final completion-derived volume, net return or final decision.
    own=events.copy();own['original_component']=own.component_original.fillna(own.component)
    own['family']=[family(n) for n in own.drug_name_raw.fillna('')]
    own['clock']=pd.to_datetime(own.record_time_local,errors='coerce')
    own['peers']=own.groupby(['form_id','clock','family'],dropna=False).event_id.transform('size')
    scope=[];receipts=[]
    for e in own.itertuples():
        if e.original_component not in ['crystalloid','colloid','water']:continue
        if re.search('静脉注射|静推|静注|肌注|鼻饲|口服|胃管|肠管',str(e.route_raw)):continue
        values=bykey.get((str(e.form_id),e.clock.value),[]) if pd.notna(e.clock) else []
        role=current_source_scope(e.drug_name_raw,e.input_kind_raw,e.route_raw,e.volume_ml,values,same_quantity_peers=int(e.peers))
        scope.append(dict(patient_id=pid,event_id=e.event_id,source_component=e.original_component,
            source_volume_ml=e.volume_ml,record_time_local=e.record_time_local,form_id=str(e.form_id),**role))
        if pd.isna(e.volume_ml) and pd.notna(e.clock):
            parsed=resolve_record(e.drug_name_raw,e.form_id,e.clock.value,bykey,int(e.peers))
            if parsed['method']=='same_clock_form_actual_mention' and not parsed['conflict']:
                receipts.append(dict(patient_id=pid,hour=float(e.hour_from_injury),available_hour=(e.clock-pd.Timestamp(injury)).total_seconds()/3600,
                    kind='addition',value=float(parsed['volume_ml']),component=e.original_component,source_id=e.event_id,quantity_source=parsed['method']))
    return notes,unresolved,scope,receipts
