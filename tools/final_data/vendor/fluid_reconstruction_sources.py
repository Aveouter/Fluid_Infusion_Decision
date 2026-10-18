"""Source ledgers for the workflow approved on 2026-10-14.

Bottle amounts are additions to stock. Doctor values are component rates.
No component rate is derived from mixture proportions or bottle sizes.
"""
from collections import defaultdict,Counter
from pathlib import Path
import difflib,hashlib,io,json,re,unicodedata,zipfile
import numpy as np
import pandas as pd

CATS={'晶体':'crystalloid','胶体':'colloid','水':'water','口服':'oral'}
ORAL={'口服','鼻饲','胃管注入','胃管泵入','鼻肠管泵入','鼻肠管注入','鼻胃管注入'}
IV={'','静滴','静滴续','静脉续','输入','三升袋静滴','输液泵静滴','静脉泵入','泵入','泵静滴'}
FIELDS=['入量:名称','入量:量','入量:途径','入量:实用量单位','入量:备用量单位','入量:备用量','入量:入量分类']
COLS=['FORM_TIME','FORM_ID','INNER_SEQ','组合项目名称','NUMBER_VALUE','STRING_VALUE','REFERENCE_ID']
ANCHORS=['葡萄糖酸钙','氯化钠','氯化钾','碳酸氢钠','白蛋白','林格','红细胞','血浆','右旋糖酐','甲硝唑']

def is_injection_route(value):
    return bool(re.search(r'静脉注射|静脉推注|静推|静注',norm(value)))

def norm(value):
    return '' if pd.isna(value) else unicodedata.normalize('NFKC',str(value)).strip()

def drug_key(value):
    text=re.sub(r'[★☆!\s]','',norm(value))
    text=re.sub(r'^\(集采\)','',text)
    text=re.sub(r'\([^()%]*\d+(?:\.\d+)?\s*ml[^()%]*\)','',text,flags=re.I)
    return text

def dose_free_key(value):
    # Used only for classification. Original dose and source name remain intact.
    return re.sub(r'(?:\d+(?:\.\d+)?(?:万)?(?:ml|mg|ug|g|iu|u|单位|支|瓶|袋))+$','',drug_key(value),flags=re.I)

def fluid_subject(value):
    """Specific source identity for event binding, never a rate inference."""
    text=norm(value)
    if '右旋糖酐' in text or '代血浆' in text:return 'other_colloid'
    if '白蛋白' in text:return 'albumin'
    if '红细胞' in text:return 'red_cells'
    if '血浆' in text:return 'plasma'
    return ''

def local_time(value):
    try:
        t=pd.Timestamp(value)
        return t.tz_localize(None) if t.tzinfo is not None else t
    except (ValueError,TypeError):return pd.NaT

def read_raw(path):
    blob=Path(path).read_bytes()
    for enc in ['utf-8-sig','gb18030']:
        try:return pd.read_csv(io.BytesIO(blob),dtype=str,usecols=COLS,encoding=enc,keep_default_na=False),enc,hashlib.sha256(blob).hexdigest()
        except UnicodeDecodeError:pass
    raise ValueError('Unrecognized CSV encoding')

class DrugReference:
    def __init__(self,path):
        data=pd.read_excel(path,dtype=str).fillna('')
        self.exact=defaultdict(set);self.keyed=defaultdict(set);self.byname=defaultdict(set);self.examples={}
        for _,row in data.iterrows():
            name,route,category=[norm(row[c]) for c in ['入量:名称','入量:途径','分类']]
            self.exact[(name,route)].add(category)
            key=drug_key(name)
            self.keyed[(key,route)].add(category);self.byname[key].add(category)
            self.examples.setdefault(key,name)
        self.keys=list(self.byname)
        self.dose_free=defaultdict(set)
        for key,categories in self.byname.items():self.dose_free[dose_free_key(key)].update(categories)
        rules_path=Path(__file__).with_name('fluid_classification_rules_20261014.json')
        self.reviewed=json.loads(rules_path.read_text(encoding='utf-8'))

    def classify(self,name,route):
        name=norm(name);route=norm(route);key=drug_key(name)
        if is_injection_route(route):return '排除注射用药',name,'user_confirmed_injection_exclusion',''
        if route in ORAL or re.search(r'口服|鼻饲|胃管|肠管',route):return '口服',name,'explicit_oral_route',''
        for mapping,k,method in [(self.exact,(name,route),'reference_exact'),(self.keyed,(key,route),'reference_normalized')]:
            values=mapping.get(k,set())
            if len(values)==1:return next(iter(values)),name,method,''
            if len(values)>1:
                if not values.intersection(CATS):return '其他',name,'reference_non_target_consensus',''
                return '',name,method,'reference_category_conflict'
        alias=self.reviewed.get('route_aliases',{}).get(route)
        if alias:
            values=self.exact.get((name,alias),set()) or self.keyed.get((key,alias),set())
            if len(values)==1:return next(iter(values)),name,'reference_route_alias',''
        if route in IV:
            reviewed=self.reviewed.get('reviewed_names',{}).get(key)
            if reviewed:return reviewed['category'],name,reviewed['rule_id'],''
            for rule in self.reviewed.get('reference_family_rules',[]):
                if re.search(rule['pattern'],name):
                    evidence=set().union(*(v for (n,r),v in self.exact.items() if re.search(rule['pattern'],n) and r in IV))
                    if evidence==set(rule['require_reference_categories']):return rule['category'],name,rule['rule_id'],''
            for rule in self.reviewed.get('non_target_reference_families',[]):
                if re.search(rule['pattern'],name):
                    evidence=set().union(*(v for (n,r),v in self.exact.items() if re.search(rule['pattern'],n) and r in IV))
                    if evidence and not evidence.intersection(CATS):return '其他',name,rule['rule_id'],''
        values=self.byname.get(key,set())
        if len(values)==1:return next(iter(values)),name,'unique_name_reference',''
        if values and not values.intersection(CATS):return '其他',name,'reference_non_target_consensus',''
        values=self.dose_free.get(dose_free_key(key),set())
        if len(values)==1:return next(iter(values)),name,'unique_reference_after_dose_suffix_removal',''
        if values and not values.intersection(CATS):return '其他',name,'reference_non_target_consensus',''
        # Do not change concentrations, doses or chemically meaningful names.
        numbers=re.findall(r'\d+(?:\.\d+)?%?',key)
        anchors={a for a in ANCHORS if a in key}
        candidates=[]
        if len(key)>=8:
            for other in difflib.get_close_matches(key,self.keys,n=6,cutoff=.94):
                if re.findall(r'\d+(?:\.\d+)?%?',other)!=numbers:continue
                if {a for a in ANCHORS if a in other}!=anchors:continue
                cats=self.byname[other]
                if len(cats)==1:candidates.append((difflib.SequenceMatcher(None,key,other).ratio(),other,next(iter(cats))))
        candidates.sort(reverse=True)
        if candidates and (len(candidates)==1 or candidates[0][0]-candidates[1][0]>=.03):
            _,other,cat=candidates[0]
            return cat,self.examples[other],'unique_conservative_typo_correction',''
        return '',name,'unresolved','unknown_drug_classification'

def quantity(value,unit,name,backup_unit=''):
    text=norm(value);unit=norm(unit).lower();method='explicit_unit'
    if not text:return None,'','missing_amount'
    pattern=r'([+-]?\d+(?:\.\d+)?)\s*(ml|毫升|cc|l|升|袋|瓶)?'
    match=re.fullmatch(pattern,text,re.I)
    if not match:return None,'','ambiguous_amount_text'
    number=float(match.group(1));suffix=(match.group(2) or '').lower()
    factors={'ml':1,'毫升':1,'cc':1,'l':1000,'升':1000}
    if unit and suffix and unit!=suffix and (unit not in factors or suffix not in factors or factors[unit]!=factors[suffix]):return None,'','unit_conflict'
    unit=unit or suffix
    if not unit and norm(backup_unit).lower() in {'ml','毫升','cc','l','升'}:
        unit=norm(backup_unit).lower();method='unit_from_same_entry_backup_unit'
    if unit in factors:value=number*factors[unit]
    elif unit in {'袋','瓶'}:
        sizes=re.findall(r'(\d+(?:\.\d+)?)\s*ml',norm(name),re.I)
        if len(set(sizes))!=1:return None,'','missing_unique_package_volume'
        value=number*float(sizes[0]);method='count_times_explicit_package_volume'
    else:return None,'','missing_volume_unit' if not unit else 'unsupported_volume_unit'
    return value,method,'negative_amount_pending_adjudication' if value<0 else ''

def extract_intake(raw,pid,injury,reference):
    selected=raw[raw['组合项目名称'].isin(FIELDS)]
    groups={}
    for index,row in selected.iterrows():
        key=(row.FORM_TIME,row.FORM_ID,row.INNER_SEQ)
        group=groups.setdefault(key,{'fields':defaultdict(lambda: {'num':set(),'text':set()}),'records':[]})
        group['records'].append(index+1)
        field=group['fields'][row['组合项目名称']]
        if norm(row.NUMBER_VALUE):field['num'].add(norm(row.NUMBER_VALUE))
        if norm(row.STRING_VALUE):field['text'].add(norm(row.STRING_VALUE))
    events=[]
    for key,group in groups.items():
        issues=[]
        def one(field):
            values=group['fields'].get(field,{'num':set(),'text':set()})
            seq=values['num'] or values['text']
            if len(seq)>1:
                # Numerically equivalent renderings are not conflicting quantities.
                if field in {'入量:量','入量:备用量'}:
                    nums={float(v) for v in seq if re.fullmatch(r'[+-]?\d+(?:\.\d+)?',v)}
                    if len(nums)==1 and all(re.fullmatch(r'[+-]?\d+(?:\.\d+)?',v) for v in seq):return str(next(iter(nums)))
                issues.append('conflicting_'+field);return ''
            return next(iter(seq),'')
        name,amount,route,unit,backupunit,backupamount,kind=[one(f) for f in FIELDS]
        category,corrected,classification_method,problem=reference.classify(name,route)
        if problem:issues.append(problem)
        # Package size belongs to the raw source name, never a canonical name
        # borrowed from a differently sized reference product.
        volume,volume_method,problem=quantity(amount,unit,name,backupunit)
        if problem:issues.append(problem)
        if not name:issues.append('missing_drug_name')
        if category in CATS and category!='口服' and route not in IV:issues.append('noncontinuous_iv_route')
        record_time=local_time(key[0])
        if pd.isna(record_time):issues.append('invalid_record_time')
        if injury is None:issues.append('missing_injury_time')
        hour=(record_time-injury).total_seconds()/3600 if injury is not None and pd.notna(record_time) else None
        event_id=hashlib.sha256((pid+'|'+ '|'.join(key)).encode()).hexdigest()[:24]
        events.append({'event_id':event_id,'patient_id':pid,'source_form_time_raw':key[0],
            'record_time_local':str(record_time),'event_time_local':str(record_time),'time_source':'form_record_local_clock',
            'time_precision':'record_seconds','source_z_interpreted_as_local_clock':True,
            'hour_from_injury':hour,'form_id':key[1],'inner_seq':key[2],'drug_name_raw':name,
            'drug_name_corrected':corrected,'category_reference':category,'component':CATS.get(category,'other'),
            'classification_method':classification_method,'route_raw':route,'route_inferred':not bool(route) and category in CATS,
            'amount_raw':amount,'unit_raw':unit,'backup_unit_raw':backupunit,'backup_amount_raw':backupamount,
            'volume_ml':volume,'volume_method':volume_method,'input_kind_raw':kind,
            'eligible_stock_addition':category in CATS and volume is not None and volume>=0 and not issues,
            'issues':';'.join(sorted(set(issues))), 'source_file':f'data_speed/original_data/{pid}.csv',
            'source_record_numbers':';'.join(map(str,group['records'])),
            'fields_raw_json':json.dumps({f:{k:sorted(v) for k,v in vals.items()} for f,vals in group['fields'].items()},ensure_ascii=False)})
    return events

TYPE=re.compile(r'晶体(?:液|组|通道)?|胶体(?:液|组|通道)?|糖水(?:组)?|葡萄糖(?:组|水)|水组|平衡盐(?:液|组)?|乳酸钠林格(?:注射液)?|林格(?:液)?|白蛋白|血浆|红细胞')
RATE=re.compile(r'(?<![\d.])(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>ml\s*(?:/\s*(?:hr?|小时)|每小时)|毫升\s*(?:/|每)\s*小时|l\s*/\s*(?:hr?|小时)|升\s*(?:/|每)\s*小时|滴\s*/\s*分)',re.I)
CLOCK=re.compile(r'(?<![A-Za-z0-9.:/%])(?P<h>\d{1,2})(?::(?P<m>\d{2})(?![0-9.%A-Za-z])|时(?:(?P<cm>\d{1,2})分?)?)')

def component_for(word):
    if re.search('晶体|平衡盐|林格',word):return 'crystalloid'
    if re.search('胶体|白蛋白|血浆|红细胞',word):return 'colloid'
    return 'water'

def event_clock(text,anchor,record_time):
    # Bind a nearby explicit clock to this event, not to unrelated admission text.
    prefix=text[max(0,anchor-35):anchor]
    matches=[m for m in CLOCK.finditer(prefix)
             if not re.search(r'比例|配比|稀释|按\s*$',prefix[max(0,m.start()-8):m.start()])
             and not re.match(r'稀释|比例|配比',prefix[m.end():])]
    if matches:
        m=matches[-1];h=int(m.group('h'));minute=int(m.group('m') or m.group('cm') or 0)
        if h<24 and minute<60:
            t=record_time.normalize()+pd.Timedelta(hours=h,minutes=minute)
            if t-record_time>pd.Timedelta(hours=1):t-=pd.Timedelta(days=1)
            if pd.Timedelta(0)<=record_time-t<=pd.Timedelta(hours=12):return t,'explicit_clock_in_event_context','minute'
    return record_time,'form_record_local_clock','record_seconds'

def rate_statement_reasons(text,start,end,statement_start):
    tail=re.search(r'[。；;\n]',text[end:])
    statement_end=end+tail.start() if tail else len(text)
    statement=text[statement_start:statement_end]
    reasons=[]
    if re.search(r'或(?:者)?\s*(?:为)?\s*(?:晶体|胶体|糖水|水组|葡萄糖)',statement):
        reasons.append('alternative_component_rate_statement')
    before=text[max(statement_start,start-35):start]
    if re.search(r'(?:计划|拟予|拟调|如果|若|建议|待)[^。；;]{0,25}(?:补液|速度|晶体|胶体|糖水|水组)',before):
        reasons.append('conditional_or_planned')
    return reasons

def parse_narrative(text,record_time):
    """Preserve clear numbers, omitted units, controls, and flush evidence."""
    text=norm(text);results=[];covered=[]
    breaks=list(re.finditer(r'[。；;\n]',text))
    for m in RATE.finditer(text):
        begin=max([x.end() for x in breaks if x.end()<=m.start()] or [0])
        prefix=text[max(begin,m.start()-70):m.start()]
        # Recover a subject across a comma only when no intervening rate appears.
        previous=list(RATE.finditer(prefix))
        if previous:prefix=prefix[previous[-1].end():]
        current_clause=re.split(r'[，,]',prefix)[-1]
        if TYPE.search(current_clause) or re.search('治疗组|治疗液体|三升袋|葡萄糖酸钙|曲马多|去甲|镇痛|镇静|洛赛克|抗生素|药物|营养|另一通道|另一组',current_clause):
            prefix=current_clause
        elif re.search('输完|完毕|暂停|停止',prefix):
            prefix=current_clause
        subjects=list(TYPE.finditer(prefix));components={component_for(x.group()) for x in subjects}
        component=next(iter(components)) if len(components)==1 else 'joint' if len(components)>1 else 'total' if '总' in prefix and re.search('补液|速度',prefix) else 'unspecified'
        reasons=rate_statement_reasons(text,m.start(),m.end(),begin)
        if not subjects and not re.search('补液|输液泵|输注速度|输液速度',prefix):continue
        if re.search('曲马多|葡萄糖酸钙|治疗组|治疗液体|三升袋|去甲|多巴胺|镇痛|镇静|丙泊酚|洛赛克|抗生素|药物|营养',current_clause) and not TYPE.search(current_clause):
            component='other_channel'
        if re.search(r'\d\s*[-~～至/+]\s*$',prefix):reasons.append('rate_range_or_composite')
        if re.search(r'(减少|增加|加快|减慢|上调|下调)[^至为]*$',prefix):reasons.append('relative_rate_change')
        if re.search('计划|拟予|拟调|如果|若|建议',prefix):reasons.append('conditional_or_planned')
        unit=m.group('unit');value=float(m.group('num'))
        if re.match(r'^(?:l\s*/|升)',unit,re.I):value*=1000
        if '滴' in unit:reasons.append('drop_factor_required');value=None
        time,source,precision=event_clock(text,m.start(),record_time)
        results.append({'event_kind':'rate_change','component':component,'rate_ml_h':value,
            'rate_raw':m.group(),'unit_inferred':False,'event_time_local':str(time),'time_source':source,
            'time_precision':precision,'match_start':max(begin,m.start()-len(prefix)),'match_end':m.end(),
            'matched_text':prefix+m.group(),'issues':';'.join(sorted(set(reasons))),'eligible':not reasons and component!='other_channel'})
        covered.append((m.start(),m.end()))
    # Infer omitted mL/h only in an explicitly fluid-rate context.
    if re.search('速度|滴速|输液泵',text):
        pattern=re.compile(r'(?P<subject>晶体(?:液|组)?|胶体(?:液|组)?|水组|糖水组?)\s*(?:补液)?\s*(?:速度)?\s*(?:为|调至|调为|:)?\s*(?P<num>\d+(?:\.\d+)?)')
        for m in pattern.finditer(text):
            if any(a<=m.start('num')<b for a,b in covered):continue
            following=text[m.end():m.end()+8]
            if re.match(r'\s*(?:ml|毫升|%|g|袋|瓶)',following,re.I):continue
            begin=max([x.end() for x in breaks if x.end()<=m.start()] or [0])
            reasons=rate_statement_reasons(text,m.start('num'),m.end(),begin)
            if re.match(r'\s*(?:[-~～至+]\s*\d|/\s*\d)',following):reasons.append('rate_range_or_composite')
            prefix=text[max(begin,m.start()-25):m.start()]
            if re.search(r'(减少|增加|加快|减慢|上调|下调)[^至为]*$',prefix):reasons.append('relative_rate_change')
            time,source,precision=event_clock(text,m.start(),record_time)
            results.append({'event_kind':'rate_change','component':component_for(m.group('subject')),
                'rate_ml_h':float(m.group('num')),'rate_raw':m.group(),'unit_inferred':True,
                'event_time_local':str(time),'time_source':source,'time_precision':precision,
                'match_start':m.start(),'match_end':m.end(),'matched_text':m.group(),'issues':';'.join(sorted(set(reasons))),'eligible':not reasons})
    # Keep controls separate from numeric rates. Do not turn "unchanged" into zero.
    for kind,rx in [('pause',r'(?:暂停|停止|停用|停输)[^，,。；;\n]{0,8}?(晶体|胶体|糖水|水组)|(晶体|胶体|糖水|水组)[^，,。；;\n]{0,8}?(?:暂停|停止|停用|停输)'),
        ('complete',r'(晶体|胶体|血浆|红细胞|白蛋白|糖水|水组)[^，,。；;\n]{0,8}?(?:输完|输入完毕|输注完毕|输毕)'),
        ('resume',r'(?:恢复|重新开始)[^，,。；;\n]{0,8}?(晶体|胶体|糖水|水组)')]:
        for m in re.finditer(rx,text):
            subject=next(v for v in m.groups() if v)
            reasons=[]
            clause=re.split(r'[，,。；;\n]',text[:m.start()])[-1]
            if (re.search(r'未|无需|不必|待|计划|拟|准备|建议|若|如果',clause[-20:]) or re.search('未|无需|不必',m.group()) or
                    (text[m.end():].startswith('后') and not re.search('已|现已',clause))):
                reasons.append('conditional_or_negated_control')
            time,source,precision=event_clock(text,m.start(),record_time)
            results.append({'event_kind':kind,'component':component_for(subject),'rate_ml_h':None,'rate_raw':'',
                'completion_subject':fluid_subject(subject) if kind=='complete' else '',
                'unit_inferred':False,'event_time_local':str(time),'time_source':source,'time_precision':precision,
                'match_start':m.start(),'match_end':m.end(),'matched_text':m.group(),'issues':';'.join(reasons),'eligible':not reasons})
    for m in re.finditer(r'冲管|冲洗(?:管道|残留|导管)?',text):
        clause=text[max(0,m.start()-35):min(len(text),m.end()+35)]
        if not re.search('胶体|血浆|白蛋白|红细胞|葡萄糖|糖水',clause):continue
        vols=re.findall(r'(\d+(?:\.\d+)?)\s*(?:ml|毫升)(?!\s*/)',clause,re.I)
        volume=float(vols[0]) if len(vols)==1 else None
        time,source,precision=event_clock(text,m.start(),record_time)
        results.append({'event_kind':'flush','component':'water','rate_ml_h':None,'rate_raw':'',
            'unit_inferred':False,'event_time_local':str(time),'time_source':source,'time_precision':precision,
            'match_start':m.start(),'match_end':m.end(),'matched_text':clause,'flush_volume_ml':volume,
            'issues':'' if volume is not None else 'flush_volume_not_recorded','eligible':True})
    return results

def extract_notes(raw,pid,injury):
    mask=~raw['组合项目名称'].str.startswith(('入量:','出量:'),na=False)&raw.STRING_VALUE.str.contains('补液|输液|晶体|胶体|糖水|葡萄糖|冲管|冲洗|血浆|白蛋白|红细胞|林格|平衡盐',regex=True,na=False)
    notes=raw[mask].drop_duplicates(['FORM_TIME','FORM_ID','INNER_SEQ','组合项目名称','STRING_VALUE'])
    results=[];review=[]
    for index,row in notes.iterrows():
        time=local_time(row.FORM_TIME)
        if pd.isna(time):continue
        text=norm(row.STRING_VALUE);parsed=parse_narrative(text,time)
        base={'patient_id':pid,'record_time_local':str(time),'source_form_time_raw':row.FORM_TIME,
            'source_record_number':index+1,'form_id':row.FORM_ID,'field_name':row['组合项目名称'],
            'source_file':f'data_speed/original_data/{pid}.csv','text_raw':text}
        if not parsed and re.search('速度|冲管|冲洗|暂停|停止|输完|输毕',text):review.append(base|{'issue':'unresolved_narrative_event'})
        for item in parsed:
            hour=(local_time(item['event_time_local'])-injury).total_seconds()/3600 if injury is not None else None
            item=base|item|{'hour_from_injury':hour,'source':'narrative'}
            item['rate_event_id']=hashlib.sha256(f'{pid}|{index+1}|{item["match_start"]}|{item["event_kind"]}'.encode()).hexdigest()[:24]
            results.append(item)
    return results,review

def doctor_rates(zip_path,injuries):
    results=[];manifest=[]
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith('.csv'):continue
            blob=archive.read(member)
            try:df=pd.read_csv(io.BytesIO(blob),encoding='gb18030',dtype=str,keep_default_na=False)
            except UnicodeDecodeError:df=pd.read_csv(io.BytesIO(blob),encoding='utf-8-sig',dtype=str,keep_default_na=False)
            if not {'时间','晶体速度','胶体速度','水速度'}<=set(df):continue
            pid=Path(member).stem;injury=injuries.get(pid)
            manifest.append({'patient_id':pid,'member':member,'sha256':hashlib.sha256(blob).hexdigest(),'rows':len(df)})
            if injury is None:continue
            for i,row in df.iterrows():
                h=pd.to_numeric(row['时间'],errors='coerce')
                if pd.isna(h) or h!=int(h) or not 1<=h<=72:continue
                for cn,en in list(CATS.items())[:3]:
                    text=norm(row[cn+'速度'])
                    if not text:continue
                    value=pd.to_numeric(text,errors='coerce');valid=pd.notna(value) and np.isfinite(value) and value>=0
                    time=injury+pd.Timedelta(hours=int(h)-1)
                    results.append({'rate_event_id':hashlib.sha256(f'{member}|{i}|{cn}'.encode()).hexdigest()[:24],
                        'patient_id':pid,'component':en,'event_kind':'rate_change','rate_ml_h':float(value) if valid else None,
                        'rate_raw':text,'event_time_local':str(time),'hour_from_injury':int(h)-1,'source':'doctor_table',
                        'time_source':'doctor_hour_start_approximation','time_precision':'hour_estimate',
                        'source_member':member,'source_record_number':i+1,'channel_raw':norm(row['通道']),
                        'eligible':bool(valid),'issues':'' if valid else 'invalid_doctor_rate','text_raw':''})
    return results,manifest
