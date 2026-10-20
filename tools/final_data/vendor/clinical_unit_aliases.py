"""Narrow, auditable spelling/separator repairs; unknown units remain unknown."""
import re
import unicodedata


def unit_alias_observations(text):
    text=unicodedata.normalize('NFKC',str(text));out=[]
    rules=[('cardiac_index_l_min_m2',r'PCCI[\s:：=]*([+-]?\d+(?:\.\d+)?)\s*(L\s*/\s*mln\s*/\s*m2)','L/min/m2','literal PCCI unit typo mln -> min'),
           ('gedi_ml_m2',r'GEDI[\s:：=]*([+-]?\d+(?:\.\d+)?)\s*:\s*(ml\s*/\s*m2)','mL/m2','redundant colon before explicit GEDI unit')]
    for field,pattern,unit,reason in rules:
        for match in re.finditer(pattern,text,re.I):
            before=text[max(0,match.start()-12):match.start()]
            if re.search('若|如果|目标|<|>|≤|≥',before):continue
            out.append({'field':field,'value':float(match.group(1)),'raw_match':match.group(),'raw_unit':match.group(2),'unit':unit,'unit_correction':reason})
    return out
