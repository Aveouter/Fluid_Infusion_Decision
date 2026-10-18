"""Narrow, auditable completed-disposition extraction, not planned discharge.

When a note says only 'discharged today', it supplies a day interval, not an
invented exact departure time. Uncertain boundary hours are excluded from full
hour supervision. Transfers within hospital are not hospital discharge.
"""
import re
import unicodedata

import pandas as pd


_DIGITS = dict(zip('零〇一二三四五六七八九', (0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)))
_NUM = r'[0-9零〇一二三四五六七八九十]+'


def chinese_number(text):
    if text.isdigit():
        return int(text)
    if '十' in text:
        left, right = text.split('十')
        return (chinese_number(left) if left else 1) * 10 + (chinese_number(right) if right else 0)
    return int(''.join(str(_DIGITS[c]) for c in text))


def dispositions(text, record_time):
    text = unicodedata.normalize('NFKC', str(text))
    record = pd.Timestamp(record_time).tz_localize(None)
    found = []
    death = re.compile(r'死亡于(' + _NUM + r')年(' + _NUM + r')月(' + _NUM + r')日(' + _NUM + r')时(' + _NUM + r')分')
    for m in death.finditer(text):
        try:
            point = pd.Timestamp(*[chinese_number(t) for t in m.groups()])
        except (ValueError, TypeError, KeyError):
            continue
        if abs((point - record).total_seconds()) <= 86400:
            found.append(dict(kind='death', lower=point, upper=point,
                              precision='explicit_death_clock', matched_clause=m.group()))
    clock = re.compile(r'于(?:今日)?(?P<h>\d{1,2}):(?P<m>\d{2})(?:自动)?(?:出院|离院)')
    for m in clock.finditer(text):
        if re.search(r'明日|拟|计划|定于', text[max(0, m.start() - 6):m.start()]):
            continue
        h, minute = int(m['h']), int(m['m'])
        if h >= 24 or minute >= 60:
            continue
        point = record.normalize() + pd.Timedelta(hours=h, minutes=minute)
        if point > record + pd.Timedelta(hours=6):
            point -= pd.Timedelta(days=1)
        found.append(dict(kind='discharge', lower=point, upper=point,
                          precision='explicit_discharge_clock', matched_clause=m.group()))
    if not any(v['kind'] == 'discharge' for v in found):
        match = re.search(r'于今日(?:自动)?出院|患者今日出院', text)
        if match and not re.search(r'拟|计划|明日|定于', text[max(0, match.start() - 6):match.start()]):
            found.append(dict(kind='discharge', lower=record.normalize(),
                              upper=record.normalize() + pd.Timedelta(days=1),
                              precision='same_day_interval', matched_clause=match.group()))
    return found
