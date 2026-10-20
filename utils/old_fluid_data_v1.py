"""Causal, windowed view of the original 363-patient fluid data.

The old six-column label protocol stays authoritative (``docs/label_protocol.md``):
the recorded rate is the target, ``bit = 1 <=> rate > 0``, unrecorded hours are
zero-flow, and rates are never back-filled across an interruption.  This module
only *adds* predictors derived from the same old records; it never imports the
rebuilt ``final_v1`` labels.

Two corrections are applied because they are documented defects of the old
pipeline rather than new data:

* the old ``output_data.pkl`` ``tdata`` was scaled per patient over the whole
  72 h, so a window's scaler saw future hours.  Here every statistic is fitted
  on training windows only.
* a small number of raw values are impossible (temperature 3609 C, respiratory
  rate 7047/min, SpO2 999).  They are set to missing under the guards below and
  counted, instead of being carried into a median/IQR fit.

Everything else is the raw table.  Windows end at the decision origin: the
target row ``i`` is predicted from rows ``i - history .. i - 1`` only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.eval_protocol import CHANGE_ABS_ML, CHANGE_REL, REL_BASE_ML

#: Clinically plausible value domains for the raw hourly chart.  Values outside
#: a domain are treated as missing (never as zero) and counted in ``guards``.
VITAL_COLUMNS = (
    "体温", "心率", "脉搏", "呼吸", "收缩压", "舒张压", "CVP", "氧饱和度", "尿量", "口服.1",
)
VITAL_NAMES = (
    "temperature", "heart_rate", "pulse", "respiratory_rate", "sbp", "dbp", "cvp",
    "spo2", "urine_output", "oral_intake",
)
VITAL_GUARDS = {
    "体温": (25.0, 45.0),
    "心率": (20.0, 250.0),
    "脉搏": (20.0, 250.0),
    "呼吸": (4.0, 80.0),
    "收缩压": (40.0, 300.0),
    "舒张压": (20.0, 200.0),
    "CVP": (0.0, 40.0),
    "氧饱和度": (50.0, 100.0),
    "尿量": (0.0, 2000.0),
    "口服.1": (0.0, 2000.0),
}
RATE_COLUMNS = ("晶体速度", "胶体速度", "水速度")
ADDED_COLUMNS = ("晶体每小时入量", "胶体每小时入量", "水每小时入量")
BASELINE_COLUMNS = ("性别", "年龄", "身高", "体重", "体重指数", "烧伤总面积",
                    "合计Ⅲ°", "合计深Ⅱ°", "合计浅Ⅱ°", "吸入性损伤")
HORIZONS = (6, 12, 24)
EPS = 1e-6


def action_change(previous, next_value):
    """Registered change rule, kept local so the module is self-contained."""
    delta = np.abs(next_value - previous)
    base = np.maximum(np.abs(previous), np.abs(next_value))
    return (delta >= CHANGE_ABS_ML) | ((base >= REL_BASE_ML) & (delta >= CHANGE_REL * base))


def action_classes(previous, current):
    """0 hold/small, 1 start, 2 stop, 3 up, 4 down. Unknown never becomes hold."""
    previous, current = np.asarray(previous, float), np.asarray(current, float)
    valid = np.isfinite(previous) & np.isfinite(current)
    result = np.full(previous.shape, -1, int)
    result[valid] = 0
    changed = valid & action_change(previous, current)
    result[changed & (previous <= 1e-6)] = 1
    result[changed & (current <= 1e-6)] = 2
    result[changed & (previous > 1e-6) & (current > previous)] = 3
    result[changed & (current > 1e-6) & (current < previous)] = 4
    return result


def _guarded(values: np.ndarray, low: float, high: float):
    values = np.asarray(values, dtype=float)
    bad = np.isfinite(values) & ((values < low) | (values > high))
    values = values.copy()
    values[bad] = np.nan
    return values, int(bad.sum())


def load_series(root: Path):
    """Return per-patient old-data series keyed by the bare patient id."""
    root = Path(root)
    frame = pd.read_csv(root / "data/all_combined_data.csv")
    frame = frame[np.isfinite(frame["时间"].to_numpy(float))].copy()
    baseline = pd.read_csv(root / "data/baseline_processed.csv").set_index("Unnamed: 0")
    baseline.index = baseline.index.astype(str)
    train_ids = set(pd.read_csv(root / "data/train_split.csv")["PID"].astype(str))
    test_ids = set(pd.read_csv(root / "data/test_split.csv")["PID"].astype(str))
    if train_ids & test_ids:
        raise ValueError("Old train/test splits share patients")
    guards = {name: 0 for name in VITAL_NAMES}
    guards["negative_added"] = 0
    series = {}
    for pid, group in frame.groupby("PID", sort=False):
        group = group.sort_values("时间")
        pid = str(int(pid))
        rate = group[list(RATE_COLUMNS)].to_numpy(float)
        if not np.isfinite(rate).all() or (rate < 0).any():
            raise ValueError(f"Invalid recorded rate for patient {pid}")
        added_raw = group[list(ADDED_COLUMNS)].to_numpy(float)
        guards["negative_added"] += int((added_raw < 0).sum())
        added = np.maximum(added_raw, 0.0)
        vitals = np.full((len(group), len(VITAL_COLUMNS)), np.nan)
        for column, name in zip(VITAL_COLUMNS, VITAL_NAMES):
            values, count = _guarded(group[column].to_numpy(float), *VITAL_GUARDS[column])
            guards[name] += count
            vitals[:, VITAL_NAMES.index(name)] = values
        values = baseline.reindex([pid]).to_numpy(float)[0]
        series[pid] = dict(rate=rate, added=added, added_signed=added_raw,
                           vitals=vitals, baseline=values,
                           length=len(group),
                           split=0 if pid in train_ids else (1 if pid in test_ids else 2))
    if sum(item["split"] == 2 for item in series.values()):
        raise ValueError("A patient is in neither old split file")
    if len(series) != 363:
        raise ValueError(f"Expected the old 363-patient cohort, found {len(series)}")
    return series, guards


def _rolling_mean(values: np.ndarray, window: int):
    out = np.empty_like(values)
    for i in range(len(values)):
        out[i] = values[max(0, i - window + 1):i + 1].mean(0)
    return out


def _time_since_change(rate: np.ndarray):
    """Hours since the last registered action change at or before each row."""
    age = np.zeros(rate.shape, dtype=float)
    last = np.zeros(rate.shape[1], dtype=int)
    for i in range(len(rate)):
        if i:
            changed = action_change(rate[i - 1], rate[i])
            last = np.where(changed, i, last)
        age[i] = i - last
    return age


def _sequence_features(rate, added, added_signed, vitals, valid_rows=None):
    """(T, F) per-hour causal features.  Row j uses rows <= j only."""
    T = len(rate)
    delta = np.zeros_like(rate)
    if T > 1:
        delta[1:] = rate[1:] - rate[:-1]
        delta[0] = rate[0]
    age = _time_since_change(rate)
    mean6 = _rolling_mean(rate, 6)
    mean24 = _rolling_mean(rate, 24)
    correction = (added_signed < 0).astype(float)
    valid = np.ones((T, 1)) if valid_rows is None else valid_rows[:, None].astype(float)
    hour = (np.arange(T) + 1.0)[:, None] / 72.0
    missing = (~np.isfinite(vitals)).astype(float)
    clean = np.where(np.isfinite(vitals), vitals, np.nan)
    return np.concatenate([rate, (rate > EPS).astype(float), delta, age, mean6, mean24,
                           added, correction,
                           clean[:, 9:10], missing[:, 9:10],
                           clean[:, :9], missing[:, :9],
                           valid, hour], axis=1)


def build(root: Path, history: int = 24, horizons=HORIZONS, group: str = "joint"):
    """Build aligned windows for the old cohort.

    Returns ``rows`` (targets, masks, patient/hour indices) and ``windows``
    (unscaled sequence/static arrays).  A target row ``i`` uses history rows
    ``i-history..i-1``; rows before the record start are zero-padded and marked
    invalid.

    ``group='history'`` blanks the oral and vital channels (values and their
    missingness) while keeping the fluid history and the static baseline.  It is
    the matched information-set control for the reviewer's physiology question.
    """
    if group not in ("joint", "history"):
        raise ValueError("group must be 'joint' or 'history'")
    series, guards = load_series(root)
    patients = sorted(series)
    folds = np.full(len(patients), -1, dtype=int)
    development = [i for i, pid in enumerate(patients) if series[pid]["split"] == 0]
    order = np.array(development)
    rng = np.random.default_rng(20261021)
    shuffled = order.copy()
    rng.shuffle(shuffled)
    for rank, index in enumerate(shuffled):
        folds[index] = rank % 5

    sequences, statics, patient_index, hour_index = [], [], [], []
    target_rate, target_active, previous_rate, actions = [], [], [], []
    cum_targets, cum_masks = [], []
    for index, pid in enumerate(patients):
        item = series[pid]
        rate = item["rate"]
        added = item["added"]
        vitals = item["vitals"]
        features = _sequence_features(rate, added, item["added_signed"], vitals)
        if group == "history":
            features[:, 24:44] = 0.0  # oral (2) + vitals (9) + vital missingness (9)
        T = len(rate)
        if T < 2:
            continue
        for i in range(1, T):
            lo = i - history
            block = features[max(0, lo):i]
            pad = history - len(block)
            window = np.zeros((history, features.shape[1]), dtype=np.float32)
            if pad:
                window[pad:] = block
            else:
                window[:] = block
            sequences.append(window)
            statics.append(np.concatenate([item["baseline"],
                                           (~np.isfinite(item["baseline"])).astype(float),
                                           [item["length"] / 72.0]]))
            patient_index.append(index)
            hour_index.append(i + 1)
            target_rate.append(rate[i])
            target_active.append((rate[i] > EPS).astype(float))
            previous_rate.append(rate[i - 1])
            actions.append(action_classes(rate[i - 1:i], rate[i:i + 1])[0])
            cum = np.full((len(horizons), 3), np.nan)
            mask = np.zeros(len(horizons), dtype=bool)
            for h_index, horizon in enumerate(horizons):
                if i + horizon <= T:
                    cum[h_index] = rate[i:i + horizon].sum(0)
                    mask[h_index] = True
            cum_targets.append(cum)
            cum_masks.append(mask)
    rows = dict(patient=np.asarray(patient_index, int),
                hour=np.asarray(hour_index, int),
                fold=np.asarray([folds[p] for p in patient_index], int),
                split=np.asarray([series[patients[p]]["split"] for p in patient_index], int),
                target_rate=np.asarray(target_rate, float),
                target_active=np.asarray(target_active, float),
                previous_rate=np.asarray(previous_rate, float),
                action=np.asarray(actions, int),
                action_mask=np.ones((len(patient_index), 3), dtype=bool),
                cum_target=np.asarray(cum_targets, float),
                cum_mask=np.asarray(cum_masks, bool))
    windows = dict(sequence=np.asarray(sequences, np.float32),
                   static=np.asarray(statics, float),
                   history=np.asarray([history]))
    return rows, windows, dict(patients=patients, series=series, folds=folds,
                               guards=guards, horizons=tuple(horizons))


def fit_scaler(windows, rows, train):
    """Median/IQR fitted on training windows only, per feature."""
    train = np.asarray(train, bool)
    seq = windows["sequence"][train]
    static = windows["static"][train]
    result = {}
    for name, values in (("sequence", seq), ("static", static)):
        flat = values.reshape(-1, values.shape[-1])
        median = np.nanmedian(flat, axis=0)
        q75 = np.nanpercentile(flat, 75, axis=0)
        q25 = np.nanpercentile(flat, 25, axis=0)
        scale = q75 - q25
        scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
        median = np.where(np.isfinite(median), median, 0.0)
        result[name] = dict(median=median.astype(np.float32),
                            scale=scale.astype(np.float32))
    return result


def transform(windows, scaler):
    """Impute with training medians, scale, and zero out padded history."""
    out = {}
    for name in ("sequence", "static"):
        values = np.asarray(windows[name], float)
        stats = scaler[name]
        missing = ~np.isfinite(values)
        normalized = (np.where(missing, stats["median"], values) - stats["median"]) / stats["scale"]
        normalized = np.clip(normalized, -20, 20)
        if name == "sequence":
            valid = values[:, :, -2].copy()   # raw 0/1 marker, never scaled
            normalized *= valid[:, :, None]
            normalized[:, :, -2] = valid      # keep the marker itself unscaled
        out[name] = normalized.astype(np.float32)
    return out


def patient_weights(patient, mask, train):
    """Each patient contributes equal total weight inside ``mask``."""
    result = np.zeros(mask.shape, dtype=np.float32)
    n = int(np.asarray(train, bool).sum())
    for c in range(mask.shape[1] if mask.ndim > 1 else 1):
        column = mask[:, c] if mask.ndim > 1 else mask
        eligible = np.asarray(train, bool) & column
        ids, counts = np.unique(patient[eligible], return_counts=True)
        for pid, count in zip(ids, counts):
            result[eligible & (patient == pid), c] = n / (len(ids) * count)
    return result
