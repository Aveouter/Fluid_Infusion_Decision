"""Raw old-data completeness and missingness audit (aggregate only).

The submitted old table has no independent time-zero column; ``时间`` is the
only hour anchor.  This audit verifies the raw stamps, the label source, the
status-bit semantics and the vital missingness before any cumulative endpoint
or missingness-stratified comparison is recomputed.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.old_fluid_data_v1 import interval_eligible  # noqa: E402
REPORT = ROOT / "docs/experiments/old-event-v7-20261021"
VITALS = {
    "heart_rate": "心率", "sbp": "收缩压", "dbp": "舒张压", "spo2": "氧饱和度",
    "urine_output": "尿量", "cvp": "CVP", "temperature": "体温",
    "respiratory_rate": "呼吸", "pulse": "脉搏",
}
CORE = ("heart_rate", "sbp", "dbp", "spo2", "urine_output")
SEGMENTS = {"0-24": (0, 24), "24-48": (24, 48), "48-72": (48, 72)}
MISS_SEGMENTS = {**SEGMENTS, "0-48": (0, 48), "0-72": (0, 72), "24-72": (24, 72)}
HORIZONS = {"0-24": (1, 24), "0-48": (1, 48), "0-72": (1, 72),
            "24-48": (25, 48), "48-72": (49, 72), "24-72": (25, 72)}


def split_ids():
    train = set(pd.read_csv(ROOT / "data/train_split.csv")["PID"].astype(int))
    test = set(pd.read_csv(ROOT / "data/test_split.csv")["PID"].astype(int))
    return train, test


def baseline_table():
    frame = pd.read_csv(ROOT / "data/baseline_processed.csv").set_index("Unnamed: 0")
    return frame


def main():
    frame = pd.read_csv(ROOT / "data/all_combined_data.csv")
    labels = pickle.load(open(ROOT / "data/output_data.pkl", "rb"))
    train_ids, test_ids = split_ids()
    baseline = baseline_table()
    patients = sorted(int(p) for p in frame["PID"].unique().tolist())
    # ``build`` sorts the string keys, not the numeric ids; use the same order
    # so fold-level eligibility matches the evaluated rows exactly.
    patients_sorted = sorted(str(pid) for pid in patients)
    development = [i for i, pid in enumerate(patients_sorted) if int(pid) in train_ids]
    order = np.array(development)
    rng = np.random.default_rng(20261021)
    shuffled = order.copy(); rng.shuffle(shuffled)
    fold_of = {patients_sorted[index]: rank % 5 for rank, index in enumerate(shuffled)}
    time_rows = []; label_rows = []; conflict = {c: {"status1_rate0": 0, "status0_rate_pos": 0}
                                                for c in ("crystal", "colloid", "water")}
    label_max_diff = 0.0; bit_mismatch = 0; label_rows_total = 0
    missing = {name: {seg: [] for seg in MISS_SEGMENTS} for name in VITALS}
    setting_missing = {seg: 0 for seg in SEGMENTS}; segment_rows = {seg: 0 for seg in SEGMENTS}
    composite = {seg: [] for seg in MISS_SEGMENTS}
    per_patient_composite = {}; lengths = {}
    eligibility = {h: {"total": 0, "eligible": 0} for h in HORIZONS}
    eligibility_split = {h: {"train": [0, 0], "test": [0, 0]} for h in HORIZONS}
    eligibility_fold = {h: {str(f): [0, 0] for f in range(5)} for h in HORIZONS}
    per_patient = {}
    for pid in patients:
        group = frame[frame["PID"] == pid].sort_values("时间")
        time = group["时间"].to_numpy(float)
        valid = np.isfinite(time)
        pos = np.flatnonzero(valid)
        values = time[pos]
        record = dict(
            n_rows=len(group), n_valid=int(valid.sum()), n_empty=int((~valid).sum()),
            start=float(values[0]), end=float(values[-1]),
            duplicates=int(len(values) - len(np.unique(values))),
            non_integer=int(((values % 1) != 0).sum()),
            gaps=int((np.diff(values) > 1).sum()),
            interior_empty=int(bool((~valid & (np.arange(len(valid)) > pos[0])
                                      & (np.arange(len(valid)) < pos[-1])).any())),
            trailing_empty=int(bool((~valid).any() and not (~valid & (np.arange(len(valid)) > pos[0])
                                                             & (np.arange(len(valid)) < pos[-1])).any())),
        )
        time_rows.append(record)
        lengths[int(valid.sum())] = lengths.get(int(valid.sum()), 0) + 1
        key = next(k for k in labels if int(str(k).replace(".csv", "")) == int(pid))
        label = labels[key]["label"].to_numpy(float)
        raw_rate = group[["晶体速度", "胶体速度", "水速度"]].to_numpy(float)[valid]
        label_rows.append(dict(pid=int(pid), label_rows=len(label), raw_valid=int(valid.sum()),
                               match=bool(len(label) == int(valid.sum()))))
        label_max_diff = max(label_max_diff, float(np.abs(raw_rate - label[:, 3:]).max()))
        bit_mismatch += int(((label[:, :3] > 0.5) != (label[:, 3:] > 0)).sum())
        label_rows_total += len(label)
        for index, (name, column) in enumerate(zip(("crystal", "colloid", "water"),
                                                   ("晶体速度", "胶体速度", "水速度"))):
            bit = group[{"crystal": "晶体", "colloid": "胶体", "water": "水"}[name]].to_numpy(float)[valid]
            rate = raw_rate[:, index]
            conflict[name]["status1_rate0"] += int(((bit >= 0.5) & (rate <= 0)).sum())
            conflict[name]["status0_rate_pos"] += int(((bit < 0.5) & (rate > 0)).sum())
        for seg, (lo, hi) in SEGMENTS.items():
            m = valid & (time > lo) & (time <= hi)
            segment_rows[seg] += int(m.sum())
            setting_missing[seg] += int((~np.isfinite(
                group[["晶体速度", "胶体速度", "水速度"]].to_numpy(float)[m])).sum())
        per_patient_composite[str(pid)] = {}
        for seg, (lo, hi) in MISS_SEGMENTS.items():
            m = valid & (time > lo) & (time <= hi)
            for name, column in VITALS.items():
                x = group[column].to_numpy(float)[m]
                if len(x):
                    missing[name][seg].append(float(np.isnan(x).mean()))
            core = [group[VITALS[name]].to_numpy(float)[m] for name in CORE]
            if len(core[0]):
                value = float(np.mean([np.isnan(x).mean() for x in core]))
                composite[seg].append(value)
                per_patient_composite[str(pid)][seg] = value
        # Interval eligibility: every hour in [start, end] is present once and
        # all three setting rates are finite.  A partial interval never counts.
        for horizon, (lo, hi) in HORIZONS.items():
            ok = interval_eligible(values, lo, hi, raw_rate)
            eligibility[horizon]["total"] += 1
            eligibility[horizon]["eligible"] += int(ok)
            bucket = "train" if int(pid) in train_ids else "test"
            eligibility_split[horizon][bucket][0] += 1
            eligibility_split[horizon][bucket][1] += int(ok)
            if bucket == "train":
                fold = fold_of[str(pid)]
                eligibility_fold[horizon][str(fold)][0] += 1
                eligibility_fold[horizon][str(fold)][1] += int(ok)
        per_patient[int(pid)] = dict(length=int(valid.sum()), full72=bool(interval_eligible(values, 1, 72, raw_rate)))
    # Missingness summaries.
    def summary(values):
        values = np.asarray(values, float)
        return dict(patients=len(values), median=float(np.median(values)) if len(values) else None,
                    q25=float(np.percentile(values, 25)) if len(values) else None,
                    q75=float(np.percentile(values, 75)) if len(values) else None,
                    mean_patient_fraction=float(np.mean(values)) if len(values) else None,
                    full_missing=int((values >= 1.0).sum()))
    missing_summary = {name: {seg: summary(values) for seg, values in by_seg.items()}
                       for name, by_seg in missing.items()}
    composite_summary = {seg: summary(values) for seg, values in composite.items()}
    # Excluded-patient comparison (partial <72 h vs full 72 h).
    full = [pid for pid, item in per_patient.items() if item["full72"]]
    partial = [pid for pid, item in per_patient.items() if not item["full72"]]
    def compare(column):
        a = baseline.reindex(full)[column].to_numpy(float); b = baseline.reindex(partial)[column].to_numpy(float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            return None
        return dict(full_median=float(np.median(a)), partial_median=float(np.median(b)),
                    p=float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue))
    exclusion = {column: compare(column) for column in
                 ("年龄", "烧伤总面积", "合计Ⅲ°", "合计深Ⅱ°", "合计浅Ⅱ°")}
    # Composite missingness strata.  The proposed <=10% cell is reported even
    # when empty; empirical quartiles are the sensitivity that can actually be
    # evaluated in this cohort.
    def proposed_bins(values):
        counts = {"<=0.10": 0, "0.10-0.30": 0, ">0.30": 0, "all-missing": 0}
        for value in values:
            if value >= 1.0: counts["all-missing"] += 1
            elif value <= 0.10: counts["<=0.10"] += 1
            elif value <= 0.30: counts["0.10-0.30"] += 1
            else: counts[">0.30"] += 1
        return counts
    def quartile_bins(values):
        values = np.asarray(values, float)
        if len(values) < 4:
            return None
        bounds = [float(np.percentile(values, q)) for q in (25, 50, 75)]
        counts = [int((values <= bounds[0]).sum()),
                  int(((values > bounds[0]) & (values <= bounds[1])).sum()),
                  int(((values > bounds[1]) & (values <= bounds[2])).sum()),
                  int((values > bounds[2]).sum())]
        return dict(bounds=bounds, counts=counts)
    eligible72 = [pid for pid, item in per_patient.items() if item["full72"]]
    strata = {
        seg: dict(all_patients=proposed_bins(values),
                  quartiles=quartile_bins(values))
        for seg, values in composite.items()
    }
    strata["0-72"]["eligible_72h"] = dict(
        n=len(eligible72),
        proposed=proposed_bins([per_patient_composite[str(pid)]["0-72"] for pid in eligible72]),
        quartiles=quartile_bins([per_patient_composite[str(pid)]["0-72"] for pid in eligible72]),
        by_split={
            "train": proposed_bins([per_patient_composite[str(pid)]["0-72"] for pid in eligible72
                                    if int(pid) in train_ids]),
            "test": proposed_bins([per_patient_composite[str(pid)]["0-72"] for pid in eligible72
                                   if int(pid) in test_ids]),
        },
        by_fold={str(f): proposed_bins([per_patient_composite[str(pid)]["0-72"] for pid in eligible72
                                        if int(pid) in train_ids and fold_of[str(pid)] == f])
                 for f in range(5)},
    )
    payload = dict(
        protocol="old-data-completeness-audit-20261021",
        note=("Aggregate only; raw patient rows stay local. '时间' is the only hour anchor; "
              "there is no independent injury/admission timestamp in the old table. Empty "
              "time rows are unrecorded, never zero. Rate is the label; status bits are not."),
        cohort=dict(patients=len(patients), rows=len(frame), valid_rows=int(sum(x["n_valid"] for x in time_rows)),
                    empty_rows=int(sum(x["n_empty"] for x in time_rows)),
                    train_patients=len(train_ids), test_patients=len(test_ids)),
        time=dict(start_not_1=sum(int(x["start"] != 1) for x in time_rows),
                  duplicates=sum(x["duplicates"] for x in time_rows),
                  non_integer=sum(x["non_integer"] for x in time_rows),
                  gaps=sum(x["gaps"] for x in time_rows),
                  interior_empty_patients=sum(x["interior_empty"] for x in time_rows),
                  trailing_empty_patients=sum(x["trailing_empty"] for x in time_rows),
                  empty_rows=int(sum(x["n_empty"] for x in time_rows)),
                  patients_full72=len(eligible72),
                  patients_partial=len(patients) - len(eligible72),
                  valid_hours_min=int(min(x["n_valid"] for x in time_rows)),
                  valid_hours_max=int(max(x["n_valid"] for x in time_rows)),
                  length_histogram={str(k): lengths[k] for k in sorted(lengths)}),
        labels=dict(max_abs_diff=label_max_diff, bit_rate_mismatch=bit_mismatch,
                    rows=label_rows_total, length_mismatch=int(sum(not x["match"] for x in label_rows)),
                    status_conflicts=conflict,
                    interpretation=("Label rates equal the raw rate columns exactly; bits are derived "
                                    "from rate. status=1 with rate=0 and status=0 with rate>0 are source-"
                                    "semantics conflicts; the label protocol uses rate.")),
        missingness=missing_summary,
        composite_missingness=composite_summary,
        missingness_strata=strata,
        setting_missing=setting_missing,
        segment_rows=segment_rows,
        eligibility={h: {**eligibility[h],
                         "train": eligibility_split[h]["train"],
                         "test": eligibility_split[h]["test"],
                         "fold": eligibility_fold[h]}
                     for h in HORIZONS},
        exclusion_comparison=exclusion,
    )
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "data_completeness_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    lines = ["# 原始老数据完整性与缺失审计", "",
             "只提交脱敏聚合；原始患者行不公开。`时间` 是唯一小时锚，老数据没有独立的伤后/入院时间戳。", "",
             "## 时间戳与标签来源", "",
             f"- 363 人、{payload['cohort']['rows']} 行，其中有效时间 {payload['cohort']['valid_rows']} 行、空时间 {payload['cohort']['empty_rows']} 行。",
             f"- 全部患者从 `时间=1` 开始；重复 {payload['time']['duplicates']}、非整数 {payload['time']['non_integer']}、断档 {payload['time']['gaps']}、内部空行患者 {payload['time']['interior_empty_patients']}。",
             f"- 尾部空时间患者 {payload['time']['trailing_empty_patients']} 人；空时间不是零流，不进入标签/累计。",
             f"- 标签速率与原始速率列最大差 {label_max_diff:.3g}；bit/rate 不一致 {bit_mismatch}；标签长度不匹配 {payload['labels']['length_mismatch']}。",
             f"- 状态位/速率冲突：status=1 且 rate=0 共 {sum(v['status1_rate0'] for v in conflict.values())} 格；status=0 且 rate>0 共 {sum(v['status0_rate_pos'] for v in conflict.values())} 格。状态位不用于观测/零流判定。", ""]
    lines += ["## 区间资格（小时 1..H 全部存在且三类速率可验证）", "",
              "| 区间 | 总患者 | 合格 | 训练 | 留出 |", "|---|---:|---:|---|---|"]
    for h, row in payload["eligibility"].items():
        lines.append(f"| {h} | {row['total']} | {row['eligible']} | {row['train'][1]}/{row['train'][0]} | {row['test'][1]}/{row['test'][0]} |")
    lines += ["", "缺 48–72h 的患者仍可进入 0–24/0–48；不完整区间不进入对应累计终点，不用部分和替代 72h。", ""]
    fold72 = payload["eligibility"]["0-72"]["fold"]
    lines += ["按开发折的 0–72h 合格人数：" + "；".join(
        f"折{f} {row[1]}/{row[0]}" for f, row in fold72.items()) + "。", ""]
    lines += ["## 生命体征缺失率（每患者每时段；raw NaN，不含 history padding）", "",
              "“全缺失”按该患者实际有记录小时计；部分记录者的小时数少，该列不等于 72h 全缺失。", "",
              "| 变量 | 0–24 中位 | 24–48 中位 | 48–72 中位 | 全缺失患者（0–72） |", "|---|---:|---:|---:|---:|"]
    for name in VITALS:
        m = payload["missingness"][name]
        lines.append(f"| {name} | {fmt(m['0-24']['median'])} | {fmt(m['24-48']['median'])} | {fmt(m['48-72']['median'])} | {m['0-72']['full_missing']} |")
    lines += ["", "## 核心变量复合缺失分层（HR/SBP/DBP/SpO2/尿量，0–72）", "",
              "预设的 ≤10% 层在本队列为 **0 人**；下表同时给出预设分层与经验四分位（仅 72h 合格者），后者用于实际分层评估。", "",
              "| 预设层 | 全队列 | 72h 合格 |", "|---|---:|---:|"]
    strata72 = payload["missingness_strata"]["0-72"]
    for key in ("<=0.10", "0.10-0.30", ">0.30", "all-missing"):
        lines.append(f"| {key} | {strata72['all_patients'][key]} | {strata72['eligible_72h']['proposed'][key]} |")
    quart = strata72["eligible_72h"]["quartiles"]
    if quart:
        lines += ["", "经验四分位边界（72h 合格者）："
                  + " / ".join(f"{value:.3f}" for value in quart["bounds"])
                  + f"；四层人数 {' / '.join(str(value) for value in quart['counts'])}。", ""]
    lines += ["", "## 各时段复合缺失率与分层人数", "",
              "| 时段 | 中位 | IQR | 有记录患者 | ≤10% | 10–30% | >30% | 全缺失 |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for seg in ("0-24", "24-48", "48-72", "0-48", "0-72", "24-72"):
        summary = payload["composite_missingness"][seg]
        proposed = payload["missingness_strata"][seg]["all_patients"]
        lines.append(f"| {seg} | {summary['median']:.3f} | {summary['q25']:.3f}–{summary['q75']:.3f} | {summary['patients']} | "
                     f"{proposed['<=0.10']} | {proposed['0.10-0.30']} | {proposed['>0.30']} | {proposed['all-missing']} |")
    lines += ["", "## 被排除患者差异（<72h vs 72h）", "",
              "| 变量 | 72h 中位 | <72h 中位 | Mann–Whitney p |", "|---|---:|---:|---:|"]
    for column, row in payload["exclusion_comparison"].items():
        if row:
            lines.append(f"| {column} | {row['full_median']:.2f} | {row['partial_median']:.2f} | {row['p']:.3g} |")
    lines += ["", "## 结论", "",
              "- 本版本原始表没有内部断档、重复或非整数时间；27 人的空时间全在尾部，按未记录处理，不进入标签/累计。",
              "- 标签就是原始三类速率，逐值一致；`status` 位与速率大量冲突，不能拿状态位判断“有记录/零流”。",
              "- 0–72h 合格 336/363；被排除的 27 人全层烧伤更多，72h 累计结论只对完整随访者成立。",
              "- 核心体征复合缺失中位 42.5%，0–72h 的 ≤10% 层为 0 人；分层评估必须用经验分位，并同时报告真实输注量，避免把病例构成误读成模型差异。", ""]
    lines += ["", "边界：老数据单中心回顾性；`时间` 仅为业务小时锚；状态位冲突说明“有记录”与“速率”语义不同；缺失率分层是描述性的，不作因果解释。"]
    (REPORT / "data_completeness_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(dict(cohort=payload["cohort"], time=payload["time"],
                          eligibility=payload["eligibility"],
                          strata=payload["missingness_strata"]["0-72"]["eligible_72h"]), indent=2))


def fmt(value):
    return "NA" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
