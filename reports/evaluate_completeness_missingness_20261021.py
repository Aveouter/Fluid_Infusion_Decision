"""Interval eligibility, missingness stratification and physiology controls.

This re-evaluates V7.1 / V7.1-M and the matched information-set controls on
the *raw-time* eligible set.  Cumulative endpoints are interval exposure
(rate x 1 h), not delivered dose.  Every table is paired on the same patients,
folds and seeds; the missingness strata are descriptive and are not causal
estimates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.evaluate_old_event_v7_20261021 import (SEEDS, bootstrap, load_prediction,
                                                     pool_parts, decode_parts, summarize,
                                                     per_patient_counts, f1_from, clean)
from reports.issue81_style_v7_20261021 import activity_table, ci
from utils.old_fluid_data_v1 import build, interval_eligible

REPORT = ROOT / "docs/experiments/old-event-v7-20261021"
V71 = ROOT / ".audit/old_event_v7_prev"
V71M = ROOT / ".audit/old_event_v7_mean05"
CONTROLS = {
    # The original history run already blanked indices 24:44 (oral/vital
    # values and indicators), which is the correct information set; its
    # checkpoints replay exactly after the raw-time loader change.
    "history": ROOT / ".audit/old_event_v7_history",
    "missing": ROOT / ".audit/old_event_v7_missing",
    "values": ROOT / ".audit/old_event_v7_values",
}
INTERVALS = {"0-24": (1, 24), "0-48": (1, 48), "0-72": (1, 72),
             "24-48": (25, 48), "48-72": (49, 72), "24-72": (25, 72)}
SEGMENT = {"0-24": "0-24", "0-48": "0-48", "0-72": "0-72",
           "24-48": "24-48", "48-72": "48-72", "24-72": "24-72"}
CORE = (1, 4, 5, 7, 8)  # heart_rate, sbp, dbp, spo2, urine_output in VITAL_NAMES
METHODS = ("v71", "v71_mean", "persistence", "zero", "history", "missing", "values")


def load_arrays(root, rows, *, test=False):
    """Pooled seed predictions for development folds 1-4 or the test split."""
    if not root.exists():
        return None
    rate = np.full_like(rows["target_rate"], np.nan, dtype=float)
    event = np.full_like(rows["target_rate"], np.nan, dtype=float)
    active = np.full_like(rows["target_rate"], np.nan, dtype=float)
    if test:
        selection = np.flatnonzero(rows["split"] == 1)
        try:
            records = [load_prediction(root / f"test_two_branch_seed{seed}", selection)
                       for seed in SEEDS]
        except FileNotFoundError:
            return None
        parts = pool_parts(records, "two_branch")
        rate[selection] = decode_parts(parts, "two_branch")
        event[selection] = parts["event_probability"]
        active[selection] = parts["active_probability"]
    else:
        development = rows["split"] == 0
        for fold in range(1, 5):
            selection = np.flatnonzero(development & (rows["fold"] == fold))
            try:
                records = [load_prediction(root / f"fold{fold}_two_branch_seed{seed}", selection)
                           for seed in SEEDS]
            except FileNotFoundError:
                return None
            parts = pool_parts(records, "two_branch")
            rate[selection] = decode_parts(parts, "two_branch")
            event[selection] = parts["event_probability"]
            active[selection] = parts["active_probability"]
    return rate, event, active


def missing_fraction(series_item, start, end):
    time = series_item["time"]
    use = (time > start) & (time <= end)
    values = series_item["vitals"][use][:, CORE]
    if not len(values):
        return None
    return float(np.mean(np.isnan(values)))


def interval_entries(rows, series, rate, patients, start, end):
    entries = []
    for pid in patients:
        item = series[int(pid)]
        if not interval_eligible(item["time"], start, end, item["rate"]):
            continue
        if start == 1:
            need = np.arange(2, end + 1)
            use = np.flatnonzero((rows["patient"] == pid) & (rows["hour"] >= 2) & (rows["hour"] <= end))
            if len(use) != len(need):
                continue
            order = use[np.argsort(rows["hour"][use])]
            if not np.array_equal(rows["hour"][order], need):
                continue
            true = item["rate"][0] + rows["target_rate"][order].sum(0)
            pred = item["rate"][0] + rate[order].sum(0)
        else:
            need = np.arange(start, end + 1)
            use = np.flatnonzero((rows["patient"] == pid) & (rows["hour"] >= start) & (rows["hour"] <= end))
            if len(use) != len(need):
                continue
            order = use[np.argsort(rows["hour"][use])]
            if not np.array_equal(rows["hour"][order], need):
                continue
            true = rows["target_rate"][order].sum(0)
            pred = rate[order].sum(0)
        entries.append(dict(patient=int(pid), true=true, pred=pred))
    return entries


def entry_metrics(entries):
    if not entries:
        return dict(patients=0, total_mae=None, total_bias=None, total_mae_ci=None,
                    total_tolerance_200=None, total_tolerance_10=None,
                    total_relative_mae=None, total_mae_over_mean_true=None,
                    fluid_mae=None, fluid_bias=None, fluid_mae_ci=None,
                    true_interval_total_mean=None, pred_interval_total_mean=None)
    true = np.asarray([item["true"] for item in entries], float)
    pred = np.asarray([item["pred"] for item in entries], float)
    total_true, total_pred = true.sum(1), pred.sum(1)
    error = total_pred - total_true
    absolute = np.abs(error)
    fluid_error = np.abs(pred - true)
    relative = np.abs(total_true) > 1e-6
    total_mae, total_lo, total_hi = ci(absolute)
    fluid_ci = []
    for c in range(3):
        _, lo, hi = ci(fluid_error[:, c])
        fluid_ci.append([lo, hi])
    return dict(
        patients=len(entries),
        total_mae=total_mae,
        total_bias=float(error.mean()),
        total_mae_ci=[total_lo, total_hi],
        total_tolerance_200=float((absolute <= 200.0).mean()),
        total_tolerance_10=(float((absolute[relative] <= 0.10 * np.abs(total_true[relative])).mean())
                            if relative.any() else None),
        total_relative_mae=(float(np.mean(absolute[relative] / np.abs(total_true[relative])))
                            if relative.any() else None),
        total_mae_over_mean_true=(float(absolute[relative].mean() / np.abs(total_true[relative]).mean())
                                  if relative.any() else None),
        fluid_mae=fluid_error.mean(0).tolist(),
        fluid_bias=(pred - true).mean(0).tolist(),
        fluid_mae_ci=fluid_ci,
        true_interval_total_mean=float(total_true.mean()),
        pred_interval_total_mean=float(total_pred.mean()),
    )


def per_patient_mae(patient, truth, pred, mask):
    ids = np.unique(patient)
    out = np.full(len(ids), np.nan)
    for i, pid in enumerate(ids):
        use = patient == pid
        out[i] = np.nanmean(np.abs(pred[use] - truth[use]))
    return ids, out


def pooled_f1(summary):
    counts = summary["patient_counts"]
    tp = counts["tp"].sum(0); fp = counts["fp"].sum(0); fn = counts["fn"].sum(0)
    return np.nanmean(f1_from(tp, fp, fn))


def bootstrap_f1(summary):
    counts = summary["patient_counts"]
    rng = np.random.default_rng(20261021)
    draws = rng.integers(len(counts["tp"]), size=(2000, len(counts["tp"])))
    values = []
    for draw in draws:
        tp = counts["tp"][draw].sum(0); fp = counts["fp"][draw].sum(0); fn = counts["fn"][draw].sum(0)
        values.append(np.nanmean(f1_from(tp, fp, fn)))
    return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]


def class_recall(summary, k):
    counts = summary["patient_class_counts"]
    tp = counts["tp"][:, :, k].sum(0); fn = counts["fn"][:, :, k].sum(0)
    support = counts["support"][:, :, k].sum(0)
    return {name: (float(tp[c] / support[c]) if support[c] else None)
            for c, name in enumerate(("crystal", "colloid", "water"))}


def method_metrics(rows, rate, active, mask, entries):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return None
    patient = rows["patient"][idx]
    previous = rows["previous_rate"][idx]
    truth = rows["target_rate"][idx]
    pred = rate[idx]
    prob = active[idx] if active is not None else (pred > 0).astype(float)
    summary = summarize("method", patient, previous, truth, pred, np.ones_like(truth, dtype=bool))
    activity = activity_table(patient, truth, pred, prob)
    _, mae_patient = per_patient_mae(patient, truth, pred, np.ones_like(truth, dtype=bool))
    return dict(
        setting_mae=summary["setting_mae"],
        setting_mae_ci=[float(np.nanpercentile(mae_patient, 2.5)),
                        float(np.nanpercentile(mae_patient, 97.5))],
        setting_bias=summary["setting_bias"],
        change_hour_mae=summary["change_hour_mae"],
        zero_flow_accuracy=summary["zero_flow_accuracy"],
        strict_action_f1=summary["strict_action_f1"],
        strict_action_f1_ci=bootstrap_f1(summary),
        activity_macro_f1=activity["macro_f1"],
        activity_macro_f1_ci=activity["macro_f1_ci"],
        activity_micro_f1=activity["micro_f1"],
        start_recall=class_recall(summary, 1),
        stop_recall=class_recall(summary, 2),
        up_recall=class_recall(summary, 3),
        down_recall=class_recall(summary, 4),
        per_class=summary["per_class"],
        interval=entry_metrics(entries),
    )


def assign_proposed(value):
    if value is None:
        return None
    if value >= 1.0:
        return "all-missing"
    if value <= 0.10:
        return "<=0.10"
    if value <= 0.30:
        return "0.10-0.30"
    return ">0.30"


def assign_quartile(value, bounds):
    if value is None:
        return None
    if value <= bounds[0]:
        return "Q1"
    if value <= bounds[1]:
        return "Q2"
    if value <= bounds[2]:
        return "Q3"
    return "Q4"


def split_masks(rows):
    return {"development_folds1to4": (rows["split"] == 0) & (rows["fold"] > 0),
            "historical_test": rows["split"] == 1}


def slim_method(item):
    keys = ("setting_mae", "setting_bias", "change_hour_mae", "zero_flow_accuracy",
            "strict_action_f1", "activity_macro_f1", "activity_micro_f1",
            "start_recall", "stop_recall", "up_recall", "down_recall", "interval")
    return {key: item.get(key) for key in keys}


def slim_payload(payload):
    """Keep the reviewer-facing aggregate compact; full detail stays local."""
    keep = ("v71", "v71_mean", "persistence")
    for split, intervals in payload["methods"].items():
        for interval, block in intervals.items():
            payload["methods"][split][interval] = {
                name: slim_method(block[name]) for name in keep if name in block}
    for split, intervals in payload["strata"].items():
        for interval, block in intervals.items():
            for label, group in block.get("groups", {}).items():
                if group.get("evaluated"):
                    group["methods"] = {name: slim_method(group["methods"][name])
                                        for name in keep if name in group["methods"]}
    return payload


def main():
    rows, windows, meta = build(ROOT)
    series = {i: meta["series"][pid] for i, pid in enumerate(meta["patients"])}
    methods = {"v71": load_arrays(V71, rows), "v71_mean": load_arrays(V71M, rows)}
    for name, root in CONTROLS.items():
        methods[name] = load_arrays(root, rows)
    # test arrays are separate because each root has its own test fits.
    test_methods = {"v71": load_arrays(V71, rows, test=True),
                    "v71_mean": load_arrays(V71M, rows, test=True)}
    for name, root in CONTROLS.items():
        test_methods[name] = load_arrays(root, rows, test=True)
    # Baselines use the same row arithmetic.
    for bundle in (methods, test_methods):
        bundle["persistence"] = (rows["previous_rate"].astype(float), None, None)
        bundle["zero"] = (np.zeros_like(rows["target_rate"]), None, None)
    payload = dict(protocol="old-event-v7-interval-missingness-20261021",
                   note=("Raw-time interval eligibility; interval totals are rate x 1 h "
                         "exposure, not delivered dose. Development folds 1-4 and the "
                         "historical test are separate. Missingness strata are descriptive."),
                   eligibility={}, methods={}, physiology={}, strata={})
    data = {}
    for split, base in split_masks(rows).items():
        bundle = test_methods if split == "historical_test" else methods
        patients = np.unique(rows["patient"][base])
        data[split] = {}
        payload["eligibility"][split] = {}
        payload["methods"][split] = {}
        payload["strata"][split] = {}
        for interval, (start, end) in INTERVALS.items():
            entries = {name: interval_entries(rows, series, bundle[name][0], patients, start, end)
                       for name in METHODS if bundle.get(name) is not None}
            n_base = len(patients)
            payload["eligibility"][split][interval] = dict(
                n_total=int(n_base),
                n_eligible=int(len(entries["v71"])) if "v71" in entries else None,
                n_excluded=(int(n_base - len(entries["v71"])) if "v71" in entries else None),
                interval_hours=end - start + 1,
                includes_initial_hour=(start == 1),
            )
            data[split][interval] = dict(entries=entries, patients=patients)
            payload["methods"][split][interval] = {}
            row_mask_all = base & (rows["hour"] >= max(2, start)) & (rows["hour"] <= end)
            for name in METHODS:
                if bundle.get(name) is None:
                    continue
                active = bundle[name][2]
                payload["methods"][split][interval][name] = method_metrics(
                    rows, bundle[name][0], active, row_mask_all, entries.get(name, []))
            # Missingness strata on the same eligible patients.
            seg_start, seg_end = [int(x) for x in SEGMENT[interval].split("-")]
            missing = {int(pid): missing_fraction(series[int(pid)], seg_start, seg_end) for pid in patients}
            eligible_ids = [item["patient"] for item in entries.get("v71", [])]
            values = np.asarray([missing[pid] for pid in eligible_ids if missing.get(pid) is not None])
            payload["strata"][split][interval] = dict(
                eligible_missing_median=(float(np.median(values)) if len(values) else None),
                eligible_missing_iqr=([float(np.percentile(values, 25)), float(np.percentile(values, 75))]
                                      if len(values) else None),
            )
            # Proposed bins are fixed; empirical quartile boundaries come from
            # development folds 1-4 and are applied to both splits.
            if split == "development_folds1to4":
                four = np.percentile(values, [25, 50, 75]) if len(values) >= 4 else None
                payload["strata"][split][interval]["quartile_bounds"] = (
                    [float(x) for x in four] if four is not None else None)
                data[split][interval]["quartile_bounds"] = four
            else:
                four = data["development_folds1to4"][interval].get("quartile_bounds")
            strata = {}
            for label in ("<=0.10", "0.10-0.30", ">0.30", "all-missing"):
                ids = [pid for pid in eligible_ids if assign_proposed(missing.get(pid)) == label]
                strata[label] = ids
            if four is not None:
                for label in ("Q1", "Q2", "Q3", "Q4"):
                    ids = [pid for pid in eligible_ids
                           if assign_quartile(missing.get(pid), four) == label]
                    strata[label] = ids
            payload["strata"][split][interval]["eligible_counts"] = {
                label: len(ids) for label, ids in strata.items()}
            payload["strata"][split][interval]["groups"] = {}
            for label, ids in strata.items():
                subset = np.isin(rows["patient"], ids)
                row_mask = base & subset & (rows["hour"] >= max(2, start)) & (rows["hour"] <= end)
                truth_group = rows["target_rate"][row_mask]
                joint_entries = [item for item in entries.get("v71", []) if item["patient"] in set(ids)]
                description = None
                if len(ids):
                    description = dict(
                        truth_mean_rate=float(truth_group.mean()),
                        truth_positive_rate=float((truth_group > 0).mean()),
                        truth_zero_rate=float((truth_group <= 1e-6).mean()),
                        truth_row_total_mean=float(truth_group.sum(1).mean()),
                        true_interval_total_mean=(float(np.mean([item["true"].sum() for item in joint_entries]))
                                                  if joint_entries else None),
                        missing_median=float(np.median([missing[pid] for pid in ids
                                                        if missing.get(pid) is not None])),
                    )
                if len(ids) < 5:
                    payload["strata"][split][interval]["groups"][label] = dict(
                        patients=len(ids), evaluated=False, description=description,
                        reason="fewer than 5 eligible patients")
                    continue
                group = {}
                for name in METHODS:
                    if bundle.get(name) is None:
                        continue
                    subset_entries = [item for item in entries.get(name, []) if item["patient"] in set(ids)]
                    group[name] = method_metrics(rows, bundle[name][0], bundle[name][2],
                                                 row_mask, subset_entries)
                payload["strata"][split][interval]["groups"][label] = dict(
                    patients=len(ids), evaluated=True, description=description, methods=group)
            # Paired physiology contrasts, same patients/folds/seeds.
            payload["physiology"].setdefault(split, {})[interval] = {}

            def contrast(ids):
                subset = np.isin(rows["patient"], ids)
                row_mask = base & subset & (rows["hour"] >= max(2, start)) & (rows["hour"] <= end)
                joint = summarize("joint", rows["patient"][row_mask], rows["previous_rate"][row_mask],
                                  rows["target_rate"][row_mask], bundle["v71"][0][row_mask],
                                  np.ones_like(rows["target_rate"][row_mask], dtype=bool))
                result = {}
                for name in ("history", "missing", "values"):
                    if bundle.get(name) is None:
                        result[name] = dict(available=False, reason="control predictions unavailable")
                        continue
                    control = summarize(name, rows["patient"][row_mask], rows["previous_rate"][row_mask],
                                        rows["target_rate"][row_mask], bundle[name][0][row_mask],
                                        np.ones_like(rows["target_rate"][row_mask], dtype=bool))
                    paired = bootstrap(joint, control)
                    ids_j, mae_j = per_patient_mae(rows["patient"][row_mask],
                                                   rows["target_rate"][row_mask],
                                                   bundle["v71"][0][row_mask],
                                                   np.ones(int(row_mask.sum()), dtype=bool))
                    ids_c, mae_c = per_patient_mae(rows["patient"][row_mask],
                                                   rows["target_rate"][row_mask],
                                                   bundle[name][0][row_mask],
                                                   np.ones(int(row_mask.sum()), dtype=bool))
                    delta_mae, delta_lo, delta_hi = ci(mae_j - mae_c)
                    result[name] = dict(
                        available=True,
                        delta_setting_mae=delta_mae,
                        delta_setting_mae_ci=[delta_lo, delta_hi],
                        delta_strict_f1=paired["strict_action_f1"]["difference"],
                        delta_strict_f1_ci=[paired["strict_action_f1"]["ci_low"],
                                            paired["strict_action_f1"]["ci_high"]],
                    )
                return result

            if len(eligible_ids) >= 5:
                payload["physiology"][split][interval]["all"] = contrast(eligible_ids)
            for label, ids in strata.items():
                if len(ids) < 5:
                    continue
                payload["physiology"][split][interval][label] = contrast(ids)
    REPORT.mkdir(parents=True, exist_ok=True)
    (ROOT / ".audit/interval_missingness_full.json").write_text(
        json.dumps(clean(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    slim_payload(payload)
    (REPORT / "interval_missingness.json").write_text(
        json.dumps(clean(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(clean({split: {
        interval: payload["eligibility"][split][interval] for interval in INTERVALS}
        for split in ("development_folds1to4", "historical_test")}), indent=2))


if __name__ == "__main__":
    main()
