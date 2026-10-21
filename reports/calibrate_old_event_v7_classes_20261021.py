"""Per-fluid, per-predicted-action level calibration on training folds.

A single per-fluid offset (``calibration.json``) lowered P1 but raised hourly
MAE sharply.  This variant fits offsets by predicted action class so that
under-predicted starts/ups are raised without shifting every positive hour.
It remains a post-hoc calibrated variant, not a retrained model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.calibrate_old_event_v7_20261021 import apply_offset, oof_rate
from reports.evaluate_old_event_v7_20261021 import SEEDS, load_prediction, pool, summarize
from utils.old_fluid_data_v1 import action_classes, build

V7 = ROOT / ".audit/old_event_v7_prev"
REPORT = ROOT / "docs/experiments/old-event-v7-20261021"
CLASSES = ("hold", "start", "stop", "up", "down")


def matrix(rows, mask, rate, series_rate, horizon=72):
    patients, true_total, base_total, counts = [], [], [], []
    for pid in np.unique(rows["patient"][mask]):
        use = np.flatnonzero((rows["patient"] == pid) & (rows["hour"] <= horizon))
        if len(use) != horizon - 1:
            continue
        order = use[np.argsort(rows["hour"][use])]
        if not np.array_equal(rows["hour"][order], np.arange(2, horizon + 1)):
            continue
        true = series_rate[pid][0] + rows["target_rate"][order].sum(0)
        base = series_rate[pid][0] + rate[order].sum(0)
        classes = action_classes(rows["previous_rate"][order], rate[order])
        count = np.zeros((3, 5))
        for c in range(3):
            for k in range(5):
                count[c, k] = ((rate[order, c] > 0) & (classes[:, c] == k)).sum()
        patients.append(int(pid)); true_total.append(true); base_total.append(base); counts.append(count)
    return np.asarray(true_total), np.asarray(base_total), np.asarray(counts)


def fit(rows, mask, rate, series_rate):
    true, base, counts = matrix(rows, mask, rate, series_rate)
    delta = np.zeros((3, 5))
    grids = {0: np.arange(-30, 60.5, 2.5), 1: np.arange(0, 400.5, 5.0),
             2: np.zeros(1), 3: np.arange(0, 300.5, 5.0), 4: np.arange(-100, 100.5, 5.0)}
    for _ in range(3):
        for c in range(3):
            for k in range(5):
                if k == 2:
                    continue
                best, best_value = delta[c, k], np.inf
                for value in grids[k]:
                    candidate = delta.copy(); candidate[c, k] = value
                    error = np.abs((base.sum(1) + (candidate[None] * counts).sum((1, 2))) - true.sum(1))
                    score = float(error.mean())
                    if score < best_value:
                        best, best_value = float(value), score
                delta[c, k] = best
    error = np.abs((base.sum(1) + (delta[None] * counts).sum((1, 2))) - true.sum(1))
    return delta, float(error.mean())


def apply_class(rows, rate, delta):
    classes = action_classes(rows["previous_rate"], rate)
    out = rate.copy()
    for c in range(3):
        for k in range(5):
            use = (out[:, c] > 0) & (classes[:, c] == k)
            out[use, c] += delta[c, k]
    return np.maximum(out, 0.0)


def p1(rows, mask, rate, series_rate):
    true, base, _ = matrix(rows, mask, rate, series_rate)
    error = np.abs(base.sum(1) - true.sum(1))
    return dict(n=len(error), total_mae=float(error.mean()), bias=float((base.sum(1) - true.sum(1)).mean()))


def main():
    rows, windows, meta = build(ROOT)
    series_rate = {i: meta["series"][pid]["rate"] for i, pid in enumerate(meta["patients"])}
    development = rows["split"] == 0
    test = rows["split"] == 1
    rate = oof_rate(rows, development)
    selection = np.flatnonzero(test)
    records = [load_prediction(V7 / f"test_two_branch_seed{seed}", selection) for seed in SEEDS]
    test_rate = np.full_like(rows["target_rate"], np.nan, dtype=float)
    test_rate[selection] = pool(records, "two_branch")
    result = {"protocol": "old-event-v7-class-calibration-20261021",
              "note": "Offsets by predicted action class, fitted on calibration folds only; post-hoc calibrated variant.",
              "development": {}, "historical_test": {}}
    calibrated_oof = np.full_like(rate, np.nan)
    for fold in range(5):
        held = development & (rows["fold"] == fold)
        calibration = development & (rows["fold"] != fold)
        delta, objective = fit(rows, calibration, rate, series_rate)
        calibrated_oof[held] = apply_class(rows, rate, delta)[held]
        result["development"][str(fold)] = dict(delta=delta.tolist(), calibration_p1=objective)
    delta_all, objective_all = fit(rows, development, rate, series_rate)
    calibrated_test = apply_class(rows, test_rate, delta_all)
    result["development"]["all"] = dict(delta=delta_all.tolist(), calibration_p1=objective_all)
    result["historical_test"] = dict(delta=delta_all.tolist(), calibration_p1=objective_all)
    for name, mask, raw, calibrated in (
        ("development_folds1to4", development & (rows["fold"] > 0), rate, calibrated_oof),
        ("historical_test", test, test_rate, calibrated_test)):
        result[name] = dict(
            raw_p1=p1(rows, mask, raw, series_rate),
            calibrated_p1=p1(rows, mask, calibrated, series_rate),
            raw={key: value for key, value in summarize(
                "raw", rows["patient"][mask], rows["previous_rate"][mask], rows["target_rate"][mask],
                raw[mask], rows["action_mask"][mask]).items()
                 if key in ("setting_mae", "change_hour_mae", "strict_action_f1",
                            "zero_flow_accuracy", "false_change_rate")},
            calibrated={key: value for key, value in summarize(
                "cal", rows["patient"][mask], rows["previous_rate"][mask], rows["target_rate"][mask],
                calibrated[mask], rows["action_mask"][mask]).items()
                 if key in ("setting_mae", "change_hour_mae", "strict_action_f1",
                            "zero_flow_accuracy", "false_change_rate")})
    (REPORT / "class_calibration.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
    print(json.dumps({
        "delta": delta_all.tolist(),
        "dev_raw_p1": result["development_folds1to4"]["raw_p1"]["total_mae"],
        "dev_cal_p1": result["development_folds1to4"]["calibrated_p1"]["total_mae"],
        "test_raw_p1": result["historical_test"]["raw_p1"]["total_mae"],
        "test_cal_p1": result["historical_test"]["calibrated_p1"]["total_mae"],
        "dev_raw_mae": result["development_folds1to4"]["raw"]["setting_mae"],
        "dev_cal_mae": result["development_folds1to4"]["calibrated"]["setting_mae"],
        "test_raw_mae": result["historical_test"]["raw"]["setting_mae"],
        "test_cal_mae": result["historical_test"]["calibrated"]["setting_mae"],
    }, indent=2))


if __name__ == "__main__":
    main()
