"""Train-fold-only level calibration for the V7.1 cumulative endpoint.

The raw model minimizes hourly L1 (a conditional median), which systematically
underestimates cumulative exposure.  This script fits one additive offset per
fluid on *positive* predictions using only the calibration folds, then evaluates
the held-out fold/test.  It is a post-hoc calibration, not a retrained model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.evaluate_old_event_v7_20261021 import SEEDS, load_prediction, pool, summarize
from reports.issue81_style_v7_20261021 import cumulative_entries, cumulative_table
from utils.old_fluid_data_v1 import build

V7 = ROOT / ".audit/old_event_v7_prev"
REPORT = ROOT / "docs/experiments/old-event-v7-20261021"
FLUIDS = ("crystal", "colloid", "water")


def oof_rate(rows, development):
    rate = np.full_like(rows["target_rate"], np.nan, dtype=float)
    for fold in range(5):
        selection = np.flatnonzero(development & (rows["fold"] == fold))
        records = [load_prediction(V7 / f"fold{fold}_two_branch_seed{seed}", selection)
                   for seed in SEEDS]
        rate[selection] = pool(records, "two_branch")
    return rate


def apply_offset(rate, delta):
    out = rate.copy()
    for c in range(3):
        out[:, c] = np.where(out[:, c] > 0, out[:, c] + delta[c], 0.0)
    return out


def objective(rows, mask, rate, series_rate, delta):
    calibrated = apply_offset(rate, delta)
    entries = cumulative_entries(rows, mask, calibrated, series_rate, 72)
    true = np.asarray([e[1] for e in entries])
    pred = np.asarray([e[2] for e in entries])
    return float(np.abs(pred.sum(1) - true.sum(1)).mean())


def fit_delta(rows, mask, rate, series_rate):
    grid = np.arange(0.0, 100.0 + 0.5, 0.5)
    delta = np.zeros(3)
    for c in range(3):
        best, best_value = 0.0, np.inf
        for value in grid:
            candidate = delta.copy(); candidate[c] = value
            # Only the current fluid matters for its own offset; fit per fluid on
            # the total cumulative error so cross-fluid cancellation is respected.
            score = objective(rows, mask, rate, series_rate, candidate)
            if score < best_value:
                best, best_value = float(value), score
        delta[c] = best
    return delta, objective(rows, mask, rate, series_rate, delta)


def main():
    rows, windows, meta = build(ROOT)
    series_rate = {i: meta["series"][pid]["rate"] for i, pid in enumerate(meta["patients"])}
    development = rows["split"] == 0
    test = rows["split"] == 1
    rate = oof_rate(rows, development)
    # Test predictions.
    selection = np.flatnonzero(test)
    records = [load_prediction(V7 / f"test_two_branch_seed{seed}", selection) for seed in SEEDS]
    test_rate = np.full_like(rows["target_rate"], np.nan, dtype=float)
    test_rate[selection] = pool(records, "two_branch")
    result = {"protocol": "old-event-v7-calibration-20261021",
              "note": ("Offsets are fitted on positive predictions only, on the calibration "
                       "folds; no outer/test rows enter the fit. This is a post-hoc calibrated "
                       "variant, not a retrained model."),
              "development": {}, "historical_test": {}}
    calibrated_oof = np.full_like(rate, np.nan)
    for fold in range(5):
        held = development & (rows["fold"] == fold)
        calibration = development & (rows["fold"] != fold)
        delta, objective_value = fit_delta(rows, calibration, rate, series_rate)
        calibrated_oof[held] = apply_offset(rate, delta)[held]
        result["development"][str(fold)] = dict(
            delta=delta.tolist(), calibration_objective=objective_value,
            raw_p1_72=cumulative_table(rows, held, rate, series_rate)["72"],
            calibrated_p1_72=cumulative_table(rows, held, apply_offset(rate, delta), series_rate)["72"])
    delta_all, objective_all = fit_delta(rows, development, rate, series_rate)
    calibrated_test = apply_offset(test_rate, delta_all)
    result["development"]["all"] = dict(delta=delta_all.tolist(),
                                        calibration_objective=objective_all,
                                        raw_p1_72=cumulative_table(rows, development, rate, series_rate)["72"],
                                        calibrated_p1_72=cumulative_table(rows, development,
                                                                          apply_offset(rate, delta_all),
                                                                          series_rate)["72"])
    result["historical_test"] = dict(
        delta=delta_all.tolist(), calibration_objective=objective_all,
        raw_p1_72=cumulative_table(rows, test, test_rate, series_rate)["72"],
        calibrated_p1_72=cumulative_table(rows, test, calibrated_test, series_rate)["72"])
    # Hourly / action summaries for raw vs calibrated on folds1-4 and test.
    mask_dev = development & (rows["fold"] > 0)
    mask_test = test
    for name, mask, raw, calibrated in (
        ("development_folds1to4", mask_dev, rate, calibrated_oof),
        ("historical_test", mask_test, test_rate, calibrated_test)):
        result[name] = dict(
            raw={key: value for key, value in summarize(
                "raw", rows["patient"][mask], rows["previous_rate"][mask],
                rows["target_rate"][mask], raw[mask], rows["action_mask"][mask]).items()
                if key in ("setting_mae", "change_hour_mae", "strict_action_f1",
                           "zero_flow_accuracy", "false_change_rate")},
            calibrated={key: value for key, value in summarize(
                "cal", rows["patient"][mask], rows["previous_rate"][mask],
                rows["target_rate"][mask], calibrated[mask], rows["action_mask"][mask]).items()
                if key in ("setting_mae", "change_hour_mae", "strict_action_f1",
                           "zero_flow_accuracy", "false_change_rate")},
            raw_p1={h: cumulative_table(rows, mask, raw, series_rate)[h] for h in ("24", "48", "72")},
            calibrated_p1={h: cumulative_table(rows, mask, calibrated, series_rate)[h]
                           for h in ("24", "48", "72")})
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "calibration.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    print(json.dumps({
        "delta_all": delta_all.tolist(),
        "dev_raw_p1": result["development_folds1to4"]["raw_p1"]["72"]["total_mae"],
        "dev_cal_p1": result["development_folds1to4"]["calibrated_p1"]["72"]["total_mae"],
        "test_raw_p1": result["historical_test"]["raw_p1"]["72"]["total_mae"],
        "test_cal_p1": result["historical_test"]["calibrated_p1"]["72"]["total_mae"],
        "dev_raw_mae": result["development_folds1to4"]["raw"]["setting_mae"],
        "dev_cal_mae": result["development_folds1to4"]["calibrated"]["setting_mae"],
        "test_raw_mae": result["historical_test"]["raw"]["setting_mae"],
        "test_cal_mae": result["historical_test"]["calibrated"]["setting_mae"],
    }, indent=2))


if __name__ == "__main__":
    main()
