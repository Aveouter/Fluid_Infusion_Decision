"""Evaluate the hold-previous variants and compare them with V7 hold-absolute."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.evaluate_old_event_v7_20261021 import (SEEDS, bootstrap, changepoint_summary, clean,
                                                     cumulative_summary, cumulative_with_observed,
                                                     load_prediction, pool, summarize)
from utils.old_fluid_data_v1 import build

MAIN = ROOT / ".audit/old_event_v7"
VARIANTS = {
    "prev_base": ROOT / ".audit/old_event_v7_prev",
    "prev_neg15": ROOT / ".audit/old_event_v7_prev_neg15",
}
REPORT = ROOT / "docs/experiments/old-event-v7-20261021"


def oof_rate(root, rows, development, folds=range(5), arm="two_branch"):
    rate = np.full_like(rows["target_rate"], np.nan, dtype=float)
    for fold in folds:
        selection = np.flatnonzero(development & (rows["fold"] == fold))
        records = [load_prediction(root / f"fold{fold}_{arm}_seed{seed}", selection)
                   for seed in SEEDS]
        rate[selection] = pool(records, arm)
    return rate


def test_rate(root, rows, test, arm="two_branch"):
    selection = np.flatnonzero(test)
    records = [load_prediction(root / f"test_{arm}_seed{seed}", selection) for seed in SEEDS]
    full = np.full_like(rows["target_rate"], np.nan, dtype=float)
    full[selection] = pool(records, arm)
    return full


def block(rows, mask, methods, series_rate):
    summaries = {name: summarize(name, rows["patient"][mask], rows["previous_rate"][mask],
                                 rows["target_rate"][mask], rate[mask],
                                 rows["action_mask"][mask])
                 for name, rate in methods.items()}
    paired = dict(
        variant_vs_persistence=bootstrap(summaries["variant"], summaries["persistence"]),
        variant_vs_absolute=bootstrap(summaries["variant"], summaries["absolute"])
        if "absolute" in summaries else None,
        variant_vs_zero=bootstrap(summaries["variant"], summaries["zero"])
        if "zero" in summaries else None,
        absolute_vs_persistence=bootstrap(summaries["absolute"], summaries["persistence"])
        if "absolute" in summaries else None)
    cumulative = {name: {horizon: cumulative_summary(value)
                         for horizon, value in cumulative_with_observed(
                             rows["patient"][mask], rows["hour"][mask],
                             rows["target_rate"][mask], methods[name][mask], series_rate).items()}
                  for name in methods}
    changepoints = {name: changepoint_summary(
        rows["patient"][mask], rows["hour"][mask], rows["target_rate"][mask],
        methods[name][mask], series_rate) for name in methods}
    return dict(rows=int(mask.sum()), patients=int(len(np.unique(rows["patient"][mask]))),
                summaries={name: {key: value for key, value in item.items()
                                  if key not in ("patient_mae", "patient_counts",
                                                 "patient_class_counts")}
                           for name, item in summaries.items()},
                paired=paired, cumulative=cumulative, changepoints=changepoints)


def main():
    rows, windows, meta = build(ROOT)
    patients = meta["patients"]
    series_rate = {i: meta["series"][pid]["rate"] for i, pid in enumerate(patients)}
    development = rows["split"] == 0
    test = rows["split"] == 1
    payload = dict(protocol="old-event-v7-prev-variants-20261021", seeds=list(SEEDS),
                   note=("Hold candidate is the observed previous setting. prev_base uses the "
                         "plain event BCE; prev_neg15 uses a 1.5 negative-event weight. "
                         "Folds 1-4 are the untouched part after fold 0 selected the weight."),
                   variants={}, results={})
    main_oof = {"absolute": oof_rate(MAIN, rows, development, arm="absolute")}
    for name, root in VARIANTS.items():
        variant_oof = oof_rate(root, rows, development)
        methods_all = {"variant": variant_oof, "persistence": rows["previous_rate"].astype(float),
                       "absolute": main_oof["absolute"],
                       "zero": np.zeros_like(rows["target_rate"], dtype=float)}
        methods_14 = {key: value.copy() for key, value in methods_all.items()}
        mask_all = development
        mask_14 = development & (rows["fold"] > 0)
        payload["results"][f"{name}_all"] = block(rows, mask_all, methods_all, series_rate)
        payload["results"][f"{name}_folds1to4"] = block(rows, mask_14, methods_14, series_rate)
        variant_test = test_rate(root, rows, test)
        methods_test = {"variant": variant_test, "persistence": rows["previous_rate"].astype(float),
                        "absolute": test_rate(MAIN, rows, test, arm="absolute"),
                        "zero": np.zeros_like(rows["target_rate"], dtype=float)}
        payload["results"][f"{name}_test"] = block(rows, test, methods_test, series_rate)
    # Add the hold-absolute primary for reference.
    primary_oof = oof_rate(MAIN, rows, development)
    payload["results"]["hold_absolute_folds1to4"] = block(
        rows, development & (rows["fold"] > 0),
        {"variant": primary_oof, "persistence": rows["previous_rate"].astype(float),
         "absolute": main_oof["absolute"], "zero": np.zeros_like(rows["target_rate"], dtype=float)},
        series_rate)
    payload["results"]["hold_absolute_test"] = block(
        rows, test, {"variant": test_rate(MAIN, rows, test),
                     "persistence": rows["previous_rate"].astype(float),
                     "absolute": test_rate(MAIN, rows, test, arm="absolute"),
                     "zero": np.zeros_like(rows["target_rate"], dtype=float)}, series_rate)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "prev_variants.json").write_text(
        json.dumps(clean(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    for key, value in payload["results"].items():
        summary = value["summaries"]["variant"]
        print(key, {k: round(summary[k], 4) if isinstance(summary[k], float) else summary[k]
                    for k in ("setting_mae", "change_hour_mae", "strict_action_f1",
                              "zero_flow_accuracy", "false_change_rate")})


if __name__ == "__main__":
    main()
