"""Seed-matched evaluation of V7.3-CUM (seed 20261015)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.evaluate_old_event_v7_20261021 import (decode_parts, load_prediction,
                                                     pool_parts, summarize, bootstrap)
from reports.evaluate_old_event_mean05_20261021 import paired_p1
from reports.issue81_style_v7_20261021 import evaluate
from utils.old_fluid_data_v1 import build

ROOTS = {"v71": ROOT / ".audit/old_event_v7_prev",
         "v71_mean": ROOT / ".audit/old_event_v7_mean05",
         "v7_cum": ROOT / ".audit/old_event_v7_cum"}
SEED = 20261015
REPORT = ROOT / "docs/experiments/old-event-v7-20261021"


def arrays(root, rows, *, test=False):
    rate = np.full_like(rows["target_rate"], np.nan, dtype=float)
    event = np.full_like(rows["target_rate"], np.nan, dtype=float)
    active = np.full_like(rows["target_rate"], np.nan, dtype=float)
    if test:
        selection = np.flatnonzero(rows["split"] == 1)
        records = [load_prediction(root / f"test_two_branch_seed{SEED}", selection)]
        parts = pool_parts(records, "two_branch")
        rate[selection] = decode_parts(parts, "two_branch")
        event[selection] = parts["event_probability"]
        active[selection] = parts["active_probability"]
    else:
        development = rows["split"] == 0
        for fold in range(5):
            selection = np.flatnonzero(development & (rows["fold"] == fold))
            records = [load_prediction(root / f"fold{fold}_two_branch_seed{SEED}", selection)]
            parts = pool_parts(records, "two_branch")
            rate[selection] = decode_parts(parts, "two_branch")
            event[selection] = parts["event_probability"]
            active[selection] = parts["active_probability"]
    return rate, event, active


def main():
    rows, windows, meta = build(ROOT)
    series_rate = {i: meta["series"][pid]["rate"] for i, pid in enumerate(meta["patients"])}
    development = rows["split"] == 0
    mask_dev = development & (rows["fold"] > 0)
    mask_test = rows["split"] == 1
    values = {name: arrays(root, rows) for name, root in ROOTS.items()}
    values_test = {name: arrays(root, rows, test=True) for name, root in ROOTS.items()}
    payload = dict(protocol="old-event-v7-cum-eval-seed20261015-20261021",
                   note=("Seed-matched evaluation: all methods use seed 20261015 only. "
                         "The full 3-seed V7.1/V7.1-M tables remain the main results; "
                         "this table isolates the V7.3 cumulative-loss change."),
                   seed=SEED, results={})
    for name, mask, data, data_test in (
        ("development_folds1to4", mask_dev, values, None),
        ("historical_test", mask_test, values_test, None)):
        payload["results"][name] = {}
        for method, item in data.items():
            payload["results"][name][method] = evaluate(
                rows, meta, item[0], item[1], item[2], mask, method)
            payload["results"][name][method + "_summary"] = {
                key: value for key, value in summarize(
                    method, rows["patient"][mask], rows["previous_rate"][mask],
                    rows["target_rate"][mask], item[0][mask], rows["action_mask"][mask]).items()
                if key in ("setting_mae", "change_hour_mae", "strict_action_f1",
                           "zero_flow_accuracy", "false_change_rate")}
        payload["results"][name]["paired_vs_v71"] = paired_p1(
            rows, mask, data["v7_cum"][0], data["v71"][0], series_rate)
        payload["results"][name]["paired_vs_mean"] = paired_p1(
            rows, mask, data["v7_cum"][0], data["v71_mean"][0], series_rate)
        payload["results"][name]["mae_vs_v71"] = bootstrap(
            summarize("cum", rows["patient"][mask], rows["previous_rate"][mask],
                      rows["target_rate"][mask], data["v7_cum"][0][mask], rows["action_mask"][mask]),
            summarize("v71", rows["patient"][mask], rows["previous_rate"][mask],
                      rows["target_rate"][mask], data["v71"][0][mask], rows["action_mask"][mask]))
    (REPORT / "cum_issue81_seed20261015.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({split: dict(
        v71_p1=payload["results"][split]["v71"]["cumulative"]["72"]["total_mae"],
        mean_p1=payload["results"][split]["v71_mean"]["cumulative"]["72"]["total_mae"],
        cum_p1=payload["results"][split]["v7_cum"]["cumulative"]["72"]["total_mae"],
        cum_p1_24=payload["results"][split]["v7_cum"]["cumulative"]["24"]["total_mae"],
        cum_mae=payload["results"][split]["v7_cum"]["secondary"]["all_hours_mae"],
        cum_f1=payload["results"][split]["v7_cum"]["activity"]["macro_f1"],
        cum_strict=payload["results"][split]["v7_cum_summary"]["strict_action_f1"],
        paired_vs_v71=payload["results"][split]["paired_vs_v71"],
        paired_vs_mean=payload["results"][split]["paired_vs_mean"],
    ) for split in ("development_folds1to4", "historical_test")}, indent=2))


if __name__ == "__main__":
    main()
