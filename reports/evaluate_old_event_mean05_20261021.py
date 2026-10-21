"""Issue81-style evaluation of the mean-matching variant (mean_weight 0.5)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.evaluate_old_event_v7_20261021 import bootstrap, load_prediction, pool, summarize
from reports.issue81_style_v7_20261021 import cumulative_entries, evaluate
from utils.old_fluid_data_v1 import build

BASE = ROOT / ".audit/old_event_v7_prev"
MEAN = ROOT / ".audit/old_event_v7_mean05"
SEEDS = (20261015, 20261016, 20261017)
REPORT = ROOT / "docs/experiments/old-event-v7-20261021"


def oof(root, rows, development):
    rate = np.full_like(rows["target_rate"], np.nan, dtype=float)
    event = np.full_like(rows["target_rate"], np.nan, dtype=float)
    active = np.full_like(rows["target_rate"], np.nan, dtype=float)
    for fold in range(5):
        selection = np.flatnonzero(development & (rows["fold"] == fold))
        records = [load_prediction(root / f"fold{fold}_two_branch_seed{seed}", selection)
                   for seed in SEEDS]
        from reports.evaluate_old_event_v7_20261021 import decode_parts, pool_parts
        parts = pool_parts(records, "two_branch")
        rate[selection] = decode_parts(parts, "two_branch")
        event[selection] = parts["event_probability"]
        active[selection] = parts["active_probability"]
    return rate, event, active


def test(root, rows):
    test_mask = rows["split"] == 1
    selection = np.flatnonzero(test_mask)
    records = [load_prediction(root / f"test_two_branch_seed{seed}", selection) for seed in SEEDS]
    from reports.evaluate_old_event_v7_20261021 import decode_parts, pool_parts
    parts = pool_parts(records, "two_branch")
    rate = np.full_like(rows["target_rate"], np.nan, dtype=float)
    event = np.full_like(rows["target_rate"], np.nan, dtype=float)
    active = np.full_like(rows["target_rate"], np.nan, dtype=float)
    rate[selection] = decode_parts(parts, "two_branch")
    event[selection] = parts["event_probability"]
    active[selection] = parts["active_probability"]
    return rate, event, active


def p1_errors(rows, mask, rate, series_rate):
    entries = cumulative_entries(rows, mask, rate, series_rate, 72)
    true = np.asarray([e[1] for e in entries]); pred = np.asarray([e[2] for e in entries])
    return np.abs(pred.sum(1) - true.sum(1))


def paired_p1(rows, mask, first, second, series_rate, *, replicates=2000, seed=20261021):
    a, b = p1_errors(rows, mask, first, series_rate), p1_errors(rows, mask, second, series_rate)
    rng = np.random.default_rng(seed); draws = rng.integers(len(a), size=(replicates, len(a)))
    diff = a[draws].mean(1) - b[draws].mean(1)
    return dict(difference=float(a.mean() - b.mean()),
                ci_low=float(np.percentile(diff, 2.5)), ci_high=float(np.percentile(diff, 97.5)))


def main():
    rows, windows, meta = build(ROOT)
    series_rate = {i: meta["series"][pid]["rate"] for i, pid in enumerate(meta["patients"])}
    development = rows["split"] == 0
    mask_dev = development & (rows["fold"] > 0)
    mask_test = rows["split"] == 1
    base_dev = oof(BASE, rows, development)
    mean_dev = oof(MEAN, rows, development)
    base_test = test(BASE, rows)
    mean_test = test(MEAN, rows)
    payload = dict(protocol="old-event-v7-mean05-20261021",
                   note=("mean_weight=0.5 total-residual mean-matching loss; same architecture, "
                         "hold=previous, plain event BCE, native 0.5 thresholds. Development folds "
                         "1-4 also selected the variant; historical test is independent."),
                   results={})
    for name, mask, base, mean in (
        ("development_folds1to4", mask_dev, base_dev, mean_dev),
        ("historical_test", mask_test, base_test, mean_test)):
        v71_summary = summarize("v71", rows["patient"][mask], rows["previous_rate"][mask],
                                rows["target_rate"][mask], base[0][mask], rows["action_mask"][mask])
        mean_summary = summarize("mean05", rows["patient"][mask], rows["previous_rate"][mask],
                                 rows["target_rate"][mask], mean[0][mask], rows["action_mask"][mask])
        payload["results"][name] = dict(
            v71=evaluate(rows, meta, base[0], base[1], base[2], mask, "v71"),
            mean05=evaluate(rows, meta, mean[0], mean[1], mean[2], mask, "mean05"),
            v71_summary={key: value for key, value in v71_summary.items()
                         if key in ("setting_mae", "change_hour_mae", "strict_action_f1",
                                    "zero_flow_accuracy", "false_change_rate")},
            mean_summary={key: value for key, value in mean_summary.items()
                          if key in ("setting_mae", "change_hour_mae", "strict_action_f1",
                                     "zero_flow_accuracy", "false_change_rate")},
            paired_p1=paired_p1(rows, mask, base[0], mean[0], series_rate),
            paired_mae=bootstrap(summarize("v71", rows["patient"][mask], rows["previous_rate"][mask],
                                           rows["target_rate"][mask], base[0][mask],
                                           rows["action_mask"][mask]),
                                 summarize("mean05", rows["patient"][mask], rows["previous_rate"][mask],
                                           rows["target_rate"][mask], mean[0][mask],
                                           rows["action_mask"][mask]))["setting_mae"])
    (REPORT / "mean05_issue81.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    print(json.dumps({split: dict(
        v71_p1=payload["results"][split]["v71"]["cumulative"]["72"]["total_mae"],
        mean_p1=payload["results"][split]["mean05"]["cumulative"]["72"]["total_mae"],
        v71_mae=payload["results"][split]["v71"]["secondary"]["all_hours_mae"],
        mean_mae=payload["results"][split]["mean05"]["secondary"]["all_hours_mae"],
        v71_f1=payload["results"][split]["v71"]["activity"]["macro_f1"],
        mean_f1=payload["results"][split]["mean05"]["activity"]["macro_f1"],
        paired_p1=payload["results"][split]["paired_p1"],
    ) for split in ("development_folds1to4", "historical_test")}, indent=2))


if __name__ == "__main__":
    main()
