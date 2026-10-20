"""Matched information-set contrast: V7 joint vs V7 history-only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.evaluate_old_event_v7_20261021 import (SEEDS, bootstrap, clean, load_prediction,
                                                     pool, summarize)
from utils.old_fluid_data_v1 import build

JOINT = ROOT / ".audit/old_event_v7_prev"
HISTORY = ROOT / ".audit/old_event_v7_history"
REPORT = ROOT / "docs/experiments/old-event-v7-20261021"


def oof(root, rows, development):
    rate = np.full_like(rows["target_rate"], np.nan, dtype=float)
    for fold in range(5):
        selection = np.flatnonzero(development & (rows["fold"] == fold))
        records = [load_prediction(root / f"fold{fold}_two_branch_seed{seed}", selection)
                   for seed in SEEDS]
        rate[selection] = pool(records, "two_branch")
    return rate


def test_rate(root, rows, test):
    selection = np.flatnonzero(test)
    records = [load_prediction(root / f"test_two_branch_seed{seed}", selection) for seed in SEEDS]
    full = np.full_like(rows["target_rate"], np.nan, dtype=float)
    full[selection] = pool(records, "two_branch")
    return full


def main():
    rows, windows, meta = build(ROOT)
    development = rows["split"] == 0
    test = rows["split"] == 1
    result = {}
    joint_oof = oof(JOINT, rows, development)
    history_oof = oof(HISTORY, rows, development)
    joint_test = test_rate(JOINT, rows, test)
    history_test = test_rate(HISTORY, rows, test)
    blocks = (
        ("development", development, joint_oof, history_oof),
        ("development_folds1to4", development & (rows["fold"] > 0), joint_oof, history_oof),
        ("historical_test", test, joint_test, history_test),
    )
    for split, mask, joint_full, history_full in blocks:
        joint, history = joint_full[mask], history_full[mask]
        summaries = {
            "joint": summarize("joint", rows["patient"][mask], rows["previous_rate"][mask],
                               rows["target_rate"][mask], joint, rows["action_mask"][mask]),
            "history_only": summarize("history_only", rows["patient"][mask],
                                      rows["previous_rate"][mask], rows["target_rate"][mask],
                                      history, rows["action_mask"][mask]),
        }
        result[split] = dict(
            rows=int(mask.sum()),
            patients=int(len(np.unique(rows["patient"][mask]))),
            summaries={name: {key: value for key, value in item.items()
                              if key not in ("patient_mae", "patient_counts",
                                             "patient_class_counts")}
                       for name, item in summaries.items()},
            paired=bootstrap(summaries["joint"], summaries["history_only"]))
    payload = dict(protocol="old-event-v7-history-20261021", seeds=list(SEEDS),
                   note=("Joint vs matched history-only control. Static baseline and fluid "
                         "history are identical; only oral/vital values and their missingness differ. "
                         "The prespecified physiology criterion was +0.02 strict action F1 with a "
                         "paired lower bound above zero."),
                   results=result)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "history_ablation.json").write_text(
        json.dumps(clean(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({split: {name: dict(setting_mae=item["setting_mae"],
                                         strict_action_f1=item["strict_action_f1"])
                              for name, item in block["summaries"].items()}
                      for split, block in result.items()}, indent=2))


if __name__ == "__main__":
    main()
