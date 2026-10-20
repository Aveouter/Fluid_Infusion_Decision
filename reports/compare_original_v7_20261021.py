"""Compare V7 with the original BurnResu checkpoint on the same 55-patient cohort.

The original predictions come from PR #101 (`bundle_original_epoch15.npz`), which
re-scored the submitted epoch-15 checkpoint.  V7 is fitted on the other 308 old-
cohort patients and predicts the same 3,622 windows, so the comparison is paired.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.evaluate_old_event_v7_20261021 import (SEEDS, clean, load_prediction, pool,
                                                     summarize)
from utils.old_fluid_data_v1 import build

BUNDLE = ROOT / ".audit/eval_v2/bundle_original_epoch15.npz"
PREDICTIONS = ROOT / ".audit/old_event_original_cohort"
REPORT = ROOT / "docs/experiments/old-event-v7-20261021"


def previous_rates(patient, step, target, observed):
    previous = np.full_like(target, np.nan)
    for pid in np.unique(patient):
        index = np.flatnonzero(patient == pid)
        index = index[np.argsort(step[index])]
        previous[index[0]] = observed[index[0]]
        previous[index[1:]] = target[index[:-1]]
    return previous


def main():
    rows, windows, meta = build(ROOT)
    patient_id = np.array(meta["patients"])[rows["patient"]]
    with np.load(BUNDLE, allow_pickle=False) as bundle:
        bundle = {key: bundle[key] for key in bundle.files}
    cohort = set(str(x) for x in bundle["patient_id"])
    in_cohort = np.isin(patient_id, list(cohort))
    selection = np.flatnonzero(in_cohort)
    position = {int(row): i for i, row in enumerate(selection)}
    lookup = {(patient_id[row], int(rows["hour"][row])): int(row) for row in selection}
    # make_eval_bundle stores the decision origin t (1..T-1); the target row is t+1.
    keys = [(str(pid), int(hour) + 1) for pid, hour in zip(bundle["patient_id"], bundle["step_hour"])]
    missing = [key for key in keys if key not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} bundle windows are not in the cohort rows")
    order = np.array([position[lookup[key]] for key in keys])
    np.testing.assert_allclose(rows["target_rate"][selection][order], bundle["target_rates"], atol=1e-6)
    patient_ids = np.unique(bundle["patient_id"])
    patient = np.array([np.flatnonzero(patient_ids == pid)[0] for pid in bundle["patient_id"]])
    step = bundle["step_hour"].astype(int)
    target = bundle["target_rates"].astype(float)
    previous = previous_rates(patient, step, target, bundle["observed_rates"].astype(float))
    mask = np.ones_like(target, dtype=bool)
    methods = {"original_BurnResu_rate": bundle["rate"].astype(float),
               "persistence": bundle["pers_rates"].astype(float),
               "moving_average_k3": bundle["ma_rates"].astype(float)}
    for variant in ("prev_base", "hold_absolute"):
        records = [load_prediction(PREDICTIONS / f"{variant}_seed{seed}", selection)
                   for seed in SEEDS]
        pooled = pool(records, "two_branch")
        methods[f"V7_{variant}"] = pooled[order]
    summary = {name: summarize(name, patient, previous, target, value, mask)
               for name, value in methods.items()}
    payload = dict(source=str(BUNDLE.relative_to(ROOT)).replace("\\", "/"),
                   predictions=str(PREDICTIONS.relative_to(ROOT)).replace("\\", "/"),
                   windows=int(len(target)), patients=int(len(patient_ids)),
                   step_hour=[int(step.min()), int(step.max())],
                   note=("Paired on the original model's 55-patient test cohort. V7 is fitted "
                         "on the other 308 old-cohort patients. The original model's rate is "
                         "the action-decoded rate, not its speed head."),
                   summaries={name: {key: value for key, value in item.items()
                                     if key not in ("patient_mae", "patient_counts",
                                                    "patient_class_counts")}
                              for name, item in summary.items()})
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "original_comparison.json").write_text(
        json.dumps(clean(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({name: dict(setting_mae=item["setting_mae"],
                                 strict_action_f1=item["strict_action_f1"],
                                 zero_flow_accuracy=item["zero_flow_accuracy"],
                                 change_hour_mae=item["change_hour_mae"])
                      for name, item in payload["summaries"].items()}, indent=2))


if __name__ == "__main__":
    main()
