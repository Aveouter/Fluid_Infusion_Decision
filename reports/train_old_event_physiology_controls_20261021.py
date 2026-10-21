"""Matched physiology information-set controls for V7.1.

Three arms use exactly the same rows, folds, seeds and config as the joint
V7.1 run; only the physiology information set changes:

* ``history``: no vital values, no vital missing indicators;
* ``missing``: missing indicators only;
* ``values``: vital values only.

The oral and vital channels are blanked by feature position in
``utils.old_fluid_data_v1.build``; the valid marker and hour feature are kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from utils.old_fluid_data_v1 import build
from utils.old_event_training_v1 import DEFAULT_CONFIG, fit, predict

SEEDS = (20261015, 20261016, 20261017)
FOLDS = (0, 1, 2, 3, 4)
CONFIG = dict(DEFAULT_CONFIG, mode="two_branch", epochs=60, batch_size=512, hidden=64,
              dropout=0.1, learning_rate=0.002, weight_decay=0.01, gradient_clip=5.0,
              event_weight=0.5, direction_weight=0.5, magnitude_weight=0.2,
              active_weight=0.2, absolute_weight=0.5, balanced_event=False,
              use_delta=False, use_horizon=False, min_change=50.0,
              hold_mode="previous", event_negative_weight=1.0)
KEYS = ("rate_ml_h", "event_probability", "action_probability", "direction_probability",
        "rate_candidates_ml_h", "active_probability", "absolute_rate_ml_h", "horizon_ml")
DATA_FILES = ("data/all_combined_data.csv", "data/baseline_processed.csv",
              "data/train_split.csv", "data/test_split.csv",
              "docs/label_protocol.md", "utils/eval_protocol.py")
CODE_FILES = ("utils/old_fluid_data_v1.py", "utils/old_event_model_v1.py",
              "utils/old_event_training_v1.py",
              "reports/train_old_event_physiology_controls_20261021.py")
PATHS = {
    "history": (ROOT / "docs/old_event_v7_history_v2_protocol_20261021.json",
                ROOT / ".audit/old_event_v7_history_v2"),
    "missing": (ROOT / "docs/old_event_v7_missing_protocol_20261021.json",
                ROOT / ".audit/old_event_v7_missing"),
    "values": (ROOT / "docs/old_event_v7_values_protocol_20261021.json",
               ROOT / ".audit/old_event_v7_values"),
}
NOTES = {
    "history": ("Corrected matched control: oral values, vital values and their missing "
                "indicators are blanked; the valid marker and raw hour feature are kept. "
                "The earlier history arm blanked the valid/hour columns by mistake."),
    "missing": ("Missing indicators only: oral/vital values blanked, oral/vital missing "
                "indicators kept."),
    "values": ("Values only: oral/vital values kept, oral/vital missing indicators blanked."),
}


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def register(group):
    protocol, output = PATHS[group]
    payload = dict(protocol=f"old-event-v7-{group}-v2-20261021", history=24,
                   seeds=list(SEEDS), folds=list(FOLDS), group=group,
                   arms={"two_branch": CONFIG},
                   data_sha256={p: sha(p) for p in DATA_FILES},
                   code_sha256={p: sha(p) for p in CODE_FILES},
                   output=str(output.relative_to(ROOT)).replace("\\", "/"),
                   note=NOTES[group])
    if protocol.exists():
        current = json.loads(protocol.read_text(encoding="utf-8"))
        assert current == payload, "Protocol drifted; write a new protocol instead of editing"
    else:
        protocol.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=tuple(PATHS), required=True)
    args = parser.parse_args()
    group = args.group
    protocol, output = PATHS[group]
    register(group)
    output.mkdir(parents=True, exist_ok=True)
    rows, windows, meta = build(ROOT, history=24, group=group)
    development = rows["split"] == 0
    test = rows["split"] == 1
    started = time.time()
    for fold in FOLDS:
        test_mask = development & (rows["fold"] == fold)
        train_mask = development & (rows["fold"] != fold)
        selection = np.flatnonzero(test_mask)
        for seed in SEEDS:
            folder = output / f"fold{fold}_two_branch_seed{seed}"
            folder.mkdir(exist_ok=True)
            if (folder / "prediction_private.npz").exists():
                continue
            bundle, model = fit(windows, rows, train_mask, dict(CONFIG, seed=seed), seed,
                                progress=lambda record: print(json.dumps(
                                    dict(group=group, fold=fold, seed=seed, **record)), flush=True))
            raw = predict(bundle, windows, rows, selection=selection)
            np.savez_compressed(folder / "prediction_private.npz", row_index=selection,
                                **{key: raw[key] for key in KEYS})
            torch.save(bundle, folder / "model_private.pt")
            print(json.dumps(dict(group=group, completed=f"fold{fold}_seed{seed}",
                                  seconds=time.time() - started)), flush=True)
    selection = np.flatnonzero(test)
    for seed in SEEDS:
        folder = output / f"test_two_branch_seed{seed}"
        folder.mkdir(exist_ok=True)
        if (folder / "prediction_private.npz").exists():
            continue
        bundle, model = fit(windows, rows, development, dict(CONFIG, seed=seed), seed,
                            progress=lambda record: print(json.dumps(
                                dict(group=group, test=seed, **record)), flush=True))
        raw = predict(bundle, windows, rows, selection=selection)
        np.savez_compressed(folder / "prediction_private.npz", row_index=selection,
                            **{key: raw[key] for key in KEYS})
        torch.save(bundle, folder / "model_private.pt")
        print(json.dumps(dict(group=group, completed=f"test_seed{seed}",
                              seconds=time.time() - started)), flush=True)
    (output / "complete.json").write_text(json.dumps(
        dict(status="fits_and_predictions_complete", group=group,
             seconds=time.time() - started), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
