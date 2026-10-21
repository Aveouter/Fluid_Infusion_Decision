"""Registered V7.3-CUM: patient-level 24/48/72 h cumulative loss.

Fold-0 selection showed cumulative_weight=0.2 lowered the 72 h total MAE from
3,608.9 (V7.1-M) to 2,081.6, below the original BurnResu 2,699.7, at a cost of
setting MAE 51.8 -> 60.3 and strict F1 0.384 -> 0.355.  This run repeats the
selected setting on all five development folds and the historical test.
"""
from __future__ import annotations

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

PROTOCOL = ROOT / "docs/old_event_v7_cum_protocol_20261021.json"
OUTPUT = ROOT / ".audit/old_event_v7_cum"
SEEDS = (20261015, 20261016, 20261017)
FOLDS = (0, 1, 2, 3, 4)
CONFIG = dict(DEFAULT_CONFIG, mode="two_branch", epochs=60, batch_size=512, hidden=64,
              dropout=0.1, learning_rate=0.002, weight_decay=0.01, gradient_clip=5.0,
              event_weight=0.5, direction_weight=0.5, magnitude_weight=0.2,
              active_weight=0.2, absolute_weight=0.5, balanced_event=False,
              use_delta=False, use_horizon=False, min_change=50.0,
              hold_mode="previous", event_negative_weight=1.0,
              cumulative_weight=0.2, cumulative_scale=1000.0)
KEYS = ("rate_ml_h", "event_probability", "action_probability", "direction_probability",
        "rate_candidates_ml_h", "active_probability", "absolute_rate_ml_h", "horizon_ml")
DATA_FILES = ("data/all_combined_data.csv", "data/baseline_processed.csv",
              "data/train_split.csv", "data/test_split.csv",
              "docs/label_protocol.md", "utils/eval_protocol.py")
CODE_FILES = ("utils/old_fluid_data_v1.py", "utils/old_event_model_v1.py",
              "utils/old_event_training_v1.py",
              "reports/train_old_event_cum_20261021.py")


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def register():
    payload = dict(protocol="old-event-v7-cum-20261021", history=24, seeds=list(SEEDS),
                   folds=list(FOLDS), arms={"two_branch": CONFIG},
                   data_sha256={p: sha(p) for p in DATA_FILES},
                   code_sha256={p: sha(p) for p in CODE_FILES},
                   output=str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
                   note=("V7.3-CUM selected on fold 0 only: whole-patient batches plus a "
                         "patient-level 24/48/72 h cumulative L1 term. Zeros, activity "
                         "gate, action head and decoder are unchanged. Historical test "
                         "is not used for selection."))
    if PROTOCOL.exists():
        current = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        assert current == payload, "Protocol drifted; write a new protocol instead of editing"
    else:
        PROTOCOL.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main():
    register()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, windows, meta = build(ROOT, history=24, group="joint")
    development = rows["split"] == 0
    test = rows["split"] == 1
    started = time.time()
    for fold in FOLDS:
        test_mask = development & (rows["fold"] == fold)
        train_mask = development & (rows["fold"] != fold)
        selection = np.flatnonzero(test_mask)
        for seed in SEEDS:
            folder = OUTPUT / f"fold{fold}_two_branch_seed{seed}"
            folder.mkdir(exist_ok=True)
            if (folder / "prediction_private.npz").exists():
                continue
            bundle, model = fit(windows, rows, train_mask, dict(CONFIG, seed=seed), seed,
                                progress=lambda record: print(json.dumps(
                                    dict(fold=fold, seed=seed, **record)), flush=True))
            raw = predict(bundle, windows, rows, selection=selection)
            np.savez_compressed(folder / "prediction_private.npz", row_index=selection,
                                **{key: raw[key] for key in KEYS})
            torch.save(bundle, folder / "model_private.pt")
            print(json.dumps(dict(completed=f"fold{fold}_seed{seed}",
                                  seconds=time.time() - started)), flush=True)
    selection = np.flatnonzero(test)
    for seed in SEEDS:
        folder = OUTPUT / f"test_two_branch_seed{seed}"
        folder.mkdir(exist_ok=True)
        if (folder / "prediction_private.npz").exists():
            continue
        bundle, model = fit(windows, rows, development, dict(CONFIG, seed=seed), seed,
                            progress=lambda record: print(json.dumps(
                                dict(test=seed, **record)), flush=True))
        raw = predict(bundle, windows, rows, selection=selection)
        np.savez_compressed(folder / "prediction_private.npz", row_index=selection,
                            **{key: raw[key] for key in KEYS})
        torch.save(bundle, folder / "model_private.pt")
        print(json.dumps(dict(completed=f"test_seed{seed}", seconds=time.time() - started)), flush=True)
    (OUTPUT / "complete.json").write_text(json.dumps(
        dict(status="fits_and_predictions_complete", seconds=time.time() - started), indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
