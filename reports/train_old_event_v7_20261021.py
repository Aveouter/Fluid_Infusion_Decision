"""Registered old-data event-model run (V7): five outer folds, three seeds.

The old 291/72 split is kept.  The 291 development patients are split into five
patient-disjoint folds; each fold's model never sees its held-out patients.  The
72 historical test patients are evaluated once with a model fitted on all 291.
No outer-fold quantity is used for early stopping, threshold choice or arm
selection.
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

PROTOCOL = ROOT / "docs/old_event_v7_protocol_20261021.json"
OUTPUT = ROOT / ".audit/old_event_v7"
SEEDS = (20261015, 20261016, 20261017)
FOLDS = (0, 1, 2, 3, 4)
ARMS = {
    "absolute": dict(mode="absolute"),
    "two_branch": dict(mode="two_branch", hold_mode="absolute",
                       balanced_event=False, event_weight=0.5, direction_weight=0.5,
                       event_negative_weight=1.0, use_delta=False, use_horizon=False),
}
CONFIG = dict(DEFAULT_CONFIG, epochs=60, batch_size=512, hidden=64, dropout=0.1,
              learning_rate=0.002, weight_decay=0.01, gradient_clip=5.0,
              event_weight=0.5, direction_weight=0.5, magnitude_weight=0.2,
              active_weight=0.2, absolute_weight=0.5, balanced_event=False,
              use_delta=False, use_horizon=False, min_change=50.0,
              hold_mode="absolute", event_negative_weight=1.0)
DATA_FILES = ("data/all_combined_data.csv", "data/baseline_processed.csv",
              "data/train_split.csv", "data/test_split.csv",
              "docs/label_protocol.md", "utils/eval_protocol.py")
CODE_FILES = ("utils/old_fluid_data_v1.py", "utils/old_event_model_v1.py",
              "utils/old_event_training_v1.py", "reports/train_old_event_v7_20261021.py")
KEYS = ("rate_ml_h", "event_probability", "action_probability", "direction_probability",
        "rate_candidates_ml_h", "active_probability", "absolute_rate_ml_h", "horizon_ml")
ABSOLUTE_KEYS = ("rate_ml_h", "event_probability", "active_probability", "horizon_ml")


def arm_keys(arm):
    return ABSOLUTE_KEYS if arm == "absolute" else KEYS


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def register():
    payload = dict(protocol="old-event-v7-20261021", history=24, seeds=list(SEEDS),
                   folds=list(FOLDS), arms=ARMS, configuration=CONFIG,
                   data_sha256={p: sha(p) for p in DATA_FILES},
                   code_sha256={p: sha(p) for p in CODE_FILES},
                   output=str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
                   note=("Old 363-patient cohort and old 291/72 split; no final_v1 labels. "
                         "Thresholds are native 0.5; no outer-fold selection."),
                   post_registration_note=("The main predictions were produced before hold_mode "
                                           "and the event negative weight were made explicit. "
                                           "A fold0/seed20261015 replay with the final code and "
                                           "config is bit-identical (max abs diff 0.0)."))
    if PROTOCOL.exists():
        current = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        assert current == payload, "Protocol drifted; write a new protocol instead of editing"
    else:
        PROTOCOL.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main():
    payload = register()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, windows, meta = build(ROOT, history=payload["history"])
    development = rows["split"] == 0
    test = rows["split"] == 1
    progress = {}
    started = time.time()
    for fold in FOLDS:
        test_mask = development & (rows["fold"] == fold)
        train_mask = development & (rows["fold"] != fold)
        assert not set(rows["patient"][train_mask]) & set(rows["patient"][test_mask])
        selection = np.flatnonzero(test_mask)
        for arm, options in ARMS.items():
            pooled = []
            for seed in SEEDS:
                folder = OUTPUT / f"fold{fold}_{arm}_seed{seed}"
                folder.mkdir(exist_ok=True)
                state_path = folder / "prediction_private.npz"
                if state_path.exists():
                    with np.load(state_path, allow_pickle=False) as saved:
                        pooled.append({key: saved[key] for key in saved.files if key != "row_index"})
                    continue
                config = dict(CONFIG, **options, seed=seed)
                bundle, model = fit(windows, rows, train_mask, config, seed,
                                    progress=lambda record: print(json.dumps(
                                        dict(fold=fold, arm=arm, seed=seed, **record)), flush=True))
                raw = predict(bundle, windows, rows, selection=selection)
                np.savez_compressed(state_path, row_index=selection,
                                    **{key: raw[key] for key in arm_keys(arm)})
                torch.save(bundle, folder / "model_private.pt")
                progress[f"fold{fold}_{arm}_seed{seed}"] = dict(
                    seconds=time.time() - started,
                    training_patients=int(len(np.unique(rows["patient"][train_mask]))),
                    test_patients=int(len(np.unique(rows["patient"][test_mask]))))
                (OUTPUT / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
                with np.load(state_path, allow_pickle=False) as saved:
                    pooled.append({key: saved[key] for key in saved.files if key != "row_index"})
            if len(pooled) == len(SEEDS):
                for key in arm_keys(arm):
                    assert np.isfinite(pooled[0][key]).all(), key
    # Historical test: fit on all development patients, evaluate once.
    selection = np.flatnonzero(test)
    for arm, options in ARMS.items():
        for seed in SEEDS:
            folder = OUTPUT / f"test_{arm}_seed{seed}"
            folder.mkdir(exist_ok=True)
            state_path = folder / "prediction_private.npz"
            if state_path.exists():
                continue
            config = dict(CONFIG, **options, seed=seed)
            bundle, model = fit(windows, rows, development, config, seed,
                                progress=lambda record: print(json.dumps(
                                    dict(test=arm, seed=seed, **record)), flush=True))
            raw = predict(bundle, windows, rows, selection=selection)
            np.savez_compressed(state_path, row_index=selection,
                                **{key: raw[key] for key in arm_keys(arm)})
            torch.save(bundle, folder / "model_private.pt")
            progress[f"test_{arm}_seed{seed}"] = dict(seconds=time.time() - started,
                                                      test_patients=int(len(np.unique(rows["patient"][test]))))
            (OUTPUT / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
    (OUTPUT / "complete.json").write_text(json.dumps(
        dict(status="fits_and_predictions_complete", seeds=list(SEEDS), folds=list(FOLDS),
             arms=list(ARMS), seconds=time.time() - started), indent=2), encoding="utf-8")
    print(json.dumps(dict(status="complete", seconds=time.time() - started)))


if __name__ == "__main__":
    main()
