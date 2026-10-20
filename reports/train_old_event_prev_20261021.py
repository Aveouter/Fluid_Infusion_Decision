"""V7 hold-previous variant: same two-branch model with hold = previous setting."""
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

OUTPUT = ROOT / ".audit/old_event_v7_prev"
PROTOCOL = ROOT / "docs/old_event_v7_prev_protocol_20261021.json"
SEEDS = (20261015, 20261016, 20261017)
FOLDS = (0, 1, 2, 3, 4)
CONFIG = dict(DEFAULT_CONFIG, mode="two_branch", hold_mode="previous", epochs=60,
              batch_size=512, hidden=64, dropout=0.1, learning_rate=0.002,
              weight_decay=0.01, gradient_clip=5.0, event_weight=0.5,
              direction_weight=0.5, magnitude_weight=0.2, active_weight=0.2,
              absolute_weight=0.5, balanced_event=False, use_delta=False,
              use_horizon=False, min_change=50.0)
KEYS = ("rate_ml_h", "event_probability", "action_probability", "direction_probability",
        "rate_candidates_ml_h", "active_probability", "absolute_rate_ml_h", "horizon_ml")
DATA_FILES = ("data/all_combined_data.csv", "data/baseline_processed.csv",
              "data/train_split.csv", "data/test_split.csv",
              "docs/label_protocol.md", "utils/eval_protocol.py")
CODE_FILES = ("utils/old_fluid_data_v1.py", "utils/old_event_model_v1.py",
              "utils/old_event_training_v1.py", "reports/train_old_event_prev_20261021.py")


def register():
    payload = dict(protocol="old-event-v7-prev-20261021", history=24, seeds=list(SEEDS),
                   folds=list(FOLDS), configuration=CONFIG,
                   data_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                for p in DATA_FILES},
                   code_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                for p in CODE_FILES},
                   output=str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
                   note=("Hold candidate is the observed previous setting, not the absolute "
                         "rate head. Everything else matches the registered V7 two-branch arm."))
    if PROTOCOL.exists():
        assert json.loads(PROTOCOL.read_text(encoding="utf-8")) == payload, "Protocol drifted"
    else:
        PROTOCOL.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    register()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, windows, meta = build(ROOT, history=24, group="joint")
    development = rows["split"] == 0
    test = rows["split"] == 1
    started = time.time()
    for fold in FOLDS:
        train = development & (rows["fold"] != fold)
        selection = np.flatnonzero(development & (rows["fold"] == fold))
        for seed in SEEDS:
            folder = OUTPUT / f"fold{fold}_two_branch_seed{seed}"
            folder.mkdir(exist_ok=True)
            if (folder / "prediction_private.npz").exists():
                continue
            bundle, model = fit(windows, rows, train, dict(CONFIG, seed=seed), seed)
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
        bundle, model = fit(windows, rows, development, dict(CONFIG, seed=seed), seed)
        raw = predict(bundle, windows, rows, selection=selection)
        np.savez_compressed(folder / "prediction_private.npz", row_index=selection,
                            **{key: raw[key] for key in KEYS})
        torch.save(bundle, folder / "model_private.pt")
        print(json.dumps(dict(completed=f"test_seed{seed}",
                              seconds=time.time() - started)), flush=True)
    (OUTPUT / "complete.json").write_text(json.dumps(
        dict(status="complete", seconds=time.time() - started), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
