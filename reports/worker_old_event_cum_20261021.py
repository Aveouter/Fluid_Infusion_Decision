"""Parallel worker for the registered V7.3-CUM config (fold or test)."""
from __future__ import annotations

import argparse
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

OUTPUT = ROOT / ".audit/old_event_v7_cum"
CONFIG = dict(DEFAULT_CONFIG, mode="two_branch", epochs=60, batch_size=512, hidden=64,
              dropout=0.1, learning_rate=0.002, weight_decay=0.01, gradient_clip=5.0,
              event_weight=0.5, direction_weight=0.5, magnitude_weight=0.2,
              active_weight=0.2, absolute_weight=0.5, balanced_event=False,
              use_delta=False, use_horizon=False, min_change=50.0,
              hold_mode="previous", event_negative_weight=1.0,
              cumulative_weight=0.2, cumulative_scale=1000.0)
KEYS = ("rate_ml_h", "event_probability", "action_probability", "direction_probability",
        "rate_candidates_ml_h", "active_probability", "absolute_rate_ml_h", "horizon_ml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True, choices=("0", "1", "2", "3", "4", "test"))
    parser.add_argument("--seeds", default="20261015,20261016,20261017")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    rows, windows, meta = build(ROOT, history=24, group="joint")
    development = rows["split"] == 0
    started = time.time()
    if args.fold == "test":
        selection = np.flatnonzero(rows["split"] == 1)
        train_mask = development
        prefix = "test"
    else:
        fold = int(args.fold)
        selection = np.flatnonzero(development & (rows["fold"] == fold))
        train_mask = development & (rows["fold"] != fold)
        prefix = f"fold{fold}"
    for seed in seeds:
        folder = OUTPUT / f"{prefix}_two_branch_seed{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "prediction_private.npz").exists():
            continue
        bundle, _ = fit(windows, rows, train_mask, dict(CONFIG, seed=seed), seed,
                        progress=lambda record: print(json.dumps(
                            dict(prefix=prefix, seed=seed, **record)), flush=True))
        raw = predict(bundle, windows, rows, selection=selection)
        np.savez_compressed(folder / "prediction_private.npz", row_index=selection,
                            **{key: raw[key] for key in KEYS})
        torch.save(bundle, folder / "model_private.pt")
        print(json.dumps(dict(completed=f"{prefix}_seed{seed}",
                              seconds=time.time() - started)), flush=True)


if __name__ == "__main__":
    main()
