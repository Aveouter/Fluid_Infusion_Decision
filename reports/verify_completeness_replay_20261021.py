"""Replay saved V7.1/control checkpoints on the hardened raw-time loader."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.old_fluid_data_v1 import build
from utils.old_event_training_v1 import predict

ARMS = {
    "joint": (ROOT / ".audit/old_event_v7_prev", "joint"),
    "history": (ROOT / ".audit/old_event_v7_history", "history"),
    "missing": (ROOT / ".audit/old_event_v7_missing", "missing"),
    "values": (ROOT / ".audit/old_event_v7_values", "values"),
}
KEYS = ("rate_ml_h", "event_probability", "active_probability", "rate_candidates_ml_h")
FOLD = 0
SEED = 20261015


def main():
    result = {}
    for name, (root, group) in ARMS.items():
        folder = root / f"fold{FOLD}_two_branch_seed{SEED}"
        if not (folder / "prediction_private.npz").exists():
            result[name] = dict(available=False)
            continue
        rows, windows, _ = build(ROOT, group=group)
        selection = np.flatnonzero((rows["split"] == 0) & (rows["fold"] == FOLD))
        bundle = torch.load(folder / "model_private.pt", map_location="cpu", weights_only=False)
        raw = predict(bundle, windows, rows, selection=selection)
        saved = np.load(folder / "prediction_private.npz")
        if not np.array_equal(saved["row_index"], selection):
            raise ValueError(f"row index mismatch in {folder}")
        result[name] = dict(
            available=True,
            group=group,
            max_abs_diff={key: float(np.max(np.abs(raw[key] - saved[key]))) for key in KEYS},
        )
    out = ROOT / "docs/experiments/old-event-v7-20261021/replay_verification.json"
    out.write_text(json.dumps(dict(note=("Checkpoint replay after the raw-time loader change; "
                                        "zero means the joint/history training features are unchanged."),
                                   fold=FOLD, seed=SEED, arms=result), indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
