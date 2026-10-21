"""Fit V7 variants on the original model's 55-patient test cohort (308 training patients)."""
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

BUNDLE = ROOT / ".audit/eval_v2/bundle_original_epoch15.npz"
OUTPUT = ROOT / ".audit/old_event_original_cohort"
PROTOCOL = ROOT / "docs/old_event_original_cohort_protocol_20261021.json"
SEEDS = (20261015, 20261016, 20261017)
BASE = dict(DEFAULT_CONFIG, mode="two_branch", epochs=60, batch_size=512, hidden=64,
            dropout=0.1, learning_rate=0.002, weight_decay=0.01, gradient_clip=5.0,
            event_weight=0.5, direction_weight=0.5, magnitude_weight=0.2,
            active_weight=0.2, absolute_weight=0.5, balanced_event=False,
            use_delta=False, use_horizon=False, min_change=50.0)
VARIANTS = {
    "prev_base": dict(BASE, hold_mode="previous"),
    "hold_absolute": dict(BASE, hold_mode="absolute"),
}
KEYS = ("rate_ml_h", "event_probability", "action_probability", "direction_probability",
        "rate_candidates_ml_h", "active_probability", "absolute_rate_ml_h", "horizon_ml")


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def register():
    payload = dict(protocol="old-event-original-cohort-20261021", seeds=list(SEEDS),
                   variants=VARIANTS,
                   bundle=str(BUNDLE.relative_to(ROOT)).replace("\\", "/"),
                   bundle_sha256=sha(BUNDLE),
                   code_sha256={p: sha(p) for p in (
                       "utils/old_fluid_data_v1.py", "utils/old_event_model_v1.py",
                       "utils/old_event_training_v1.py",
                       "reports/train_old_event_original_cohort_20261021.py")},
                   note=("Train on the 308 old-cohort patients outside the original model's "
                         "55-patient test cohort; predict its 3,622 windows for a paired "
                         "comparison with the submitted checkpoint."))
    if PROTOCOL.exists():
        assert json.loads(PROTOCOL.read_text(encoding="utf-8")) == payload, "Protocol drifted"
    else:
        PROTOCOL.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    register()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with np.load(BUNDLE, allow_pickle=False) as bundle:
        cohort = set(str(x) for x in bundle["patient_id"])
    rows, windows, meta = build(ROOT)
    patient_id = np.array(meta["patients"])[rows["patient"]]
    in_cohort = np.isin(patient_id, list(cohort))
    train = ~in_cohort
    selection = np.flatnonzero(in_cohort)
    if len(selection) != 3622:
        raise ValueError(f"Expected 3,622 cohort windows, found {len(selection)}")
    if set(patient_id[selection]) != cohort:
        raise ValueError("Cohort patient set mismatch")
    started = time.time()
    for name, config in VARIANTS.items():
        for seed in SEEDS:
            folder = OUTPUT / f"{name}_seed{seed}"
            folder.mkdir(exist_ok=True)
            if (folder / "prediction_private.npz").exists():
                continue
            bundle_, model = fit(windows, rows, train, dict(config, seed=seed), seed)
            raw = predict(bundle_, windows, rows, selection=selection)
            np.savez_compressed(folder / "prediction_private.npz", row_index=selection,
                                **{key: raw[key] for key in KEYS})
            torch.save(bundle_, folder / "model_private.pt")
            print(json.dumps(dict(completed=f"{name}_seed{seed}",
                                  seconds=time.time() - started)), flush=True)
    (OUTPUT / "complete.json").write_text(json.dumps(
        dict(status="complete", seconds=time.time() - started), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
