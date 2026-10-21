"""Smoke test for the patient-level cumulative training loss."""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_old_data_completeness import _write_cohort
from utils.old_event_training_v1 import DEFAULT_CONFIG, fit, predict
from utils.old_fluid_data_v1 import build


@pytest.fixture()
def cohort(tmp_path):
    _write_cohort(tmp_path, {1: list(range(1, 36)) + list(range(37, 73)),
                             2: list(range(1, 49))})
    return tmp_path


def test_cumulative_loss_runs_and_predicts(cohort):
    rows, windows, meta = build(cohort, expected_patients=2)
    config = dict(DEFAULT_CONFIG, mode="two_branch", epochs=2, batch_size=32,
                  hidden=8, cumulative_weight=0.2)
    bundle, _ = fit(windows, rows, rows["split"] == 0, config, 1)
    raw = predict(bundle, windows, rows, selection=np.arange(3))
    assert raw["rate_ml_h"].shape == (3, 3)
    assert np.isfinite(raw["rate_ml_h"]).all()
