"""Tests for the old-data causal features and the event/action decoder."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from utils.old_event_model_v1 import OldTwoBranchNet
from utils.old_fluid_data_v1 import _sequence_features, _time_since_change


def test_time_since_change_uses_only_past_and_present():
    rate = np.array([[0, 0, 0], [100, 0, 0], [100, 0, 0], [200, 0, 0]], float)
    age = _time_since_change(rate)
    assert age[0, 0] == 0
    assert age[1, 0] == 0
    assert age[2, 0] == 1
    assert age[3, 0] == 0


def test_sequence_features_do_not_read_future_rows():
    rate = np.array([[0, 0, 0], [100, 0, 0], [100, 0, 50], [150, 0, 50]], float)
    added = np.zeros_like(rate)
    signed = np.zeros_like(rate)
    vitals = np.array([[36.5, 100, np.nan, 20, 120, 70, np.nan, 98, 30, 0],
                       [36.6, 105, np.nan, 21, 118, 68, np.nan, 97, 35, 0],
                       [36.7, 110, np.nan, 22, 115, 65, np.nan, 96, 40, 0],
                       [36.8, 115, np.nan, 23, 112, 62, np.nan, 95, 45, 0]], float)
    first = _sequence_features(rate, added, signed, vitals)
    changed = rate.copy()
    changed[3:] += 1000.0
    second = _sequence_features(changed, added, signed, vitals)
    np.testing.assert_allclose(first[:3], second[:3])


class _RateBranch(nn.Module):
    def __init__(self, rate, active_logit):
        super().__init__()
        self.rate = float(rate)
        self.active_logit = float(active_logit)

    def forward(self, sequence, static, previous_rate):
        shape = previous_rate.shape
        rate = torch.full(shape, self.rate, dtype=previous_rate.dtype)
        active = torch.full(shape, self.active_logit, dtype=previous_rate.dtype)
        horizon = torch.ones((shape[0], 3, 3), dtype=previous_rate.dtype)
        return dict(rate_ml_h=rate, active_logit=active, horizon_ml=horizon)


class _ActionBranch(nn.Module):
    def __init__(self, event, direction):
        super().__init__()
        self.event = float(event)
        self.direction = int(direction)

    def forward(self, sequence, static, previous_rate):
        shape = previous_rate.shape
        event = torch.full(shape, self.event, dtype=previous_rate.dtype)
        probability = torch.zeros((shape[0], 3, 5), dtype=previous_rate.dtype)
        probability[..., 0] = 1 - self.event
        probability[..., self.direction + 1] = self.event
        direction = torch.zeros((shape[0], 3, 4), dtype=previous_rate.dtype)
        direction[..., self.direction] = 1.0
        candidates = torch.zeros((shape[0], 3, 5), dtype=previous_rate.dtype)
        candidates[..., 0] = previous_rate
        candidates[..., 1] = 1.0
        return dict(event_probability=event, action_probability=probability,
                    direction_probability=direction,
                    rate_candidates_ml_h=candidates,
                    event_logit=torch.logit(event.clamp(1e-6, 1 - 1e-6)),
                    delta_ml_h=torch.zeros_like(previous_rate),
                    horizon_ml=torch.ones((shape[0], 3, 3), dtype=previous_rate.dtype))


def _decoded(event, direction, absolute=180.0, previous=200.0, active_logit=10.0,
             hold_mode="previous"):
    model = OldTwoBranchNet(3, 4, hidden=8, hold_mode=hold_mode)
    model.rate_branch = _RateBranch(absolute, active_logit)
    model.action_branch = _ActionBranch(event, direction)
    model.eval()
    previous_rate = torch.full((2, 3), previous)
    output = model(torch.zeros((2, 4, 3)), torch.zeros((2, 4)),
                   previous_rate, event_threshold=0.5, activity_threshold=0.5)
    return output["rate_ml_h"].detach().numpy()


def test_stop_decodes_to_exact_zero():
    rates = _decoded(event=1.0, direction=1)  # start=0, stop=1, up=2, down=3
    assert np.all(rates == 0.0)


def test_up_uses_absolute_magnitude_but_crosses_threshold():
    rates = _decoded(event=1.0, direction=2)  # up
    assert np.all(rates >= 250.0)             # previous 200 + 50


def test_down_crosses_threshold():
    rates = _decoded(event=1.0, direction=3, absolute=190.0)  # down
    assert np.all(rates <= 150.0)             # previous 200 - 50


def test_activity_gate_zeroes_hold_output():
    rates = _decoded(event=0.0, direction=0, active_logit=-10.0)
    assert np.all(rates == 0.0)


def test_hold_copies_previous_by_default():
    rates = _decoded(event=0.0, direction=0, absolute=180.0, previous=200.0)
    assert np.allclose(rates, 200.0)


def test_hold_uses_absolute_rate_in_absolute_hold_mode():
    rates = _decoded(event=0.0, direction=0, absolute=180.0, previous=200.0,
                     hold_mode="absolute")
    assert np.allclose(rates, 180.0)
