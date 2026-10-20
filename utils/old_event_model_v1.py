"""Event-gated, action-residual model for the original 363-patient task.

The old model's output was ``rate = prob * speed`` with a softplus speed, so it
could not emit an exact zero and therefore could not stop.  This model keeps the
same causal inputs but makes the decision explicit:

* an event head decides whether the recorded setting changes;
* a conditional direction head chooses start / stop / up / down;
* the final rate is a *candidate* built from the observed previous setting:
  hold -> previous, start -> learned positive, stop -> 0,
  up -> previous + learned step, down -> previous - learned step.

Every candidate is a physical quantity, so the decoded action and the evaluated
action agree by construction.  ``absolute`` mode is the control: the same GRU
with a direct softplus rate head and no event or action supervision.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MIN_CHANGE_ML_H = 50.0


class OldEventNet(nn.Module):
    def __init__(self, sequence_channels, static_channels, *, hidden=64, dropout=0.1,
                 mode="event", horizons=3, min_change=MIN_CHANGE_ML_H, detach_actions=False):
        super().__init__()
        if mode not in ("event", "absolute", "hybrid", "guarded"):
            raise ValueError("Unknown model mode")
        self.mode = mode
        self.detach_actions = bool(detach_actions)
        self.horizons = int(horizons)
        self.min_change = float(min_change)
        self.encoder = nn.GRU(sequence_channels, hidden, batch_first=True)
        self.context = nn.Sequential(nn.Linear(hidden + static_channels, hidden),
                                     nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout))
        if mode == "absolute":
            self.rate_head = nn.Linear(hidden, 3)
            self.active_head = nn.Linear(hidden, 3)
            self.horizon_head = nn.Linear(hidden, 3 * self.horizons)
            return
        self.event_head = nn.Linear(hidden, 3)
        self.direction_head = nn.Linear(hidden, 3 * 4)
        self.step_head = nn.Linear(hidden, 3 * 2)
        self.start_head = nn.Linear(hidden, 3)
        self.delta_head = nn.Linear(hidden, 3)
        self.horizon_head = nn.Linear(hidden, 3 * self.horizons)
        if mode in ("hybrid", "guarded"):
            self.absolute_head = nn.Linear(hidden, 3)
        nn.init.zeros_(self.event_head.weight)
        nn.init.zeros_(self.event_head.bias)

    def forward(self, sequence, static, previous_rate, *, event_threshold=0.5, hard=True):
        _, history = self.encoder(sequence)
        shared = self.context(torch.cat([history[-1], static], -1))
        base = torch.nan_to_num(previous_rate, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        if self.mode == "absolute":
            rate = F.softplus(self.rate_head(shared)) * 100.0
            active_probability = torch.sigmoid(self.active_head(shared))
            return dict(rate_ml_h=rate, active_logit=self.active_head(shared),
                        active_probability=active_probability,
                        event_probability=active_probability,
                        horizon_ml=F.softplus(self.horizon_head(shared).reshape(-1, 3, self.horizons)) * 1000.0)
        event_logit = self.event_head(shared.detach() if (self.detach_actions and self.training) else shared)
        event_probability = torch.sigmoid(event_logit)
        action_shared = shared.detach() if (self.detach_actions and self.training) else shared
        direction = F.softmax(self.direction_head(action_shared).reshape(-1, 3, 4), -1)
        probabilities = torch.cat([(1 - event_probability).unsqueeze(-1),
                                   event_probability.unsqueeze(-1) * direction], -1)
        steps = F.softplus(self.step_head(action_shared).reshape(-1, 3, 2)) * 100.0 + self.min_change
        start = F.softplus(self.start_head(action_shared)) * 100.0 + 1.0
        absolute_rate = (F.softplus(self.absolute_head(shared)) * 100.0
                         if self.mode in ("hybrid", "guarded") else None)
        if self.mode == "guarded":
            above = torch.maximum(absolute_rate, base + self.min_change)
            below = torch.minimum(absolute_rate, (base - self.min_change).clamp_min(0.0))
            start_candidate = torch.where(base <= 1e-6, torch.maximum(absolute_rate,
                                                                     torch.ones_like(base)),
                                          above)
            candidates = torch.stack([absolute_rate, start_candidate, torch.zeros_like(base),
                                      above, below], -1)
        else:
            candidates = torch.stack([base, start, torch.zeros_like(base),
                                      base + steps[..., 0],
                                      (base - steps[..., 1]).clamp_min(0.0)], -1)
        index = direction.argmax(-1) + 1
        index = torch.where(event_probability >= torch.as_tensor(event_threshold, device=sequence.device),
                            index, torch.zeros_like(index))
        hard_rate = torch.gather(candidates, -1, index.unsqueeze(-1)).squeeze(-1)
        if hard:
            if self.training:
                soft = (probabilities * candidates).sum(-1)
                if absolute_rate is not None:
                    soft = (event_probability * soft
                            + (1 - event_probability) * absolute_rate)
                    action_hard = torch.gather(candidates, -1, (direction.argmax(-1) + 1).unsqueeze(-1)).squeeze(-1)
                    hard_rate = torch.where(event_probability >= torch.as_tensor(event_threshold, device=sequence.device),
                                            action_hard, absolute_rate)
                rate = hard_rate + soft - soft.detach()
            else:
                rate = hard_rate
        else:
            rate = (probabilities * candidates).sum(-1)
        if self.mode in ("hybrid", "guarded"):
            active = 1 - event_probability * probabilities[..., 2]
        else:
            active = (probabilities[..., 1] + probabilities[..., 3] + probabilities[..., 4]
                      + probabilities[..., 0] * (base > 0).to(probabilities.dtype))
        result = dict(rate_ml_h=rate, event_logit=event_logit, event_probability=event_probability,
                      action_probability=probabilities, direction_probability=direction,
                      rate_candidates_ml_h=candidates, active_probability=active,
                      delta_ml_h=self.delta_head(action_shared),
                      horizon_ml=F.softplus(self.horizon_head(shared).reshape(-1, 3, self.horizons)) * 1000.0)
        if absolute_rate is not None:
            result["absolute_rate_ml_h"] = absolute_rate
        return result


def configure_determinism(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class OldTwoBranchNet(nn.Module):
    """Accuracy branch (absolute rate) + decision branch (event/direction).

    The two GRUs do not share parameters, so the action losses cannot distort
    the rate fit.  The final output is one rate: the accuracy branch is used as
    the magnitude, and the decision branch only changes its side of the
    previous setting (or zeroes it for a predicted stop).  The combination is
    part of the model, not a post-hoc selection of two outputs.
    """

    def __init__(self, sequence_channels, static_channels, *, hidden=64, dropout=0.1,
                 horizons=3, min_change=MIN_CHANGE_ML_H, hold_mode="previous"):
        super().__init__()
        if hold_mode not in ("previous", "absolute"):
            raise ValueError("hold_mode must be 'previous' or 'absolute'")
        self.hold_mode = hold_mode
        self.rate_branch = OldEventNet(sequence_channels, static_channels, hidden=hidden,
                                       dropout=dropout, mode="absolute", horizons=horizons,
                                       min_change=min_change)
        self.action_branch = OldEventNet(sequence_channels, static_channels, hidden=hidden,
                                         dropout=dropout, mode="event", horizons=horizons,
                                         min_change=min_change)
        self.min_change = float(min_change)

    def forward(self, sequence, static, previous_rate, *, event_threshold=0.5,
                activity_threshold=0.5, hard=True):
        base = torch.nan_to_num(previous_rate, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        rate_branch = self.rate_branch(sequence, static, previous_rate)
        action = self.action_branch(sequence, static, previous_rate)
        absolute = rate_branch["rate_ml_h"]
        gate = action["event_probability"] >= torch.as_tensor(event_threshold, device=sequence.device)
        direction = action["direction_probability"].argmax(-1) + 1
        previous_positive = base > 1e-6
        above = torch.maximum(absolute, base + self.min_change)
        below = torch.minimum(absolute, (base - self.min_change).clamp_min(0.0))
        start = torch.where(previous_positive, above,
                            torch.maximum(absolute, torch.ones_like(base)))
        down = torch.where(base > self.min_change, below, torch.zeros_like(base))
        hold = base if self.hold_mode == "previous" else absolute
        candidates = torch.stack([hold, start, torch.zeros_like(base), above, down], -1)
        index = torch.where(gate, direction, torch.zeros_like(direction))
        hard_rate = torch.gather(candidates, -1, index.unsqueeze(-1)).squeeze(-1)
        probabilities = action["action_probability"].detach()
        if hard:
            if self.training:
                soft = (probabilities * candidates).sum(-1)
                rate = hard_rate + soft - soft.detach()
            else:
                rate = hard_rate
        else:
            rate = (probabilities * candidates).sum(-1)
        activity_probability = torch.sigmoid(rate_branch["active_logit"])
        active_gate = activity_probability >= torch.as_tensor(activity_threshold, device=sequence.device)
        rate = torch.where(active_gate, rate, torch.zeros_like(rate))
        return dict(rate_ml_h=rate, event_probability=action["event_probability"],
                    action_probability=action["action_probability"],
                    direction_probability=action["direction_probability"],
                    rate_candidates_ml_h=candidates,
                    active_probability=activity_probability,
                    absolute_rate_ml_h=absolute,
                    rate_branch_logit=rate_branch["active_logit"],
                    event_logit=action["event_logit"],
                    delta_ml_h=action["delta_ml_h"],
                    horizon_ml=rate_branch["horizon_ml"])
