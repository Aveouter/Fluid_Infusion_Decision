"""Training for the old-data event model and its absolute-rate control."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch.nn import functional as F

from utils.old_event_model_v1 import OldEventNet, OldTwoBranchNet, configure_determinism
from utils.old_fluid_data_v1 import fit_scaler, patient_weights, transform

DEFAULT_CONFIG = dict(hidden=64, dropout=0.1, epochs=60, batch_size=512,
                      learning_rate=0.002, weight_decay=0.01, gradient_clip=5.0,
                      event_weight=0.5, direction_weight=0.5, magnitude_weight=0.2,
                      active_weight=0.2, delta_weight=0.1, horizon_weight=0.2,
                      absolute_weight=0.5, balanced_event=False, use_delta=False,
                      use_horizon=False, min_change=50.0, detach_actions=False,
                      hold_mode="previous", magnitude_source="guarded",
                      event_negative_weight=1.0, event_fbeta_weight=0.0, event_fbeta_beta=1.0,
                      mean_weight=0.0, positive_mean_weight=0.0,
                      rate_quantile=0.5, quantile_positive_only=True,
                      cumulative_weight=0.0, cumulative_scale=1000.0)


def _patient_batches(patient_rows, batch_size, rng):
    """Yield batches that keep each patient's rows together."""
    patients = list(patient_rows)
    rng.shuffle(patients)
    batch, count = [], 0
    for pid in patients:
        rows = patient_rows[pid]
        if batch and count + len(rows) > batch_size:
            yield np.concatenate(batch)
            batch, count = [], 0
        batch.append(rows)
        count += len(rows)
    if batch:
        yield np.concatenate(batch)


def _cumulative_groups(rows, train):
    """Patient -> [(horizon, global row indices)] with complete 2..H coverage."""
    groups = {}
    for pid in np.unique(rows["patient"][train]):
        idx = np.flatnonzero(train & (rows["patient"] == pid))
        hours = rows["hour"][idx]
        for horizon in (24, 48, 72):
            inside = (hours >= 2) & (hours <= horizon)
            if (int(inside.sum()) == horizon - 1
                    and int(hours[inside].min()) == 2
                    and int(hours[inside].max()) == horizon):
                groups.setdefault(int(pid), []).append((horizon, idx[inside]))
    return groups


def _class_weights(action, mask, train, *, exponent, cap, minimum):
    weights = np.ones((3, 5), dtype=np.float32)
    for c in range(3):
        labels = action[train & mask[:, c], c]
        count = np.bincount(labels, minlength=5).astype(float)
        base = max(1.0, count.sum())
        weights[c] = np.clip((base / (5.0 * np.maximum(1.0, count))) ** exponent,
                             minimum, cap)
    return weights


def _event_positive_weights(action, mask, train, *, exponent=0.75, cap=8.0):
    weights = np.ones((3, 5), dtype=np.float32)
    for c in range(3):
        count = np.bincount(action[train & mask[:, c], c], minlength=5).astype(float)
        hold = max(1.0, count[0])
        weights[c] = np.clip((hold / np.maximum(1.0, count)) ** exponent, 1.0, cap)
    return weights


def fit(windows, rows, train, config, seed, *, progress=None):
    config = dict(DEFAULT_CONFIG, **config)
    configure_determinism(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = np.asarray(train, bool)
    scaler = fit_scaler(windows, rows, train)
    transformed = transform(windows, scaler)
    mode = config.get("mode", "event")
    if mode == "two_branch":
        model = OldTwoBranchNet(transformed["sequence"].shape[-1], transformed["static"].shape[-1],
                                hidden=config["hidden"], dropout=config["dropout"],
                                horizons=rows["cum_target"].shape[1],
                                min_change=config["min_change"],
                                hold_mode=config.get("hold_mode", "previous"),
                                magnitude_source=config.get("magnitude_source", "guarded")).to(device)
    else:
        model = OldEventNet(transformed["sequence"].shape[-1], transformed["static"].shape[-1],
                            hidden=config["hidden"], dropout=config["dropout"], mode=mode,
                            horizons=rows["cum_target"].shape[1],
                            min_change=config["min_change"],
                            detach_actions=config.get("detach_actions", False)).to(device)
    tensors = {k: torch.as_tensor(v, device=device) for k, v in transformed.items()}
    previous = torch.as_tensor(rows["previous_rate"], dtype=torch.float32, device=device)
    target = torch.as_tensor(rows["target_rate"], dtype=torch.float32, device=device)
    active = torch.as_tensor(rows["target_active"], dtype=torch.float32, device=device)
    action = torch.as_tensor(rows["action"], device=device)
    action_mask = torch.as_tensor(rows["action_mask"], device=device)
    patient_all = torch.as_tensor(rows["patient"], device=device)
    hour_all = torch.as_tensor(rows["hour"], device=device)
    cum_target = torch.as_tensor(np.nan_to_num(rows["cum_target"]), dtype=torch.float32, device=device)
    cum_mask = torch.as_tensor(rows["cum_mask"], device=device)

    weights = {
        "rate": torch.as_tensor(patient_weights(rows["patient"], rows["action_mask"], train), device=device),
        "active": torch.as_tensor(patient_weights(rows["patient"], rows["action_mask"], train), device=device),
    }
    if mode in ("event", "hybrid", "guarded", "two_branch"):
        event_mask = rows["action_mask"] & (rows["action"] > 0)
        weights["event"] = torch.as_tensor(patient_weights(rows["patient"], rows["action_mask"], train), device=device)
        weights["direction"] = torch.as_tensor(patient_weights(rows["patient"], event_mask, train), device=device)
        weights["magnitude"] = weights["direction"]
        weights["delta"] = weights["direction"]
        type_weights = _class_weights(rows["action"], rows["action_mask"], train,
                                      exponent=0.5, cap=4.0, minimum=0.5)
        positive = (_event_positive_weights(rows["action"], rows["action_mask"], train)
                    if config["balanced_event"] else np.ones((3, 5), dtype=np.float32))
        type_weights = torch.as_tensor(type_weights, device=device)
        positive = torch.as_tensor(positive, device=device)
        horizon_weights = []
        for h in range(rows["cum_mask"].shape[1]):
            horizon_weights.append(torch.as_tensor(
                patient_weights(rows["patient"], rows["cum_mask"][:, h][:, None].repeat(3, 1), train),
                device=device))
    else:
        type_weights = positive = None
        horizon_weights = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"], eta_min=1e-4)
    rng = np.random.default_rng(seed)
    indices = np.flatnonzero(train)
    cumulative = float(config.get("cumulative_weight", 0.0)) > 0
    patient_rows = ({int(pid): np.flatnonzero(train & (rows["patient"] == pid))
                     for pid in np.unique(rows["patient"][train])}
                    if cumulative else None)
    cumulative_groups = _cumulative_groups(rows, train) if cumulative else {}
    trace = []
    for epoch in range(config["epochs"]):
        model.train()
        losses = []
        batches = (_patient_batches(patient_rows, config["batch_size"], rng)
                   if cumulative else
                   np.array_split(rng.permutation(indices),
                                  max(1, int(np.ceil(len(indices) / config["batch_size"])))))
        for batch in batches:
            at = torch.as_tensor(batch, device=device)
            output = model(tensors["sequence"][at], tensors["static"][at], previous[at])
            error = output["rate_ml_h"] - target[at]
            quantile = float(config.get("rate_quantile", 0.5))
            if quantile != 0.5:
                pinball = torch.maximum(quantile * error, (quantile - 1.0) * error)
                element = (torch.where(target[at] > 0, pinball, error.abs())
                           if config.get("quantile_positive_only", True) else pinball)
            else:
                element = error.abs()
            rate_loss = (element / 100.0 * weights["rate"][at]).mean()
            active_logit = (output.get("active_logit", output.get("rate_branch_logit")) if mode in ("absolute", "two_branch")
                            else torch.logit(output["active_probability"].clamp(1e-6, 1 - 1e-6)))
            active_loss = (F.binary_cross_entropy_with_logits(
                active_logit, active[at], reduction="none") * weights["active"][at]).mean()
            loss = rate_loss + config["active_weight"] * active_loss
            if config["mean_weight"] > 0:
                total_residual = ((output["rate_ml_h"] - target[at]).sum(1) / 100.0).mean()
                loss = loss + config["mean_weight"] * total_residual ** 2
            if config["positive_mean_weight"] > 0:
                positive_rows = target[at] > 0
                terms = []
                for c in range(3):
                    if positive_rows[:, c].any():
                        residual = ((output["rate_ml_h"][positive_rows[:, c], c]
                                     - target[at][positive_rows[:, c], c]) / 100.0).mean()
                        terms.append(residual ** 2)
                if terms:
                    loss = loss + config["positive_mean_weight"] * torch.stack(terms).mean()
            if cumulative:
                position = {int(row): i for i, row in enumerate(batch)}
                patient_set = {int(pid) for pid in np.unique(rows["patient"][batch])}
                terms = []
                for pid in patient_set:
                    for horizon, global_rows in cumulative_groups.get(pid, ()):
                        local = torch.as_tensor([position[int(row)] for row in global_rows],
                                                device=device)
                        residual = (output["rate_ml_h"][local]
                                    - target[at][local]).sum(0).sum()
                        terms.append(residual.abs() / float(config["cumulative_scale"]))
                if terms:
                    loss = loss + config["cumulative_weight"] * torch.stack(terms).mean()
            if mode in ("hybrid", "guarded", "two_branch"):
                absolute_loss = ((output["absolute_rate_ml_h"] - target[at]).abs()
                                 / 100.0 * weights["rate"][at]).mean()
                loss = loss + config["absolute_weight"] * absolute_loss
            if mode in ("event", "hybrid", "guarded", "two_branch"):
                event_target = (action[at] > 0).to(output["event_probability"].dtype)
                positive_weight = positive.unsqueeze(0).expand(len(at), -1, -1).gather(
                    -1, action[at].clamp_min(0).unsqueeze(-1)).squeeze(-1)
                event = F.binary_cross_entropy_with_logits(
                    output["event_logit"], event_target, reduction="none")
                negative = torch.full_like(positive_weight, config["event_negative_weight"])
                event = (event * torch.where(event_target > 0, positive_weight, negative)
                         * weights["event"][at]).mean()
                direction = torch.stack([
                    F.cross_entropy(output["direction_probability"][:, c].clamp_min(1e-8).log(),
                                    (action[at, c] - 1).clamp_min(0), weight=type_weights[c, 1:],
                                    reduction="none") for c in range(3)], -1)
                direction = (direction * weights["direction"][at]).mean()
                source = (output.get("action_candidates_ml_h", output["rate_candidates_ml_h"])
                          if config.get("magnitude_source", "guarded") == "action"
                          else output["rate_candidates_ml_h"])
                chosen = source.gather(-1, action[at].clamp(0, 4).unsqueeze(-1)).squeeze(-1)
                magnitude = ((chosen - target[at]).abs() / 100.0 * weights["magnitude"][at]).mean()
                loss = loss + config["event_weight"] * event + config["direction_weight"] * direction
                loss = loss + config["magnitude_weight"] * magnitude
                if config["event_fbeta_weight"] > 0:
                    probability = torch.sigmoid(output["event_logit"])
                    beta2 = float(config["event_fbeta_beta"]) ** 2
                    weight = weights["event"][at]
                    tp = (weight * probability * event_target).sum()
                    fp = (weight * probability * (1 - event_target)).sum()
                    fn = (weight * (1 - probability) * event_target).sum()
                    fbeta = (1 + beta2) * tp / ((1 + beta2) * tp + beta2 * fn + fp + 1e-6)
                    loss = loss + config["event_fbeta_weight"] * (1 - fbeta)
                if config["use_delta"]:
                    delta_target = (target[at] - previous[at]) / 100.0
                    delta = F.smooth_l1_loss(output["delta_ml_h"] / 100.0, delta_target,
                                             reduction="none")
                    loss = loss + config["delta_weight"] * (delta * weights["delta"][at]).mean()
                if config["use_horizon"]:
                    horizon = ((output["horizon_ml"] / 1000.0 - cum_target[at] / 1000.0).abs()
                               * cum_mask[at].unsqueeze(-1).to(torch.float32))
                    horizon = (horizon * torch.stack([w[at] for w in horizon_weights], 1)).sum() / cum_mask[at].sum().clamp_min(1)
                    loss = loss + config["horizon_weight"] * horizon
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite old-data training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach()))
        schedule.step()
        record = dict(epoch=epoch + 1, training_loss=float(np.mean(losses)))
        trace.append(record)
        if progress:
            progress(record)
    return dict(architecture=dict(sequence_channels=transformed["sequence"].shape[-1],
                                  static_channels=transformed["static"].shape[-1],
                                  hidden=config["hidden"], dropout=config["dropout"],
                                  mode=mode, horizons=rows["cum_target"].shape[1],
                                  min_change=config["min_change"],
                                  detach_actions=config.get("detach_actions", False),
                                  hold_mode=config.get("hold_mode", "previous"),
                                  magnitude_source=config.get("magnitude_source", "guarded"),
                                  kind=("two_branch" if mode == "two_branch" else "single")),
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                scaler=scaler, config=copy.deepcopy(config), trace=trace,
                seed=int(seed), training_patients=np.unique(rows["patient"][train])), model


def predict(bundle, windows, rows, selection=None, *, event_threshold=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kind = bundle["architecture"].get("kind", "single")
    kwargs = {k: v for k, v in bundle["architecture"].items() if k != "kind"}
    if kind == "two_branch":
        kwargs.pop("mode", None)
        kwargs.pop("detach_actions", None)
    else:
        kwargs.pop("hold_mode", None)
        kwargs.pop("magnitude_source", None)
    model = (OldTwoBranchNet if kind == "two_branch" else OldEventNet)(**kwargs).to(device)
    model.load_state_dict(bundle["state"])
    model.eval()
    transformed = transform({k: (v[selection] if (selection is not None and k in ("sequence", "static")) else v)
                             for k, v in windows.items()}, bundle["scaler"])
    previous = rows["previous_rate"] if selection is None else rows["previous_rate"][selection]
    output = {}
    with torch.no_grad():
        for start in range(0, len(transformed["static"]), 1024):
            batch = {k: torch.as_tensor(v[start:start + 1024], device=device)
                     for k, v in transformed.items()}
            base = torch.as_tensor(previous[start:start + 1024], dtype=torch.float32, device=device)
            result = model(batch["sequence"], batch["static"], base,
                           event_threshold=event_threshold)
            for key, value in result.items():
                output.setdefault(key, []).append(value.cpu().numpy())
    return {key: np.concatenate(values) for key, values in output.items()}
