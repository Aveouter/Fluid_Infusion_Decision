"""Issue #81-style endpoint summary for V7.1 on the old setting-rate task.

P1/P2 are cumulative *setting exposure* (rate x 1 h), not a reconstructed
delivered dose.  P3 is multilabel fluid activity.  All tables state unit, mask,
aggregation, horizon and eval mode.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.evaluate_old_event_v7_20261021 import (SEEDS, changepoint_summary, decode_parts,
                                                     load_prediction, pool_parts)
from utils.old_fluid_data_v1 import build

V7 = ROOT / ".audit/old_event_v7_prev"
REPORT = ROOT / "docs/experiments/old-event-v7-20261021"
FLUIDS = ("crystal", "colloid", "water")
SEED = 20261021
REPLICATES = 2000


def oof_arrays(folds):
    rows, windows, meta = build(ROOT)
    rate = np.full_like(rows["target_rate"], np.nan, dtype=float)
    event = np.full_like(rows["target_rate"], np.nan, dtype=float)
    active = np.full_like(rows["target_rate"], np.nan, dtype=float)
    development = rows["split"] == 0
    for fold in folds:
        selection = np.flatnonzero(development & (rows["fold"] == fold))
        records = [load_prediction(V7 / f"fold{fold}_two_branch_seed{seed}", selection)
                   for seed in SEEDS]
        parts = pool_parts(records, "two_branch")
        rate[selection] = decode_parts(parts, "two_branch")
        event[selection] = parts["event_probability"]
        active[selection] = parts["active_probability"]
    return rows, meta, rate, event, active


def test_arrays():
    rows, windows, meta = build(ROOT)
    test = rows["split"] == 1
    selection = np.flatnonzero(test)
    records = [load_prediction(V7 / f"test_two_branch_seed{seed}", selection) for seed in SEEDS]
    parts = pool_parts(records, "two_branch")
    rate = np.full_like(rows["target_rate"], np.nan, dtype=float)
    event = np.full_like(rows["target_rate"], np.nan, dtype=float)
    active = np.full_like(rows["target_rate"], np.nan, dtype=float)
    rate[selection] = decode_parts(parts, "two_branch")
    event[selection] = parts["event_probability"]
    active[selection] = parts["active_probability"]
    return rows, meta, rate, event, active


def ci(values, *, statistic=np.nanmean):
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(values), size=(REPLICATES, len(values)))
    sampled = np.asarray([statistic(values[draw]) for draw in draws])
    return float(statistic(values)), float(np.nanpercentile(sampled, 2.5)), float(np.nanpercentile(sampled, 97.5))


def patient_errors(patient, truth, prediction, mask):
    ids = np.unique(patient)
    out = np.full((len(ids), 3), np.nan)
    for i, pid in enumerate(ids):
        use = patient == pid
        for c in range(3):
            m = use & mask[:, c]
            if m.any():
                out[i, c] = np.abs(prediction[m, c] - truth[m, c]).mean()
    return ids, out


def cumulative_entries(rows, mask, rate, series_rate, horizon):
    entries = []
    for pid in np.unique(rows["patient"][mask]):
        use = np.flatnonzero((rows["patient"] == pid) & (rows["hour"] <= horizon))
        if len(use) != horizon - 1:
            continue
        order = use[np.argsort(rows["hour"][use])]
        if not np.array_equal(rows["hour"][order], np.arange(2, horizon + 1)):
            continue
        true = series_rate[pid][0] + rows["target_rate"][order].sum(0)
        pred = series_rate[pid][0] + rate[order].sum(0)
        entries.append((int(pid), true, pred))
    return entries


def cumulative_table(rows, mask, rate, series_rate):
    result = {}
    for horizon in (24, 48, 72):
        entries = cumulative_entries(rows, mask, rate, series_rate, horizon)
        true = np.asarray([e[1] for e in entries])
        pred = np.asarray([e[2] for e in entries])
        total_true, total_pred = true.sum(1), pred.sum(1)
        total_error = np.abs(total_pred - total_true)
        fluid_error = np.abs(pred - true)
        mae, lo, hi = ci(total_error)
        fluid_mae = []
        for c in range(3):
            fluid_mae.append(float(np.mean(fluid_error[:, c])))
        relative = np.abs(total_true) > 1e-6
        zero = ~relative
        result[str(horizon)] = dict(
            patients=len(entries),
            total_mae=mae, total_mae_ci=[lo, hi],
            total_bias=float(np.mean(total_pred - total_true)),
            fluid_mae=fluid_mae,
            tolerance_200=float(np.mean(total_error <= 200.0)),
            tolerance_10=float(np.mean(total_error[relative] <= 0.10 * np.abs(total_true[relative]))),
            relative_n_patients=int(relative.sum()),
            zero_dose_patients=int(zero.sum()),
            zero_dose_correct=int(np.sum(zero & (np.abs(total_pred) <= 1e-6))),
            zero_dose_accuracy=(float(np.mean(np.abs(total_pred[zero]) <= 1e-6)) if zero.any() else None),
        )
    return result


def activity_table(patient, truth, prediction, probability):
    ids = np.unique(patient)
    counts = np.zeros((len(ids), 3, 3), int)  # tp, fp, fn
    support = np.zeros((len(ids), 3), int)
    for i, pid in enumerate(ids):
        use = patient == pid
        actual = truth[use] > 0
        forecast = prediction[use] > 0
        support[i] = actual.sum(0)
        counts[i, :, 0] = (actual & forecast).sum(0)
        counts[i, :, 1] = (~actual & forecast).sum(0)
        counts[i, :, 2] = (actual & ~forecast).sum(0)
    tp, fp, fn = counts.sum(0).T
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.full(3, np.nan), where=(2 * tp + fp + fn) > 0)
    macro = float(np.nanmean(f1))
    micro = float(2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()))
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(ids), size=(REPLICATES, len(ids)))
    sampled = []
    for draw in draws:
        a = counts[draw]
        t, p, n = a[:, :, 0].sum(0), a[:, :, 1].sum(0), a[:, :, 2].sum(0)
        sampled.append(np.nanmean(np.divide(2 * t, 2 * t + p + n,
                                           out=np.full(3, np.nan), where=(2 * t + p + n) > 0)))
    per_class = {}
    for c, name in enumerate(FLUIDS):
        per_class[name] = dict(support=int(support[:, c].sum()),
                               precision=(float(tp[c] / (tp[c] + fp[c])) if tp[c] + fp[c] else None),
                               recall=(float(tp[c] / (tp[c] + fn[c])) if tp[c] + fn[c] else None),
                               f1=(float(f1[c]) if np.isfinite(f1[c]) else None))
    # Brier/ECE for the activity probability, window-level per fluid.
    calibration = {}
    target = (truth > 0).astype(float)
    for c, name in enumerate(FLUIDS):
        prob = probability[:, c]
        brier = float(np.mean((prob - target[:, c]) ** 2))
        bins = np.clip(np.floor(prob * 10).astype(int), 0, 9)
        ece = 0.0
        for b in range(10):
            m = bins == b
            if m.any():
                ece += m.mean() * abs(prob[m].mean() - target[m, c].mean())
        calibration[name] = dict(brier=brier, ece=float(ece))
    return dict(macro_f1=macro, macro_f1_ci=[float(np.nanpercentile(sampled, 2.5)),
                                             float(np.nanpercentile(sampled, 97.5))],
                micro_f1=micro, per_class=per_class, calibration=calibration)


def event_calibration(truth, probability, mask):
    result = {}
    for c, name in enumerate(FLUIDS):
        use = mask[:, c]
        target = (truth[use, c] > 0).astype(float)
        prob = probability[use, c]
        brier = float(np.mean((prob - target) ** 2))
        bins = np.clip(np.floor(prob * 10).astype(int), 0, 9)
        ece = 0.0
        for b in range(10):
            m = bins == b
            if m.any():
                ece += m.mean() * abs(prob[m].mean() - target[m].mean())
        result[name] = dict(brier=brier, ece=float(ece))
    return result


def secondary(rows, mask, rate, series_rate):
    patient = rows["patient"][mask]
    truth = rows["target_rate"][mask]
    prediction = rate[mask]
    ids = np.unique(patient)
    mae = np.full((len(ids), 3), np.nan)
    rmse = np.full((len(ids), 3), np.nan)
    positive = np.full((len(ids), 3), np.nan)
    zero = np.full((len(ids), 3), np.nan)
    smape = np.full((len(ids), 3), np.nan)
    change = np.full((len(ids), 3), np.nan)
    nochange = np.full((len(ids), 3), np.nan)
    tv_true = np.full((len(ids), 3), np.nan)
    tv_pred = np.full((len(ids), 3), np.nan)
    zero_pred = np.zeros((len(ids), 3), int)
    zero_support = np.zeros((len(ids), 3), int)
    for i, pid in enumerate(ids):
        use = patient == pid
        for c in range(3):
            error = prediction[use, c] - truth[use, c]
            mae[i, c] = np.abs(error).mean()
            rmse[i, c] = np.sqrt(np.mean(error ** 2))
            pos = truth[use, c] > 0
            if pos.any():
                positive[i, c] = np.abs(error[pos]).mean()
                denom = np.abs(prediction[use, c][pos]) + np.abs(truth[use, c][pos])
                smape[i, c] = np.mean(np.divide(2 * np.abs(error[pos]), denom,
                                                out=np.zeros_like(denom), where=denom > 0))
            zerom = truth[use, c] <= 1e-6
            if zerom.any():
                zero[i, c] = np.abs(error[zerom]).mean()
                zero_support[i, c] = zerom.sum()
                zero_pred[i, c] = (prediction[use, c][zerom] <= 1e-6).sum()
            if (truth[use, c][:-1] != truth[use, c][1:]).any():
                pass
            # TV on observed row 0 + predicted rows for the patient.
            series = np.concatenate([[series_rate[pid][0, c]], prediction[use, c]])
            true_series = series_rate[pid][:, c]
            tv_pred[i, c] = np.abs(np.diff(series)).sum()
            tv_true[i, c] = np.abs(np.diff(true_series)).sum()
    return dict(
        all_hours_mae=float(np.nanmean(np.nanmean(mae, 0))),
        all_hours_mae_by_fluid=np.nanmean(mae, 0).tolist(),
        all_hours_rmse=float(np.nanmean(np.nanmean(rmse, 0))),
        all_hours_rmse_by_fluid=np.nanmean(rmse, 0).tolist(),
        positive_flow_mae=float(np.nanmean(np.nanmean(positive, 0))),
        positive_flow_mae_by_fluid=np.nanmean(positive, 0).tolist(),
        positive_flow_smape=float(np.nanmean(np.nanmean(smape, 0))),
        positive_flow_smape_by_fluid=np.nanmean(smape, 0).tolist(),
        zero_flow_mae=float(np.nanmean(np.nanmean(zero, 0))),
        zero_flow_mae_by_fluid=np.nanmean(zero, 0).tolist(),
        zero_flow_exact=float(zero_pred.sum() / max(1, zero_support.sum())),
        tv_true_mean=float(np.nanmean(np.nanmean(tv_true, 0))),
        tv_pred_mean=float(np.nanmean(np.nanmean(tv_pred, 0))),
        tv_absolute_difference=float(np.nanmean(np.nanmean(np.abs(tv_pred - tv_true), 0))),
    )


def audit(rows, mask):
    truth = rows["target_rate"][mask]
    positive = (truth > 0).mean(0)
    all_zero = float((truth.sum(1) <= 1e-6).mean())
    full = {}
    for horizon in (24, 48, 72):
        count = 0
        for pid in np.unique(rows["patient"][mask]):
            use = np.flatnonzero((rows["patient"] == pid) & (rows["hour"] <= horizon))
            if len(use) == horizon - 1:
                count += 1
        full[str(horizon)] = count
    return dict(patients=int(len(np.unique(rows["patient"][mask]))),
                windows=int(mask.sum()), positive_rate=positive.tolist(),
                all_zero_hour_rate=all_zero, full_horizon_patients=full)


def evaluate(rows, meta, rate, event, active, mask, label):
    series_rate = {i: meta["series"][pid]["rate"] for i, pid in enumerate(meta["patients"])}
    truth = rows["target_rate"][mask]
    if not np.isfinite(rate[mask]).all():
        raise ValueError(f"Missing predictions for {label}")
    return dict(
        label=label,
        audit=audit(rows, mask),
        cumulative=cumulative_table(rows, mask, rate, series_rate),
        activity=activity_table(rows["patient"][mask], truth, rate[mask], active[mask]),
        event_calibration=event_calibration(truth, event[mask], rows["action_mask"][mask]),
        secondary=secondary(rows, mask, rate, series_rate),
        changepoints=changepoint_summary(rows["patient"][mask], rows["hour"][mask],
                                         truth, rate[mask], series_rate),
    )


def main():
    rows_dev, meta_dev, rate_dev, event_dev, active_dev = oof_arrays(range(1, 5))
    mask_dev = (rows_dev["split"] == 0) & (rows_dev["fold"] > 0)
    rows_test, meta_test, rate_test, event_test, active_test = test_arrays()
    mask_test = rows_test["split"] == 1
    payload = dict(
        protocol="issue81-style-v7-20261021",
        note=("P1/P2 are cumulative documented setting exposure (rate x 1 h), not a "
              "reconstructed delivered dose. Evaluation is one-step observed history; "
              "cumulative is the observed first hour plus one-step predictions, not a rollout. "
              "Development folds 1-4 also selected prev_base vs prev_neg15; the historical "
              "test is the independent check."),
        development=evaluate(rows_dev, meta_dev, rate_dev, event_dev, active_dev,
                             mask_dev, "development_folds1to4"),
        historical_test=evaluate(rows_test, meta_test, rate_test, event_test, active_test,
                                 mask_test, "historical_test"),
    )
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "issue81_style.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                                          allow_nan=False), encoding="utf-8")
    print(json.dumps({split: dict(
        P1_72=payload[split]["cumulative"]["72"]["total_mae"],
        P2_72_200=payload[split]["cumulative"]["72"]["tolerance_200"],
        P2_72_10=payload[split]["cumulative"]["72"]["tolerance_10"],
        P3_macro=payload[split]["activity"]["macro_f1"],
        all_hours_mae=payload[split]["secondary"]["all_hours_mae"],
        positive_mae=payload[split]["secondary"]["positive_flow_mae"],
        zero_exact=payload[split]["secondary"]["zero_flow_exact"],
    ) for split in ("development", "historical_test")}, indent=2))


if __name__ == "__main__":
    main()
