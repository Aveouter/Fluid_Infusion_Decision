"""Evaluate the registered old-data V7 run against old-data baselines.

All methods share the same rows, masks and arithmetic.  The historical test
split is reported separately from the five-fold out-of-fold development
predictions; it is not called a clean external test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.action_fluid_model import action_classes
from utils.changepoints import detect_changes, event_prf
from utils.old_fluid_data_v1 import build

PREDICTIONS = ROOT / ".audit/old_event_v7"
REPORT = ROOT / "docs/experiments/old-event-v7-20261021"
SEEDS = (20261015, 20261016, 20261017)
ARMS = ("absolute", "two_branch")
CLASS_NAMES = ("hold", "start", "stop", "up", "down")
FLUIDS = ("crystal", "colloid", "water")


def load_prediction(folder, selection):
    with np.load(folder / "prediction_private.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["row_index"], selection):
            raise ValueError(f"Row index mismatch in {folder}")
        return {key: saved[key] for key in saved.files if key != "row_index"}


def pool_parts(records, arm):
    """Equal-seed raw-decision ensemble; returns the pooled components."""
    if arm == "two_branch":
        return dict(event_probability=np.mean([r["event_probability"] for r in records], 0),
                    action_probability=np.mean([r["action_probability"] for r in records], 0),
                    rate_candidates_ml_h=np.mean([r["rate_candidates_ml_h"] for r in records], 0),
                    active_probability=np.mean([r["active_probability"] for r in records], 0))
    return dict(rate_ml_h=np.mean([r["rate_ml_h"] for r in records], 0),
                active_probability=np.mean([r["active_probability"] for r in records], 0),
                event_probability=np.mean([r["event_probability"] for r in records], 0))


def decode_parts(parts, arm, *, event_threshold=0.5, activity_threshold=0.5):
    if arm == "two_branch":
        event = parts["event_probability"]
        action = parts["action_probability"]
        candidates = parts["rate_candidates_ml_h"]
        active = parts["active_probability"]
        index = np.where(event >= event_threshold, action[..., 1:].argmax(-1) + 1, 0)
        rate = np.take_along_axis(candidates, index[..., None], -1)[..., 0]
        return np.where(active >= activity_threshold, rate, 0.0)
    rate = parts["rate_ml_h"]
    active = parts["active_probability"]
    return np.where(active >= activity_threshold, rate, 0.0)


def pool(records, arm, *, event_threshold=0.5, activity_threshold=0.5):
    return decode_parts(pool_parts(records, arm), arm, event_threshold=event_threshold,
                        activity_threshold=activity_threshold)


def baselines(rows, windows):
    valid = windows["sequence"][:, :, -2]
    history = windows["sequence"][:, :, :3]
    out = {"persistence": rows["previous_rate"].astype(float),
           "zero": np.zeros_like(rows["target_rate"], dtype=float)}
    for k in (3, 6):
        window = history[:, -k:] * valid[:, -k:, None]
        denom = valid[:, -k:].sum(1)[:, None]
        out[f"ma{k}"] = np.divide(window.sum(1), denom,
                                  out=np.zeros_like(rows["target_rate"], dtype=float),
                                  where=denom > 0)
    return out


def patient_mae(patient, truth, prediction, mask):
    out = np.full((len(np.unique(patient)), 3), np.nan)
    for i, pid in enumerate(np.unique(patient)):
        use = patient == pid
        for c in range(3):
            m = use & mask[:, c]
            if m.any():
                out[i, c] = np.abs(prediction[m, c] - truth[m, c]).mean()
    return out


def per_patient_counts(patient, previous, truth, prediction, mask):
    actual = action_classes(previous, truth)
    forecast = action_classes(previous, prediction)
    valid = mask & (actual >= 0)
    ids = np.unique(patient)
    count = {name: np.zeros((len(ids), 3), int) for name in ("tp", "fp", "fn", "support", "pred")}
    class_count = {k: np.zeros((len(ids), 3, 5), int) for k in ("tp", "fp", "fn", "support", "pred")}
    for i, pid in enumerate(ids):
        use = valid & ((patient == pid)[:, None])
        count["tp"][i] = (use & (actual > 0) & (actual == forecast)).sum(0)
        count["fp"][i] = (use & (forecast > 0) & (actual != forecast)).sum(0)
        count["fn"][i] = (use & (actual > 0) & (actual != forecast)).sum(0)
        count["support"][i] = (use & (actual > 0)).sum(0)
        count["pred"][i] = (use & (forecast > 0)).sum(0)
        for k in range(5):
            class_count["tp"][i, :, k] = (use & (actual == k) & (forecast == k)).sum(0)
            class_count["fp"][i, :, k] = (use & (forecast == k) & (actual != k)).sum(0)
            class_count["fn"][i, :, k] = (use & (actual == k) & (forecast != k)).sum(0)
            class_count["support"][i, :, k] = (use & (actual == k)).sum(0)
            class_count["pred"][i, :, k] = (use & (forecast == k)).sum(0)
    return ids, count, class_count, actual, forecast, valid


def f1_from(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.full(np.shape(tp), np.nan, dtype=float),
                     where=denom > 0)


def summarize(name, patient, previous, truth, prediction, mask):
    ids, counts, classes, actual, forecast, valid = per_patient_counts(
        patient, previous, truth, prediction, mask)
    mae = patient_mae(patient, truth, prediction, mask)
    f1 = f1_from(counts["tp"].sum(0), counts["fp"].sum(0), counts["fn"].sum(0))
    event_tp = (valid & (actual > 0) & (forecast > 0)).sum(0)
    event_fp = (valid & (actual == 0) & (forecast > 0)).sum(0)
    event_fn = (valid & (actual > 0) & (forecast == 0)).sum(0)
    event_f1 = f1_from(event_tp, event_fp, event_fn)
    zero = truth <= 1e-6
    zero_accuracy = float((prediction[zero] <= 1e-6).mean())
    hold = valid & (actual == 0)
    false_change = float((forecast[hold] > 0).mean())
    change = actual > 0
    change_mae = float(np.abs(prediction[change] - truth[change]).mean())
    hold_mask = (actual == 0) & valid
    hold_mae = float(np.abs(prediction[hold_mask] - truth[hold_mask]).mean())
    smape = np.full((len(ids), 3), np.nan)
    for i, pid in enumerate(ids):
        use = patient == pid
        for c in range(3):
            m = use & (truth[:, c] > 0) & mask[:, c]
            if m.any():
                denom = np.abs(prediction[m, c]) + np.abs(truth[m, c])
                smape[i, c] = np.mean(np.divide(2 * np.abs(prediction[m, c] - truth[m, c]),
                                                denom, out=np.zeros_like(denom),
                                                where=denom > 0))
    per_class = {}
    for c in range(3):
        per_class[FLUIDS[c]] = {}
        for k in range(5):
            tp = int(classes["tp"][:, c, k].sum()); fp = int(classes["fp"][:, c, k].sum())
            fn = int(classes["fn"][:, c, k].sum()); support = int(classes["support"][:, c, k].sum())
            pred = int(classes["pred"][:, c, k].sum())
            per_class[FLUIDS[c]][CLASS_NAMES[k]] = dict(
                support=support, predicted=pred, tp=tp, fp=fp, fn=fn,
                precision=(tp / (tp + fp) if tp + fp else None),
                recall=(tp / (tp + fn) if tp + fn else None),
                f1=(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None))
    return dict(method=name,
                setting_mae=float(np.nanmean(np.nanmean(mae, 0))),
                setting_mae_by_fluid=np.nanmean(mae, 0).tolist(),
                setting_bias=float(np.nanmean(prediction - truth)),
                strict_action_f1=float(np.nanmean(f1)),
                strict_action_f1_by_fluid=f1.tolist(),
                event_f1=float(np.nanmean(event_f1)),
                event_f1_by_fluid=event_f1.tolist(),
                zero_flow_accuracy=zero_accuracy,
                positive_flow_smape=float(np.nanmean(np.nanmean(smape, 0))),
                false_change_rate=false_change,
                change_hour_mae=change_mae,
                hold_hour_mae=hold_mae,
                per_class=per_class,
                patient_mae=mae,
                patient_counts=counts,
                patient_class_counts=classes)


def calibration(truth, event_probability, active_probability, mask):
    """Brier and 10-bin ECE for the event and activity heads, per fluid."""
    result = {}
    for c in range(3):
        use = mask[:, c]
        target = (truth[use, c] > 0).astype(float)
        def score(probability):
            brier = float(np.mean((probability - target) ** 2))
            bins = np.clip(np.floor(probability * 10).astype(int), 0, 9)
            ece = 0.0
            for b in range(10):
                use_bin = bins == b
                if use_bin.any():
                    ece += use_bin.mean() * abs(probability[use_bin].mean() - target[use_bin].mean())
            return brier, float(ece)
        event_brier, event_ece = score(event_probability[use, c])
        active_brier, active_ece = score(active_probability[use, c])
        result[FLUIDS[c]] = dict(event_brier=event_brier, event_ece=event_ece,
                                 activity_brier=active_brier, activity_ece=active_ece)
    return result


def changepoint_summary(patient, hour, truth_rate, prediction_rate, series_rate):
    """One-to-one change-point matching on the reconstructed patient series."""
    ids = np.unique(patient)
    totals = {key: {c: dict(n_true=0, n_pred=0, n_matched=0) for c in range(3)}
              for key in ("exact", "pm1")}
    for pid in ids:
        use = np.flatnonzero(patient == pid)
        order = use[np.argsort(hour[use])]
        length = len(series_rate[pid])
        if len(order) != length - 1 or not np.array_equal(hour[order], np.arange(2, length + 1)):
            continue
        true_series = series_rate[pid].copy()
        pred_series = true_series.copy()
        pred_series[1:] = prediction_rate[order]
        true_cp = detect_changes(true_series)
        pred_cp = detect_changes(pred_series)
        for tolerance, key in ((0, "exact"), (1, "pm1")):
            match = event_prf(true_cp, pred_cp, tolerance=tolerance)
            for c in range(3):
                totals[key][c]["n_true"] += match["per_channel"][c]["n_true"]
                totals[key][c]["n_pred"] += match["per_channel"][c]["n_pred"]
                totals[key][c]["n_matched"] += match["per_channel"][c]["n_matched"]
    out = {}
    for key in ("exact", "pm1"):
        f1 = []
        for c in range(3):
            row = totals[key][c]
            tp = row["n_matched"]; fp = row["n_pred"] - tp; fn = row["n_true"] - tp
            value = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
            f1.append(value)
            out.setdefault(FLUIDS[c], {})[key] = dict(row, f1=value)
        finite = [x for x in f1 if x is not None]
        out.setdefault("macro", {})[key] = float(np.nanmean(finite)) if finite else None
    return out


def cumulative_with_observed(patient, hour, truth_rate, prediction_rate, series_rate):
    """Observed hour 0 plus predicted rows 1..H-1, per patient and fluid."""
    ids = np.unique(patient)
    result = {}
    for horizon in (24, 48, 72):
        entries = []
        for pid in ids:
            use = np.flatnonzero((patient == pid) & (hour <= horizon))
            if len(use) != horizon - 1:
                continue
            order = use[np.argsort(hour[use])]
            if not np.array_equal(hour[order], np.arange(2, horizon + 1)):
                continue
            true = series_rate[pid][0] + truth_rate[order].sum(0)
            pred = series_rate[pid][0] + prediction_rate[order].sum(0)
            entries.append(dict(patient=int(pid), true=true.tolist(), pred=pred.tolist()))
        result[horizon] = entries
    return result


def cumulative_summary(entries):
    if not entries:
        return dict(patients=0, mae=None, bias=None, tolerance_200=None, tolerance_10=None,
                    zero_dose_patients=0, zero_dose_correct=0)
    true = np.asarray([e["true"] for e in entries])
    pred = np.asarray([e["pred"] for e in entries])
    error = pred - true
    mae = np.abs(error)
    total_true = true.sum(1)
    total_pred = pred.sum(1)
    zero = np.abs(total_true) <= 1e-6
    return dict(patients=len(entries),
                mae=float(mae.mean()), bias=float(error.mean()),
                mae_by_fluid=mae.mean(0).tolist(),
                tolerance_200=float((mae <= 200.0).mean()),
                tolerance_10=float((mae <= 0.10 * np.maximum(np.abs(true), 1e-9)).mean()),
                zero_dose_patients=int(zero.sum()),
                zero_dose_correct=int((zero & (np.abs(total_pred) <= 1e-6)).sum()))


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def bootstrap(candidate, baseline, *, replicates=2000, seed=20261021):
    ids = candidate["patient_mae"].shape[0]
    rng = np.random.default_rng(seed)
    draws = rng.integers(ids, size=(replicates, ids))
    result = {}
    difference = (np.nanmean(np.nanmean(candidate["patient_mae"][draws], axis=1), axis=1)
                  - np.nanmean(np.nanmean(baseline["patient_mae"][draws], axis=1), axis=1))
    result["setting_mae"] = dict(
        difference=float(np.nanmean(np.nanmean(candidate["patient_mae"], 0))
                         - np.nanmean(np.nanmean(baseline["patient_mae"], 0))),
        ci_low=float(np.nanpercentile(difference, 2.5)),
        ci_high=float(np.nanpercentile(difference, 97.5)))
    a, b = candidate["patient_counts"], baseline["patient_counts"]
    def boot_f1(counts):
        tp = counts["tp"][draws].sum(1); fp = counts["fp"][draws].sum(1); fn = counts["fn"][draws].sum(1)
        return np.nanmean(f1_from(tp, fp, fn), 1)
    difference = boot_f1(a) - boot_f1(b)
    observed = (np.nanmean(f1_from(a["tp"].sum(0), a["fp"].sum(0), a["fn"].sum(0)))
                - np.nanmean(f1_from(b["tp"].sum(0), b["fp"].sum(0), b["fn"].sum(0))))
    result["strict_action_f1"] = dict(difference=float(observed),
                                      ci_low=float(np.nanpercentile(difference, 2.5)),
                                      ci_high=float(np.nanpercentile(difference, 97.5)))
    return result


def oof_rate(arm, rows, development, event_threshold=0.5):
    rate = np.full_like(rows["target_rate"], np.nan, dtype=float)
    for fold in range(5):
        selection = np.flatnonzero(development & (rows["fold"] == fold))
        records = [load_prediction(PREDICTIONS / f"fold{fold}_{arm}_seed{seed}", selection)
                   for seed in SEEDS]
        rate[selection] = pool(records, arm, event_threshold=event_threshold)
    return rate


def oof_seed_rate(arm, seed, rows, development):
    rate = np.full_like(rows["target_rate"], np.nan, dtype=float)
    for fold in range(5):
        selection = np.flatnonzero(development & (rows["fold"] == fold))
        records = [load_prediction(PREDICTIONS / f"fold{fold}_{arm}_seed{seed}", selection)]
        rate[selection] = decode_parts(pool_parts(records, arm), arm)
    return rate


def oof_parts(arm, rows, development):
    parts = {key: [] for key in ("event_probability", "action_probability",
                                 "rate_candidates_ml_h", "active_probability")}
    for fold in range(5):
        selection = np.flatnonzero(development & (rows["fold"] == fold))
        records = [load_prediction(PREDICTIONS / f"fold{fold}_{arm}_seed{seed}", selection)
                   for seed in SEEDS]
        pooled = pool_parts(records, arm)
        for key in parts:
            parts[key].append((selection, pooled[key]))
    return parts


def main():
    rows, windows, meta = build(ROOT)
    patients = meta["patients"]
    series = meta["series"]
    series_rate = {i: series[pid]["rate"] for i, pid in enumerate(patients)}
    development = rows["split"] == 0
    test = rows["split"] == 1
    REPORT.mkdir(parents=True, exist_ok=True)

    result = {}
    oof = {arm: oof_rate(arm, rows, development) for arm in ARMS}
    for arm, prediction in oof.items():
        if not np.isfinite(prediction[development]).all():
            raise ValueError(f"Missing OOF predictions for {arm}")
    baselines_oof = baselines(rows, windows)
    methods = dict(oof)
    methods.update(baselines_oof)
    methods["train_median"] = np.tile(np.median(rows["target_rate"][development], axis=0),
                                      (len(rows["target_rate"]), 1))
    summaries = {name: summarize(name, rows["patient"][development],
                                 rows["previous_rate"][development],
                                 rows["target_rate"][development], rate[development],
                                 rows["action_mask"][development])
                 for name, rate in methods.items()}
    pairs = {arm: {base: bootstrap(summaries[arm], summaries[base])
                   for base in ("persistence", "ma3", "ma6", "absolute", "zero", "train_median")
                   if base != arm}
             for arm in ARMS}
    cumulative = {name: {horizon: cumulative_summary(value)
                         for horizon, value in cumulative_with_observed(
                             rows["patient"][development], rows["hour"][development],
                             rows["target_rate"][development], methods[name][development],
                             series_rate).items()}
                  for name in ("two_branch", "absolute", "persistence", "ma3", "ma6", "zero")}
    changepoints = {name: changepoint_summary(
        rows["patient"][development], rows["hour"][development],
        rows["target_rate"][development], methods[name][development], series_rate)
        for name in ("two_branch", "absolute", "persistence", "ma3", "ma6", "zero")}
    sensitivity = {}
    for threshold in (0.4, 0.45, 0.5, 0.55, 0.6):
        rate = oof_rate("two_branch", rows, development, event_threshold=threshold)
        summary = summarize("two_branch", rows["patient"][development],
                            rows["previous_rate"][development],
                            rows["target_rate"][development], rate[development],
                            rows["action_mask"][development])
        sensitivity[f"{threshold:.2f}"] = {key: summary[key] for key in
                                           ("setting_mae", "change_hour_mae", "strict_action_f1",
                                            "zero_flow_accuracy", "false_change_rate")}
    parts = oof_parts("two_branch", rows, development)
    event_probability = np.empty_like(rows["target_rate"])
    active_probability = np.empty_like(rows["target_rate"])
    for selection, values in parts["event_probability"]:
        event_probability[selection] = values
    for selection, values in parts["active_probability"]:
        active_probability[selection] = values
    result["development"] = dict(
        rows=int(development.sum()), patients=int(len(np.unique(rows["patient"][development]))),
        summaries={k: {kk: vv for kk, vv in v.items()
                       if kk not in ("patient_mae", "patient_counts", "patient_class_counts")}
                   for k, v in summaries.items()},
        paired=pairs, cumulative=cumulative, changepoints=changepoints,
        threshold_sensitivity=sensitivity,
        seed_stability={str(seed): {key: value for key, value in summarize(
            "two_branch", rows["patient"][development], rows["previous_rate"][development],
            rows["target_rate"][development],
            oof_seed_rate("two_branch", seed, rows, development)[development],
            rows["action_mask"][development]).items()
            if key in ("setting_mae", "strict_action_f1", "zero_flow_accuracy")}
            for seed in SEEDS},
        fold_stability={str(fold): {key: value for key, value in summarize(
            "two_branch", rows["patient"][development & (rows["fold"] == fold)],
            rows["previous_rate"][development & (rows["fold"] == fold)],
            rows["target_rate"][development & (rows["fold"] == fold)],
            oof["two_branch"][development & (rows["fold"] == fold)],
            rows["action_mask"][development & (rows["fold"] == fold)]).items()
            if key in ("setting_mae", "strict_action_f1", "zero_flow_accuracy")}
            for fold in range(5)},
        calibration=calibration(rows["target_rate"][development],
                                event_probability[development],
                                active_probability[development],
                                rows["action_mask"][development]))

    # Historical test.
    test_pred = {}
    test_parts = {}
    for arm in ARMS:
        selection = np.flatnonzero(test)
        records = [load_prediction(PREDICTIONS / f"test_{arm}_seed{seed}", selection)
                   for seed in SEEDS]
        test_parts[arm] = pool_parts(records, arm)
        decoded = decode_parts(test_parts[arm], arm)
        full = np.full_like(rows["target_rate"], np.nan, dtype=float)
        full[selection] = decoded
        test_pred[arm] = full
    baselines_test = baselines(rows, windows)
    methods_test = dict(test_pred)
    methods_test.update(baselines_test)
    methods_test["train_median"] = np.tile(np.median(rows["target_rate"][development], axis=0),
                                           (len(rows["target_rate"]), 1))
    summaries_test = {name: summarize(name, rows["patient"][test],
                                      rows["previous_rate"][test],
                                      rows["target_rate"][test], rate[test],
                                      rows["action_mask"][test])
                      for name, rate in methods_test.items()}
    pairs_test = {arm: {base: bootstrap(summaries_test[arm], summaries_test[base])
                        for base in ("persistence", "ma3", "ma6", "absolute", "zero", "train_median")
                        if base != arm}
                  for arm in ARMS}
    cumulative_test = {name: {horizon: cumulative_summary(value)
                              for horizon, value in cumulative_with_observed(
                                  rows["patient"][test], rows["hour"][test],
                                  rows["target_rate"][test], methods_test[name][test],
                                  series_rate).items()}
                       for name in ("two_branch", "absolute", "persistence", "ma3", "ma6", "zero")}
    changepoints_test = {name: changepoint_summary(
        rows["patient"][test], rows["hour"][test],
        rows["target_rate"][test], methods_test[name][test], series_rate)
        for name in ("two_branch", "absolute", "persistence", "ma3", "ma6", "zero")}
    sensitivity_test = {}
    for threshold in (0.4, 0.45, 0.5, 0.55, 0.6):
        rate = decode_parts(test_parts["two_branch"], "two_branch", event_threshold=threshold)
        summary = summarize("two_branch", rows["patient"][test], rows["previous_rate"][test],
                            rows["target_rate"][test], rate,
                            rows["action_mask"][test])
        sensitivity_test[f"{threshold:.2f}"] = {key: summary[key] for key in
                                                ("setting_mae", "change_hour_mae",
                                                 "strict_action_f1", "zero_flow_accuracy",
                                                 "false_change_rate")}
    result["historical_test"] = dict(
        rows=int(test.sum()), patients=int(len(np.unique(rows["patient"][test]))),
        summaries={k: {kk: vv for kk, vv in v.items()
                       if kk not in ("patient_mae", "patient_counts", "patient_class_counts")}
                   for k, v in summaries_test.items()},
        paired=pairs_test, cumulative=cumulative_test, changepoints=changepoints_test,
        threshold_sensitivity=sensitivity_test,
        seed_stability={str(seed): {key: value for key, value in summarize(
            "two_branch", rows["patient"][test], rows["previous_rate"][test],
            rows["target_rate"][test],
            decode_parts(pool_parts([load_prediction(PREDICTIONS / f"test_two_branch_seed{seed}", np.flatnonzero(test))],
                                    "two_branch"), "two_branch"),
            rows["action_mask"][test]).items()
            if key in ("setting_mae", "strict_action_f1", "zero_flow_accuracy")}
            for seed in SEEDS},
        calibration=calibration(rows["target_rate"][test],
                                test_parts["two_branch"]["event_probability"],
                                test_parts["two_branch"]["active_probability"],
                                rows["action_mask"][test]))

    payload = dict(protocol="old-event-v7-20261021", seeds=list(SEEDS),
                   note=("Old 363-patient cohort, old 291/72 split. Development is five-fold "
                         "out-of-fold; the 72-patient split is historical and has been viewed. "
                         "Cumulative is observed first hour plus one-step predictions, not a rollout."),
                   results=result)
    (REPORT / "aggregate.json").write_text(json.dumps(clean(payload), indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print(json.dumps({split: {name: dict(setting_mae=s["setting_mae"],
                                          strict_action_f1=s["strict_action_f1"],
                                          zero_flow_accuracy=s["zero_flow_accuracy"],
                                          change_hour_mae=s["change_hour_mae"])
                               for name, s in result[split]["summaries"].items()}
                      for split in ("development", "historical_test")}, indent=2))


if __name__ == "__main__":
    main()
