#!/usr/bin/env python
"""朴素时序基线 vs BurnResu 的对照评估（issue #9）。

回答审稿人 1 的质疑：模型是否只是在复制上一小时？

对照在**完全相同的测试窗口**上进行：同一批患者、同一批窗口、同一条
预测目标。基线只用窗口内的历史标签，与模型输入所看到的历史信息一致。

用法：
    python evaluate_baselines.py \
        --data_pkl data/output_data.pkl \
        --base_pkl data/baseline.pkl \
        --history_length 1 --pred_length 1 \
        --splits 0.7 0.15 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from utils import baselines as B
from utils import label_protocol as lp

POS_FLOOR = 1e-6


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_pkl", default="data/output_data.pkl")
    p.add_argument("--base_pkl", default="data/baseline.pkl")
    p.add_argument("--history_length", type=int, default=1)
    p.add_argument("--pred_length", type=int, default=1)
    p.add_argument("--splits", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ma_k", type=int, nargs="+", default=[1, 2, 3, 6],
                   help="候选移动平均窗口，在验证集上选")
    p.add_argument("--metrics_json", default="ckpts/test_metrics.json",
                   help="BurnResu 的记录指标，用于对照")
    p.add_argument("--out", default="", help="结果 JSON 输出路径")
    return p.parse_args(argv)


def load_pkl(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def patient_id_of(key, rec):
    """与 ``utils/data_loader.extract_patient_id`` 保持一致的推导规则。

    两处若不一致，基线就会落在与模型不同的患者集合上，对照失效。
    """
    if isinstance(rec, dict) and "patient_id" in rec:
        return str(rec["patient_id"])
    return key.split("_")[0].split(".")[0]


def build_windows_for_patients(ts_data, patients, H, P):
    """按患者产出 (决策时刻 t, 该患者完整标签序列)。

    与 ``data_loader`` 的窗口规则一致：窗口长 ``H+P``，目标为窗口最后 ``P`` 步。
    返回**完整序列**而非仅历史窗口，是因为移动平均需要回看 k 小时——
    那同样属于决策时刻之前的信息，模型看不到只是因为窗口只给了 1 步
    （``history_length=1``），而基线没有这个限制。
    """
    out = []
    for key in ts_data:
        pid = patient_id_of(key, ts_data[key])
        if pid not in patients:
            continue
        label = np.asarray(ts_data[key]["label"], dtype=np.float32)
        T = len(label)
        for i in range(0, max(0, T - H - P + 1)):
            y = label[i:i + H + P]
            if len(y) != H + P or np.isnan(y).any():
                continue
            t = i + H - 1                      # 决策时刻在完整序列中的下标
            out.append((t, label))
    return out


def history_window(label, t, H):
    """决策时刻 t 处模型可见的历史窗口（最近 H 步）。"""
    lo = max(0, t - H + 1)
    return label[lo:t + 1]


def target_window(label, t, P):
    """目标：t+1 .. t+P。"""
    return label[t + 1:t + 1 + P]


def macro_f1(y_true_bits, y_pred_bits):
    from sklearn.metrics import f1_score
    return float(f1_score(y_true_bits.astype(int), y_pred_bits.astype(int),
                          average="macro", zero_division=0))


def positive_flow_mae(y_true_rate, y_pred_rate):
    mask = y_true_rate > POS_FLOOR
    if mask.sum() == 0:
        return float("nan"), 0
    return float(np.abs(y_pred_rate[mask] - y_true_rate[mask]).mean()), int(mask.sum())


def main(argv=None):
    args = parse_args(argv)

    ts = load_pkl(args.data_pkl)
    base = load_pkl(args.base_pkl)
    H, P = args.history_length, args.pred_length

    # 复用 data_loader 的划分逻辑，确保与模型评测的是同一批患者
    from utils.data_loader import data_loader
    loaders = data_loader(
        ts, base, history_length=H, pred_length=P, batch_size=4,
        num_workers=0, pin_memory=False, oversample_water=False,
        use_robust_scaler=False, splits=tuple(args.splits), seed=args.seed,
    )
    val_loader, test_loader = loaders[1], loaders[2]
    if test_loader is None:
        print("测试划分为空，无法评估", file=sys.stderr)
        return 2

    def patients_of(loader):
        ds = loader.dataset
        pids = getattr(ds, "patient_ids", None)
        if pids is None and hasattr(ds, "dataset"):
            pids = getattr(ds.dataset, "patient_ids", None)
        return set(pids) if pids else set()

    val_p, test_p = patients_of(val_loader), patients_of(test_loader)
    print(f"[split] seed={args.seed} splits={tuple(args.splits)} "
          f"val_patients={len(val_p)} test_patients={len(test_p)}")

    test_windows = build_windows_for_patients(ts, test_p, H, P)
    val_windows = build_windows_for_patients(ts, val_p, H, P)
    print(f"[windows] val={len(val_windows)} test={len(test_windows)}")
    if not test_windows:
        print("未构造出测试窗口", file=sys.stderr)
        return 2

    y_true = np.concatenate([target_window(lab, t, P) for t, lab in test_windows], axis=0)

    def predict_all(windows, k):
        """对每个窗口用其**完整前序序列**做 k 小时因果移动平均。"""
        return np.stack(
            [B.moving_average_next(lab[:t + 1], k=k, index=t) for t, lab in windows],
            axis=0)

    def predict_persistence(windows):
        return np.stack(
            [B.persistence_next(lab[:t + 1], index=t) for t, lab in windows], axis=0)

    results = {}
    results["persistence"] = evaluate(predict_persistence(test_windows), y_true)

    # ---- 移动平均：k 在验证集上选（绝不用测试集） ----
    val_true = (np.concatenate([target_window(lab, t, P) for t, lab in val_windows], axis=0)
                if val_windows else None)
    if val_windows:
        scores = {}
        for k in args.ma_k:
            p = predict_all(val_windows, k)
            mae, _ = positive_flow_mae(val_true[:, lp.RATE_SLICE], p[:, lp.RATE_SLICE])
            scores[k] = mae
        best_k = min(scores, key=scores.get)
        print("[moving-average] 验证集 MAE(mL/h, 正流): "
              + ", ".join(f"k={k}:{v:.2f}" for k, v in scores.items())
              + f"  -> k={best_k}")
    else:
        best_k = 1
        print("[moving-average] 无验证窗口，回退 k=1")

    results[f"moving_average_k{best_k}"] = evaluate(predict_all(test_windows, best_k),
                                                    y_true)

    # ---- BurnResu 记录值 ----
    mp = Path(args.metrics_json)
    if mp.exists():
        m = json.load(open(mp, encoding="utf-8"))
        ov = m.get("overall", {})
        results["BurnResu_recorded"] = {
            "mae_positive_flow_ml_per_h": float(ov.get("mae_all_pos", float("nan"))),
            "rmse_positive_flow_ml_per_h": float(ov.get("rmse_all_pos", float("nan"))),
            "f1_macro": float(ov.get("f1_macro", float("nan"))),
            "auc_macro": float(ov.get("auc_macro", float("nan"))),
            "source": str(mp),
            "note": "来自随包记录，测试集口径；阈值 [0.45, 0.30, 0.55]",
        }

    print()
    print(f"{'method':<22} {'MAE(mL/h,pos)':>14} {'F1(macro)':>11} {'n_pos':>8}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<22} {r.get('mae_positive_flow_ml_per_h', float('nan')):>14.4f} "
              f"{r.get('f1_macro', float('nan')):>11.4f} "
              f"{r.get('n_positive_cells', 0):>8}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[written] {args.out}")
    return 0


def evaluate(pred, y_true):
    """pred / y_true 均为 (N, 6)，列序见 label_protocol。"""
    t_rate = y_true[:, lp.RATE_SLICE]
    p_rate = pred[:, lp.RATE_SLICE]
    mae, n_pos = positive_flow_mae(t_rate, p_rate)
    return {
        "mae_positive_flow_ml_per_h": mae,
        "n_positive_cells": n_pos,
        "f1_macro": macro_f1(y_true[:, lp.BIT_SLICE], pred[:, lp.BIT_SLICE]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
