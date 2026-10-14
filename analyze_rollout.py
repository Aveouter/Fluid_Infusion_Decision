#!/usr/bin/env python
"""教师强制 vs 开环滚动（issue #11）。

回答审稿人 1 的第 2 条意见：图 5/图 6 用的是真实临床历史，还是模型自己的
先前预测？本脚本在同一批测试患者上分别跑两种模式并对比。

开环的严格定义（见 ``utils/rollout.py``）：历史标签**与累计体积列**都来自
预测。只回填标签而沿用真实 ``cum`` 会偷偷把未来信息喂回去，那不是开环。

用法：
    python analyze_rollout.py --limit 20 --out .audit/rollout.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from utils import label_protocol as lp
from utils import rollout as RO

#: 速率上限，与 run_epoch / predict_test_set 保持一致
MAX_SPEED = 1200.0

# 与 data_loader 一致的特征列选择（data_mat = [t_norm] + tdata）
DATA_SEL = [0, 1, 2, 3, 4, 5, 7, 8, 9]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="ckpts/tcn_film_best.pth")
    p.add_argument("--data_pkl", default="data/output_data.pkl")
    p.add_argument("--base_pkl", default="data/baseline.pkl")
    p.add_argument("--limit", type=int, default=0, help="0 表示全部测试患者")
    p.add_argument("--thresholds", type=float, nargs=3, default=None)
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def build_patient_arrays(rec):
    """返回 (features(9), labels(6))，特征选择与 data_loader 一致。"""
    tdata = np.asarray(rec["tdata"], dtype=np.float32)
    label = np.asarray(rec["label"], dtype=np.float32)
    L = min(len(tdata), len(label))
    tdata, label = tdata[:L], label[:L]
    t_norm = np.linspace(0, 1, L, dtype=np.float32).reshape(-1, 1)
    data_mat = np.hstack([t_norm, tdata]).astype(np.float32)
    return data_mat[:, DATA_SEL].astype(np.float32), label


def make_predictor_factory(model, device, med, iqr, thr):
    """返回 ``bvec -> predict(inp)``，供 rollout 与终点分析复用。

    inp = [feat(9) | cum(4) | label(6)]，均为原始量纲；
    模型输入 = [缩放后的 13 列 ts | 原始 6 列标签]，与 run_epoch 一致。
    """
    def factory(bvec):
        def predict(inp):
            raw13 = np.concatenate([inp[:9], inp[9:13]]).astype(np.float32)
            scaled13 = (raw13 - med) / iqr
            model_in = np.concatenate([scaled13, inp[13:]]).astype(np.float32)
            xT = torch.tensor(model_in, dtype=torch.float32,
                              device=device).reshape(1, 1, 19)
            xB = torch.tensor(bvec, dtype=torch.float32, device=device).reshape(1, 10)
            with torch.no_grad():
                out = model(xT, xB)
            prob = torch.sigmoid(out["class_logits"])[0, 0]
            speed = torch.nn.functional.softplus(out["flow_pred"])[0, 0].clamp(0, MAX_SPEED)
            rate = (prob * speed).cpu().numpy()
            bits = (prob.cpu().numpy() >= thr).astype(np.float32)
            return np.concatenate([bits, rate])
        return predict
    return factory


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.tcn_film import build_model_tcn_film_ziln
    from predict_test_set import MODEL_CFG, MAX_SPEED, load_ckpt, resolve_thresholds

    model = build_model_tcn_film_ziln(**MODEL_CFG).to(device).eval()
    load_ckpt(model, args.ckpt, device)
    thr = np.array(resolve_thresholds(args.thresholds), dtype=np.float32)

    ts = pickle.load(open(args.data_pkl, "rb"))
    base = pickle.load(open(args.base_pkl, "rb"))

    from utils.data_loader import data_loader
    loaders = data_loader(
        ts, base, history_length=1, pred_length=1, batch_size=4, num_workers=0,
        pin_memory=False, oversample_water=False, use_robust_scaler=True,
        splits=(0.7, 0.15, 0.15), seed=42,
    )
    ds = loaders[2].dataset
    scaler = ds.transform
    if scaler is None or not hasattr(scaler, "median"):
        raise SystemExit("未取得 RobustScaler，无法复现模型输入口径")
    med, iqr = scaler.median, scaler.iqr
    print(f"[scaler] median[:3]={np.round(med[:3],3)}  dims={len(med)}")

    test_pids = list(dict.fromkeys(str(p) for p in ds.patient_ids))
    if args.limit:
        test_pids = test_pids[:args.limit]

    by_pid = {}
    for key in ts:
        rec = ts[key]
        pid = (str(rec["patient_id"]) if isinstance(rec, dict) and "patient_id" in rec
               else key.split("_")[0].split(".")[0])
        if pid in set(test_pids):
            by_pid[pid] = rec

    def make_predictor(bvec):
        def predict(inp):
            """inp = [feat(9) | cum(4) | label(6)]，feat/cum 已是原始量纲。"""
            # 模型输入 = [缩放后的 13 列 ts | 原始 6 列标签]，与 run_epoch 一致。
            # 标签部分不参与 RobustScaler（scaler 只作用于 ts）。
            raw13 = np.concatenate([inp[:9], inp[9:13]]).astype(np.float32)
            scaled13 = (raw13 - med) / iqr
            model_in = np.concatenate([scaled13, inp[13:]]).astype(np.float32)
            xT = torch.tensor(model_in, dtype=torch.float32,
                              device=device).reshape(1, 1, 19)
            xB = torch.tensor(bvec, dtype=torch.float32, device=device).reshape(1, 10)
            with torch.no_grad():
                out = model(xT, xB)
            prob = torch.sigmoid(out["class_logits"])[0, 0]
            speed = torch.nn.functional.softplus(out["flow_pred"])[0, 0].clamp(0, MAX_SPEED)
            rate = (prob * speed).cpu().numpy()
            bits = (prob.cpu().numpy() >= thr).astype(np.float32)
            return np.concatenate([bits, rate])
        return predict

    rows = []
    for pid in test_pids:
        rec = by_pid.get(pid)
        if rec is None:
            continue
        feats, labels = build_patient_arrays(rec)
        if len(labels) < 3:
            continue
        bvec = np.asarray(base.get(pid, np.zeros(10)), dtype=np.float32).ravel()[:10]
        if len(bvec) < 10:
            bvec = np.pad(bvec, (0, 10 - len(bvec)))
        bvec = np.nan_to_num(bvec, nan=0.0, posinf=0.0, neginf=0.0)

        pred_fn = make_predictor(bvec)
        cmp = RO.compare_modes(pred_fn, feats, labels)
        cmp["pid"] = pid
        cmp["hours"] = len(labels)
        rows.append(cmp)

    if not rows:
        print("没有可用患者")
        return 2

    def agg(mode, key):
        vals = [r[mode][key] for r in rows if np.isfinite(r[mode][key])]
        return float(np.mean(vals)) if vals else float("nan")

    print()
    print(f"患者数 {len(rows)}  平均时长 {np.mean([r['hours'] for r in rows]):.1f}h")
    print(f"{'模式':<18}{'MAE(正流, mL/h)':>18}{'终末累计量(mL)':>18}")
    print("-" * 56)
    result = {}
    for mode in RO.MODES:
        mae = agg(mode, "mae_positive_flow")
        cum = agg(mode, "cum_final_used")
        result[mode] = {"mae_positive_flow": mae, "cum_final_used": cum,
                        "n_patients": len(rows)}
        print(f"{mode:<18}{mae:>18.2f}{cum:>18.1f}")
    true_cum = float(np.mean([r["teacher_forced"]["cum_final_true"] for r in rows]))
    print(f"{'真实终末累计量':<18}{'':>18}{true_cum:>18.1f}")

    tf, ol = result["teacher_forced"]["mae_positive_flow"], result["open_loop"]["mae_positive_flow"]
    if np.isfinite(tf) and tf > 0:
        print(f"\n开环 / 教师强制 的 MAE 比值: {ol / tf:.2f}x")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
