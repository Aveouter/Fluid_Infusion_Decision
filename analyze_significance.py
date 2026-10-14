#!/usr/bin/env python
"""患者级 bootstrap 与配对比较（issue #13）。

回答审稿人 1、2、3 的核心质疑：BurnResu 与基线的差异是否超出抽样噪声？

重采样单位是**患者**——同患者的窗口高度相关，按窗口重采样会人为压窄区间。
配对比较中两个方法共享同一组重采样索引。

用法：
    python analyze_significance.py --bundle preds/bundle.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils import bootstrap as BS


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", default="preds/bundle.npz")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def per_patient_mae(pred_rates, true_rates, patient_ids, mask=None):
    """逐患者的 MAE（仅真实正流单元）。返回 (values, pids)。"""
    if mask is None:
        mask = true_rates > 0
    vals, pids = [], []
    for i, pid in enumerate(patient_ids):
        m = mask[i]
        if m.sum() == 0:
            continue
        vals.append(float(np.abs(pred_rates[i][m] - true_rates[i][m]).mean()))
        pids.append(str(pid))
    return np.array(vals, dtype=float), pids


def main(argv=None):
    args = parse_args(argv)
    d = np.load(args.bundle, allow_pickle=True)
    tr = d["target_rates"]
    pids = [str(x) for x in d["patient_id"]]

    print(f"窗口 {len(tr)}  患者 {len(set(pids))}  "
          f"正流单元 {int((tr > 0).sum())}")
    print()

    methods = {
        "BurnResu (rate=prob*speed)": d["rate"],
        "BurnResu (speed, 论文口径)": d["speed"],
        "persistence": d["pers_rates"],
        f"moving_average_k{int(d['ma_k'][0])}": d["ma_rates"],
    }

    # ---- 各方法的患者级点估计与区间（口径：正流掩码 + 实际预测）----
    results = {}
    print("== 各方法的患者级 MAE（正流，mL/h）==")
    print(f"{'方法':<30}{'MAE':>10}{'95% CI':>22}{'患者':>6}")
    print("-" * 70)
    for name, pred in methods.items():
        vals, p = per_patient_mae(pred, tr, pids)
        r = BS.patient_clustered_bootstrap(vals, p, n_boot=args.n_boot, seed=args.seed)
        results[name] = r
        ci = f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
        print(f"{name:<30}{r['point']:>10.2f}{ci:>22}{r['n_patients']:>6}")

    # ---- 配对比较 ----
    print()
    print("== 配对 bootstrap（Δ = BurnResu − 基线；**正值表示 BurnResu 更差**）==")
    print(f"{'对比':<34}{'ΔMAE':>9}{'95% CI':>22}{'跨0?':>7}")
    print("-" * 74)
    ref_vals, ref_p = per_patient_mae(d["rate"], tr, pids)
    paired = {}
    for name in ("persistence", f"moving_average_k{int(d['ma_k'][0])}"):
        vals, p = per_patient_mae(methods[name], tr, pids)
        common = sorted(set(ref_p) & set(p))
        idx_a = {q: i for i, q in enumerate(ref_p)}
        idx_b = {q: i for i, q in enumerate(p)}
        a = np.array([ref_vals[idx_a[q]] for q in common])
        b = np.array([vals[idx_b[q]] for q in common])
        r = BS.paired_bootstrap_diff(a, b, common, n_boot=args.n_boot, seed=args.seed)
        paired[name] = r
        verdict = "跨 0" if r["crosses_zero"] else ("更差" if r["delta"] > 0 else "更优")
        ci = f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
        print(f"{'BurnResu vs ' + name:<34}{r['delta']:>9.2f}{ci:>22}{verdict:>7}")

    # ---- 分类指标同样处理 ----
    from sklearn.metrics import f1_score
    print()
    print("== 分类指标（macro F1，逐患者）× 配对 bootstrap ==")
    thr = np.asarray(d["thresholds"], dtype=float)
    tb, prob = d["target_bits"], d["prob"]
    pb = (prob >= thr.reshape(1, -1)).astype(int)

    def per_patient_f1(pred_bits):
        """**先按患者合并**其全部窗口，再算该患者的 macro F1。

        逐窗口算再平均会把窗口多的小时当成独立样本，且同一患者的窗口
        高度相关——这正是本票要纠正的单位错误。
        """
        grouped = {}
        for i, pid in enumerate(pids):
            grouped.setdefault(str(pid), [[], []])
            grouped[str(pid)][0].append(tb[i].astype(int))
            grouped[str(pid)][1].append(pred_bits[i].astype(int))
        vals, p = [], []
        for pid, (ys, yhs) in sorted(grouped.items()):
            y = np.concatenate(ys, axis=0)
            yh = np.concatenate(yhs, axis=0)
            if y.sum() == 0:
                continue
            vals.append(float(f1_score(y, yh, average="macro", zero_division=0)))
            p.append(pid)
        return np.array(vals), p

    model_vals, model_p = per_patient_f1(pb)
    print(f"  BurnResu macro F1（患者级均值）= {model_vals.mean():.4f}  "
          f"患者 {len(model_vals)}")
    for name in ("persistence", f"moving_average_k{int(d['ma_k'][0])}"):
        base_bits = ((methods[name] > 0).astype(int)
                     if name != "BurnResu (speed, 论文口径)" else pb)
        bv, bp = per_patient_f1(base_bits)
        common = sorted(set(model_p) & set(bp))
        ia = {q: i for i, q in enumerate(model_p)}
        ib = {q: i for i, q in enumerate(bp)}
        a = np.array([model_vals[ia[q]] for q in common])
        b = np.array([bv[ib[q]] for q in common])
        r = BS.paired_bootstrap_diff(a, b, common, n_boot=args.n_boot, seed=args.seed)
        verdict = "跨 0" if r["crosses_zero"] else ("BurnResu 更优" if r["delta"] > 0 else "BurnResu 更差")
        print(f"  BurnResu vs {name:<20} ΔF1={r['delta']:+.4f} "
              f"CI=[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  {verdict}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"per_method": results, "paired": paired},
                  open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2, default=float)
        print(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
