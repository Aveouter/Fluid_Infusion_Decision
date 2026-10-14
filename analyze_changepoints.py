#!/usr/bin/env python
"""变更点分层分析（issue #10）。

读取 ``preds/bundle.npz``（由 ``make_eval_bundle.py`` 产出，已对齐到数据集
真实顺序），按稳定/变更/邻域分层比较 BurnResu 与朴素基线，并统计变更类型
细分与时点误差。

用法：
    python analyze_changepoints.py --bundle preds/bundle.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils import changepoints as CP


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", default="preds/bundle.npz")
    p.add_argument("--radius", type=int, default=1)
    p.add_argument("--tolerance", type=int, default=2)
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    d = np.load(args.bundle, allow_pickle=True)
    tr = d["target_rates"]
    pids = [str(x) for x in d["patient_id"]]
    methods = {"BurnResu": d["rate"], "persistence": d["pers_rates"],
               f"moving_average_k{int(d['ma_k'][0])}": d["ma_rates"]}

    changes = CP.detect_changes(tr)
    n_cells = int(tr.size)
    n_pos = int((tr > 0).sum())
    print(f"窗口 {len(tr)}  患者 {len(set(pids))}  单元 {n_cells}  正流单元 {n_pos}")
    print(f"变更点单元 {int(changes.sum())}  "
          f"含变更窗口 {int(changes.any(axis=1).sum())}  "
          f"涉及患者 {len({p for p, m in zip(pids, changes.any(axis=1)) if m})}")
    print()

    result = {}
    print(f"{'方法':<20}{'stable':>10}{'changed':>10}{'邻域':>10}")
    print("-" * 52)
    for name, pred in methods.items():
        m = CP.stratified_metrics(pred, tr, changes, radius=args.radius)
        result[name] = m
        print(f"{name:<20}{m['stable']['mae']:>10.2f}{m['changed']['mae']:>10.2f}"
              f"{m['changed_neighborhood']['mae']:>10.2f}")

    print()
    print(f"{'方法':<20}{'变更点':>8}{'命中率':>9}{'中位滞后':>10}{'未命中':>8}")
    print("-" * 56)
    timing = {}
    for name, pred in methods.items():
        e = CP.change_timing_error(pred, tr, tolerance=args.tolerance)
        timing[name] = e
        print(f"{name:<20}{e['n_true_change_points']:>8}{e['match_rate']:>9.3f}"
              f"{e['median_lag_hours']:>+10.1f}{e['n_unmatched']:>8}")

    print()
    print(f"{'变更类型':<12}{'n':>7}{'真值为0':>10}{'BurnResu':>10}{'持续性':>10}")
    print("-" * 50)
    per_type = {}
    for t in CP.CHANGE_TYPES:
        b = CP.per_type_metrics(d["rate"], tr, changes)[t]
        s = CP.per_type_metrics(d["pers_rates"], tr, changes)[t]
        per_type[t] = {"burnresu": b, "persistence": s}
        print(f"{t:<12}{b['n_cells']:>7}{b['n_zero_target']:>10}"
              f"{b['mae']:>10.2f}{s['mae']:>10.2f}")

    # 停止输注是临床上最关键的失效模式，单独报出
    types = CP.change_type_matrix(tr, changes)
    off = types == "on_to_off"
    if off.sum():
        rate = d["rate"]
        missed = float((rate[off] > 0).mean())
        pred_mean = float(rate[off].mean())
        print()
        print("== 应停止输注时的表现 ==")
        print(f"  单元数 {int(off.sum())}")
        print(f"  模型仍预测正速率的比例 {missed:.1%}")
        print(f"  模型平均仍预测 {pred_mean:.1f} mL/h（真值为 0）")
        result["stop_infusion"] = {"n_cells": int(off.sum()),
                                   "positive_rate_fraction": missed,
                                   "mean_predicted_rate": pred_mean}

    result["timing"] = timing
    result["per_type"] = per_type
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2, default=float)
        print(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
