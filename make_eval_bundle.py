#!/usr/bin/env python
"""构建对齐的评测包，供 Phase 1 的 #10–#13 共用。

**为什么需要它**：``data_loader`` 会 ``rng.shuffle(patients)``，数据集内的患者
顺序与按字典序重建的顺序不同。跨脚本比较必须先对齐，否则窗口数与目标形状都
对得上、数值却完全错位（本项目已因此产出过一次错误结论）。

本脚本产出 ``preds/bundle.npz``，其中每一行都对应**同一个测试窗口**：

    patient_id      该窗口所属患者
    target_bits     真实状态位 (N,3)
    target_rates    真实速率   (N,3)   mL/h
    prob            模型类别概率 (N,3)
    speed           模型回归头条件均值 (N,3)   mL/h
    rate            模型实际输出 prob*speed (N,3)  mL/h
    pers_rates      持续性基线预测 (N,3)  mL/h
    ma_rates        移动平均(k) 预测 (N,3)  mL/h

用法：
    python make_eval_bundle.py --k 3 --out preds/bundle.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils import baselines as B


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds", default="preds/test.npz")
    p.add_argument("--data_pkl", default="data/output_data.pkl")
    p.add_argument("--base_pkl", default="data/baseline.pkl")
    p.add_argument("--k", type=int, default=3, help="移动平均窗口")
    p.add_argument("--out", default="preds/bundle.npz")
    return p.parse_args(argv)


def aligned_series(ts, order):
    """按数据集内的患者顺序，把每个窗口还原成 (决策时刻, 完整标签序列)。

    数据集按「打乱后的患者顺序」逐患者追加窗口；这里用同一个顺序重建，
    并逐窗口对应回去。
    """
    wanted = set(order)
    grouped = {}
    for key in ts:
        rec = ts[key]
        pid = (str(rec["patient_id"]) if isinstance(rec, dict) and "patient_id" in rec
               else key.split("_")[0].split(".")[0])
        if pid not in wanted:
            continue
        label = np.asarray(rec["label"], dtype=np.float32)
        T = len(label)
        # 与 data_loader 一致：窗口长 H+P=2，起点 0..T-2
        grouped[pid] = [(i, label) for i in range(0, max(0, T - 1))]

    seen = {}
    out = []
    for pid in order:
        idx = seen.get(pid, 0)
        seen[pid] = idx + 1
        out.append(grouped[pid][idx])
    return out


def main(argv=None):
    args = parse_args(argv)
    import pickle
    ts = pickle.load(open(args.data_pkl, "rb"))

    pred = np.load(args.preds, allow_pickle=True)
    order = [str(x) for x in pred["patient_id"]]
    series = aligned_series(ts, order)

    targets = np.concatenate([pred["target_bits"][:, 0, :],
                              pred["target_rates"][:, 0, :]], axis=1)
    check = np.stack([lab[t + 1] for t, lab in series], axis=0)
    if not np.allclose(targets, check):
        raise SystemExit("对齐失败：重建的目标与数据集目标不一致，拒绝产出 bundle")

    pers = np.stack([B.persistence_next(lab[:t + 1], index=t) for t, lab in series], axis=0)
    ma = np.stack([B.moving_average_next(lab[:t + 1], k=args.k, index=t)
                   for t, lab in series], axis=0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        patient_id=np.array(order),
        target_bits=pred["target_bits"][:, 0, :],
        target_rates=pred["target_rates"][:, 0, :],
        prob=pred["prob"][:, 0, :],
        speed=pred["speed"][:, 0, :],
        rate=pred["rate"][:, 0, :],
        pers_bits=pers[:, :3], pers_rates=pers[:, 3:],
        ma_bits=ma[:, :3], ma_rates=ma[:, 3:],
        thresholds=pred["thresholds"], ma_k=np.array([args.k]),
    )
    n_pos = int((pred["target_rates"][:, 0, :] > 0).sum())
    print(f"[bundle] {out}  windows={len(order)} patients={len(set(order))} "
          f"positive_cells={n_pos}  ma_k={args.k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
