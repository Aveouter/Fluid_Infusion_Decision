#!/usr/bin/env python
"""全部预设终点的补报（issue #12）。

**为什么不能在窗口化评估上算**：出厂配置是 ``history_length=1, pred_length=1``，
每个窗口只有一个目标步，于是 ``run_epoch`` 里的
``cumsum(inst_true, dim=1).sum(-1)`` 退化成**单步体积**——
方法里承诺的"累积剂量"从未真正累积过，轨迹类终点（累计、容差、稳定性）
在该配置下无法产生。这正是它们缺席 Results 的原因。

本脚本改用 #11 的开环滚动：每位患者一条 ~62 步的轨迹，在其上计算全部终点。

用法：
    python analyze_endpoints.py --limit 20 --out .audit/endpoints.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from utils import endpoints as EP
from utils import label_protocol as lp


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="ckpts/tcn_film_best.pth")
    p.add_argument("--data_pkl", default="data/output_data.pkl")
    p.add_argument("--base_pkl", default="data/baseline.pkl")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import torch
    from analyze_rollout import build_patient_arrays, make_predictor_factory

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from models.tcn_film import build_model_tcn_film_ziln
    from predict_test_set import MODEL_CFG, load_ckpt, resolve_thresholds

    model = build_model_tcn_film_ziln(**MODEL_CFG).to(device).eval()
    load_ckpt(model, args.ckpt, device)
    thr = np.array(resolve_thresholds(None), dtype=np.float32)

    ts = pickle.load(open(args.data_pkl, "rb"))
    base = pickle.load(open(args.base_pkl, "rb"))

    from utils.data_loader import data_loader
    loaders = data_loader(ts, base, history_length=1, pred_length=1, batch_size=4,
                          num_workers=0, pin_memory=False, oversample_water=False,
                          use_robust_scaler=True, splits=(0.7, 0.15, 0.15), seed=42)
    ds = loaders[2].dataset
    scaler = ds.transform
    med, iqr = scaler.median, scaler.iqr

    pids = list(dict.fromkeys(str(p) for p in ds.patient_ids))[:args.limit]
    by_pid = {}
    for key in ts:
        rec = ts[key]
        pid = (str(rec["patient_id"]) if isinstance(rec, dict) and "patient_id" in rec
               else key.split("_")[0].split(".")[0])
        if pid in set(pids):
            by_pid[pid] = rec

    make_pred = make_predictor_factory(model, device, med, iqr, thr)

    from utils import rollout as RO
    true_seq, pred_seq = [], []
    for pid in pids:
        rec = by_pid.get(pid)
        if rec is None:
            continue
        feats, labels = build_patient_arrays(rec)
        if len(labels) < 3:
            continue
        bvec = np.nan_to_num(np.asarray(base.get(pid, np.zeros(10)),
                                        dtype=np.float32).ravel()[:10],
                             nan=0.0, posinf=0.0, neginf=0.0)
        if len(bvec) < 10:
            bvec = np.pad(bvec, (0, 10 - len(bvec)))
        r = RO.rollout(make_pred(bvec), feats, labels, mode="open_loop")
        true_seq.append(r["target"])
        pred_seq.append(r["pred"])

    lens = [len(s) for s in true_seq]
    print(f"患者 {len(true_seq)}  轨迹长度 最短 {min(lens)} / 中位 {int(np.median(lens))} "
          f"/ 最长 {max(lens)} 步")

    # 患者时长不等，**不能**按最短截断（会把长轨迹的信息丢掉）。
    # 逐患者算终点，再对患者等权平均 —— 与 #6/#7 的独立性口径一致。
    per_patient = []
    for t_i, p_i in zip(true_seq, pred_seq):
        if len(t_i) < 2:
            continue
        per_patient.append(EP.evaluate_all(t_i[None, ...], p_i[None, ...],
                                           rate_definition="rate"))

    keys = list(per_patient[0].keys())
    results = {"rate": {}}
    print()
    print("=== 速率口径 = rate（prob × softplus(flow_pred)，开环，逐患者平均）===")
    print(f"{'终点':<26}{'值':>12}{'单位':>26}{'聚合':>10}")
    print("-" * 76)
    for k in keys:
        vals = [pp[k]["value"] for pp in per_patient if np.isfinite(pp[k]["value"])]
        v = float(np.mean(vals)) if vals else float("nan")
        meta = per_patient[0][k]
        results["rate"][k] = {**{kk: vv for kk, vv in meta.items() if kk != "value"},
                              "value": v, "n_patients": len(vals)}
        shown = "nan" if not np.isfinite(v) else f"{v:.4f}"
        print(f"{k:<26}{shown:>12}{meta['unit']:>26}{meta['aggregation']:>10}")
    results["rate"]["_n_patients_total"] = len(per_patient)

    # 容差终点的构成单独看（把各患者的分母与达标数汇总）
    tot = abs_ok = pct_ok = zero_den = 0
    for t_i, p_i in zip(true_seq, pred_seq):
        if len(t_i) < 2:
            continue
        r = EP.tolerance_accuracy(t_i[:, lp.RATE_SLICE], p_i[:, lp.RATE_SLICE])
        tot += r["n"]; abs_ok += r["n_abs_ok"]
        pct_ok += r["n_pct_ok"]; zero_den += r["n_zero_denominator"]
    print()
    print(f"容差终点构成：总 {tot}  绝对达标 {abs_ok}  相对达标 {pct_ok}  "
          f"真值0的分母 {zero_den}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2, default=float)
        print(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
