#!/usr/bin/env python
"""消融变体的重评与显著性检验（issue #16，对应 R2-3c / R3-5 / R4-4）。

审稿人 2 与 3 都指出表 3 的异常：移除自注意力后 AUC 几乎不变而 MAE 从 27.44
跳到 74.67，移除 TCN 却几乎无影响。本脚本用**同一批测试窗口**重跑全部消融权重，
给出患者级区间与配对差异，取代只靠点估计判断"哪个组件重要"。

用法：
    python analyze_ablation.py --out .audit/ablation.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from utils import bootstrap as BS
from utils import label_protocol as lp
from utils import metrics as M

#: 权重文件 -> 构建函数名。table 列给出论文表 3 的对应行。
VARIANTS = {
    "full": ("tcn_film_best.pth", "build_tcnfilm_full"),
    "wo_selfattn": ("ours_wo_selfattn_best.pth", "build_tcnfilm_wo_selfattn"),
    "wo_tcn": ("ours_wo_tcn_best.pth", "build_tcnfilm_wo_tcn"),
    "wo_film": ("ours_wo_film_best.pth", "build_tcnfilm_wo_film"),
    "wo_temporal": ("ours_wo_temporal_best.pth", "build_tcnfilm_wo_temporal"),
    "wo_baseline": ("ours_wo_baseline_best.pth", "build_tcnfilm_wo_baseline"),
}

#: 论文表 3 报告的测试集 MAE（用于对账）
PAPER_TEST_MAE = {"full": 27.44, "wo_selfattn": 74.67, "wo_tcn": 27.63,
                  "wo_film": 28.88, "wo_temporal": 72.42, "wo_baseline": 74.32}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", default="ckpts")
    p.add_argument("--data_pkl", default="data/output_data.pkl")
    p.add_argument("--base_pkl", default="data/baseline.pkl")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="")
    return p.parse_args(argv)


@torch.no_grad()
def run_variant(builder, ckpt, loader, device, thresholds, P=1, max_speed=1200.0):
    from predict_test_set import MODEL_CFG, load_ckpt
    model = builder(**MODEL_CFG).to(device).eval()
    load_ckpt(model, ckpt, device)

    probs, rates, tgts = [], [], []
    for ts_b, base_b, lab_b in loader:
        ts_b = torch.nan_to_num(ts_b.to(device), nan=0., posinf=0., neginf=0.)
        lab_b = torch.nan_to_num(lab_b.to(device), nan=0., posinf=0., neginf=0.)
        inp = torch.cat((ts_b[:, :-P, :], lab_b[:, :-P, :]), dim=-1)
        out = model(torch.nan_to_num(inp, nan=0., posinf=0., neginf=0.), base_b.to(device))
        prob = torch.sigmoid(out["class_logits"])
        speed = torch.nn.functional.softplus(out["flow_pred"]).clamp(0, max_speed)
        rate = torch.clamp(prob * speed, 0, max_speed)
        probs.append(prob[:, -P:, :].float().cpu().numpy())
        rates.append(rate[:, -P:, :].float().cpu().numpy())
        tgts.append(lab_b[:, -P:, :].float().cpu().numpy())

    prob = np.concatenate(probs).reshape(len(loader.dataset), P, 3)
    rate = np.concatenate(rates).reshape(len(loader.dataset), P, 3)
    tgt = np.concatenate(tgts).reshape(len(loader.dataset), P, 6)
    bits = (prob >= np.asarray(thresholds).reshape(1, 1, 3)).astype(int)
    return {"prob": prob, "rate": rate, "tgt": tgt, "bits": bits}


def per_patient(pred_rate, pred_bits, tgt, pids):
    """逐患者 MAE（正流）与 macro F1（先按患者合并窗口）。

    本仓库 P=1，故把 (N, P, C) 压成 (N, C) 处理。
    """
    from sklearn.metrics import f1_score
    pred_rate = pred_rate[:, 0, :]
    pred_bits = pred_bits[:, 0, :]
    tr, tb = tgt[:, 0, 3:], tgt[:, 0, :3]
    grouped = {}
    for i, pid in enumerate(pids):
        m = tr[i] > 0
        d = grouped.setdefault(str(pid), {"ae": [], "y": [], "yh": [], "n": 0})
        if m.sum():
            d["ae"].extend(np.abs(pred_rate[i][m] - tr[i][m]).tolist())
            d["n"] += int(m.sum())
        d["y"].append(tb[i].astype(int))
        d["yh"].append(pred_bits[i].astype(int))

    mae_vals, f1_vals, keys = [], [], []
    for pid in sorted(grouped):
        d = grouped[pid]
        keys.append(pid)
        mae_vals.append(float(np.mean(d["ae"])) if d["ae"] else np.nan)
        y = np.concatenate(d["y"]); yh = np.concatenate(d["yh"])
        f1_vals.append(float(f1_score(y, yh, average="macro", zero_division=0)))
    return np.array(mae_vals, float), np.array(f1_vals, float), keys


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import models.tcn_film as tcn
    ts = pickle.load(open(args.data_pkl, "rb"))
    base = pickle.load(open(args.base_pkl, "rb"))

    from utils.data_loader import data_loader
    loaders = data_loader(ts, base, history_length=1, pred_length=1, batch_size=8,
                          num_workers=0, pin_memory=False, oversample_water=False,
                          use_robust_scaler=True, splits=(0.7, 0.15, 0.15), seed=42)
    test_loader = loaders[2]
    ds = test_loader.dataset
    pids = list(ds.patient_ids)
    thr = json.load(open(Path(args.ckpt_dir) / "best_thresholds.json", encoding="utf-8"))
    print(f"测试窗口 {len(ds)}  患者 {len(set(pids))}  阈值 {[round(float(t),2) for t in thr]}")
    print()

    out = {}
    for name, (fname, builder_name) in VARIANTS.items():
        ck = Path(args.ckpt_dir) / fname
        if not ck.exists():
            print(f"[skip] {name}: 缺少 {fname}")
            continue
        builder = getattr(tcn, builder_name)
        r = run_variant(builder, str(ck), test_loader, device, thr)
        mae, f1, keys = per_patient(r["rate"], r["bits"], r["tgt"], pids)
        out[name] = {"mae": mae, "f1": f1, "keys": keys,
                     "mae_mean": float(np.nanmean(mae)),
                     "f1_mean": float(np.nanmean(f1))}
        print(f"[done] {name:<12} 患者级 MAE={np.nanmean(mae):7.2f}  "
              f"macro F1={np.nanmean(f1):.4f}")

    # ---- 汇总表 ----
    print()
    print(f"{'变体':<14}{'患者级MAE':>11}{'95% CI':>20}{'论文MAE':>10}{'比值':>8}")
    print("-" * 66)
    for name, d in out.items():
        r = BS.patient_clustered_bootstrap(d["mae"], d["keys"],
                                           n_boot=args.n_boot, seed=args.seed)
        d["ci"] = r
        paper = PAPER_TEST_MAE.get(name, float("nan"))
        ratio = paper / d["mae_mean"] if d["mae_mean"] else float("nan")
        ci = f"[{r['ci_low']:.1f}, {r['ci_high']:.1f}]"
        print(f"{name:<14}{d['mae_mean']:>11.2f}{ci:>20}{paper:>10.2f}{ratio:>8.2f}")

    # ---- 配对：各变体 vs 完整模型 ----
    print()
    # 符号约定：paired_bootstrap_diff(A, B) 返回 A − B，此处 A = full、B = 变体。
    # 因此值恒为 **full − 变体**：负值表示 full 的 MAE 更低（更好）。
    # 标签必须与这个方向一致——早期版本的表头写作"变体 − full"，与实际符号相反。
    print(f"{'对比 (full − 变体)':<26}{'ΔMAE':>9}{'95% CI':>20}{'跨0?':>8}")
    print("-" * 64)
    ref = out.get("full")
    paired = {}
    if ref:
        for name, d in out.items():
            if name == "full":
                continue
            common = sorted(set(ref["keys"]) & set(d["keys"]))
            ia = {k: i for i, k in enumerate(ref["keys"])}
            ib = {k: i for i, k in enumerate(d["keys"])}
            a = np.array([ref["mae"][ia[k]] for k in common])
            b = np.array([d["mae"][ib[k]] for k in common])
            ok = np.isfinite(a) & np.isfinite(b)
            r = BS.paired_bootstrap_diff(a[ok], b[ok], [common[i] for i in np.flatnonzero(ok)],
                                         n_boot=args.n_boot, seed=args.seed)
            paired[name] = r
            ci = f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
            print(f"{'full − ' + name:<26}{r['delta']:>9.2f}{ci:>20}"
                  f"{('跨 0' if r['crosses_zero'] else '显著'):>8}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                       for kk, vv in v.items()} for k, v in out.items()},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False,
                  indent=2, default=float)
        print(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
