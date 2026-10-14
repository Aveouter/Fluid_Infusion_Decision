#!/usr/bin/env python
"""导出 BurnResu 在测试集上的逐窗口预测，供 Phase 1 的下游分析使用。

产物是一份 (N, ...) 的 npz，含患者编号、目标、概率与速率预测。有了它，
变点分析（#10）、开环对照（#11）、预设终点（#12）与患者级 bootstrap（#13）
都可以在同一份预测上进行，且彼此口径一致。

输入构造与 ``run_epoch.py`` 完全一致：
    inp_T = cat(ts[:, :-P], lab[:, :-P])      # 窗口前 H 步
    target = lab[:, -P:, :]                   # 窗口后 P 步

用法：
    python predict_test_set.py --ckpt ckpts/tcn_film_best.pth --out preds/test.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from utils import label_protocol as lp

# 与 training_metadata.json 记录的训练配置一致
MODEL_CFG = dict(
    input_dim_base=10, input_dim_temporal=19, embed_dim=256,
    num_heads=4, hidden_dim=512, num_classes=3, pred_length=1,
    dropout=0.1, tcn_k=3, tcn_layers_short=3, tcn_layers_long=5,
    n_transformer_layers=4, ff_mult=4, causal=True,
)
MAX_SPEED = 1200.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="ckpts/tcn_film_best.pth")
    p.add_argument("--data_pkl", default="data/output_data.pkl")
    p.add_argument("--base_pkl", default="data/baseline.pkl")
    p.add_argument("--history_length", type=int, default=1)
    p.add_argument("--pred_length", type=int, default=1)
    p.add_argument("--splits", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--thresholds", type=float, nargs=3, default=None,
                   help="默认读取 ckpts/best_thresholds.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="preds/test.npz")
    return p.parse_args(argv)


def build_model(device):
    from models.tcn_film import build_model_tcn_film_ziln
    model = build_model_tcn_film_ziln(**MODEL_CFG).to(device)
    return model


def load_ckpt(model, path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[ckpt] 缺失键 {len(missing)} 个，例如 {missing[:3]}")
    if unexpected:
        print(f"[ckpt] 多余键 {len(unexpected)} 个，例如 {unexpected[:3]}")
    print(f"[ckpt] 载入 {path}  epoch={ck.get('epoch')} "
          f"best_f1={ck.get('best_f1')} best_mae={ck.get('best_mae')}")
    return ck


def resolve_thresholds(path):
    if path is None:
        p = Path("ckpts/best_thresholds.json")
        if p.exists():
            thr = json.load(open(p, encoding="utf-8"))
            print(f"[thresholds] 取自 {p}: {[round(float(x), 3) for x in thr]}")
            return [float(x) for x in thr]
        print("[thresholds] 未找到，回退 0.5")
        return [0.5, 0.5, 0.5]
    return [float(x) for x in path]


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)

    ts = __import__("pickle").load(open(args.data_pkl, "rb"))
    base = __import__("pickle").load(open(args.base_pkl, "rb"))

    from utils.data_loader import data_loader
    loaders = data_loader(
        ts, base, history_length=args.history_length, pred_length=args.pred_length,
        batch_size=args.batch_size, num_workers=0, pin_memory=False,
        oversample_water=False, use_robust_scaler=True,
        splits=tuple(args.splits), seed=args.seed,
    )
    test_loader = loaders[2]
    if test_loader is None:
        print("测试划分为空")
        return 2

    ds = test_loader.dataset
    pids = getattr(ds, "patient_ids", None)
    if pids is None and hasattr(ds, "dataset"):
        pids = list(ds.dataset.patient_ids[i] for i in ds.indices)
    pids = np.array([str(p) for p in pids])
    print(f"[test] windows={len(ds)} patients={len(set(pids))}")

    model = build_model(device).eval()
    load_ckpt(model, args.ckpt, device)
    thr = resolve_thresholds(args.thresholds)

    P = args.pred_length
    probs, rates, speeds, targets = [], [], [], []
    for ts_b, base_b, lab_b in test_loader:
        ts_b = torch.nan_to_num(ts_b.to(device), nan=0., posinf=0., neginf=0.)
        lab_b = torch.nan_to_num(lab_b.to(device), nan=0., posinf=0., neginf=0.)
        inp_T = torch.cat((ts_b[:, :-P, :], lab_b[:, :-P, :]), dim=-1)
        inp_T = torch.nan_to_num(inp_T, nan=0., posinf=0., neginf=0.)
        out = model(inp_T, base_b.to(device))
        prob = torch.sigmoid(out["class_logits"])
        speed = torch.nn.functional.softplus(out["flow_pred"]).clamp(0.0, MAX_SPEED)
        rate = torch.clamp(prob * speed, 0.0, MAX_SPEED)
        probs.append(prob[:, -P:, :].float().cpu().numpy())
        rates.append(rate[:, -P:, :].float().cpu().numpy())
        speeds.append(speed[:, -P:, :].float().cpu().numpy())
        targets.append(lab_b[:, -P:, :].float().cpu().numpy())

    prob = np.concatenate(probs, axis=0).reshape(len(ds), P, 3)
    rate = np.concatenate(rates, axis=0).reshape(len(ds), P, 3)
    speed = np.concatenate(speeds, axis=0).reshape(len(ds), P, 3)
    tgt = np.concatenate(targets, axis=0).reshape(len(ds), P, 6)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        patient_id=pids, prob=prob, rate=rate, speed=speed,
        target_bits=tgt[:, :, :3], target_rates=tgt[:, :, 3:],
        thresholds=np.array(thr), history_length=args.history_length,
        pred_length=P, seed=args.seed,
    )
    print(f"[written] {out_path}  prob={prob.shape} rate={rate.shape} speed={speed.shape}")

    # 即时对照：用记录口径复算，确认与 test_metrics.json 一致
    tb, tr = tgt[:, :, :3], tgt[:, :, 3:]
    mask = tr > 0
    mae_rate = float(np.abs(rate[mask] - tr[mask]).mean()) if mask.sum() else float("nan")
    mae_speed = float(np.abs(speed[mask] - tr[mask]).mean()) if mask.sum() else float("nan")
    from sklearn.metrics import f1_score
    pred_bits = (prob >= np.array(thr).reshape(1, 1, 3)).astype(int)
    f1 = float(f1_score(tb.astype(int).reshape(-1, 3), pred_bits.reshape(-1, 3),
                        average="macro", zero_division=0))
    print(f"[recheck] MAE(正流) rate=prob*speed -> {mae_rate:.4f}")
    print(f"[recheck] MAE(正流) speed(条件均值)  -> {mae_speed:.4f}")
    print(f"[recheck] macro F1={f1:.4f}")
    mp = Path("ckpts/test_metrics.json")
    if mp.exists():
        m = json.load(open(mp, encoding="utf-8"))["overall"]
        print(f"[recorded] MAE={m['mae_all_pos']:.4f}  macro F1={m['f1_macro']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
