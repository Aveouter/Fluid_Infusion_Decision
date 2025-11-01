#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Inference for Fluid Project (集成版, 修复版)

改动要点：
1) 去掉重复 df.to_csv；
2) --device 默认更稳健（优先 cuda:0，否则 cpu）；
3) choose_test_ids 使用 Ts/Base 键交集；
4) 推理 CSV 额外写入 pred_speed_raw_* 以支持绘图阶段“高阈值重门控”；
5) ZILN/回归头输出维度更健壮的 last-step 处理；
6) 更清晰的 step_ml_scale 单位说明；
7) 空汇总提示；
8) 其它细节增强（打印加载信息、额外 PDF 导出可选）。
9) 【新】补齐完整时间轴，并且从“label 首次出现补液”时刻起才允许非零预测，之前一律清零。

用法示例：
# 仅推理：
python inference.py \
  --mode infer \
  --data_pkl data/output_data.pkl \
  --base_pkl data/baseline.pkl \
  --model flim \
  --ckpt ./ckpts/flim_lr0.0001_bs16.pth \
  --thresholds_json ./ckpts/best_thresholds.json \
  --output_dir ./per_patient_outputs

# 只出图（基于 infer 阶段保存的 CSV）：
python inference.py --mode plot --output_dir ./per_patient_outputs
"""

import os
import re
import json
import argparse
import pickle
from typing import Dict, List, Tuple, Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ====== 模型（与训练保持一致）======
from models.Baseline import BaselineNetwork, BaselineNetwork_1
from models.cnn import CNNBaselineNetwork
from models.transformer import TransformerBaselineNetwork
from models.Lstm import LSTMBaselineNetwork
from models.Flim import build_model_film_ziln
from utils.data_loader import data_loader
from models.tcn_film import build_model_tcn_film_ziln

# =============================
# 工具函数（安全数值化 / ID 对齐）
# =============================
NumericLike = Union[pd.DataFrame, pd.Series, np.ndarray, Sequence[float]]

def ensure_numeric_array(x: NumericLike, *, allow_1d: bool = False) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        arr = x.select_dtypes(include=[np.number]).to_numpy()
    elif isinstance(x, pd.Series):
        arr = x.to_numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim == 1 and not allow_1d:
        arr = arr.reshape(-1, 1)
    arr = arr.astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    return arr

def unify_id(key: str) -> str:
    key = str(key)
    if key.endswith('.csv'):
        key = key[:-4]
    return key

def normalize_keys_inplace(Ts_data: Dict[str, Any], Base_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    Ts2 = {unify_id(k): v for k, v in Ts_data.items()}
    Bs2 = {unify_id(k): v for k, v in Base_data.items()}
    return Ts2, Bs2


# --------- 金标准液体复苏公式 ---------

def calculate_formula_fluid(formula: str, tbsa: float, weight: float, phase: str = 'first') -> float:
    k = 1000.0
    formulas = {
        'Evans':    {'first': (2.0, 2*k), 'second': (1.0, 2*k)},
        'Brooke':   {'first': (2.0, 2*k), 'second': (1.0, 2*k)},
        'Parkland': {'first': (4.0, 0.0), 'second': (0.0, 2*k)},
        'Monafo':   {'first': (2.0, 0.0), 'second': (1.0, 0.0)},
        'TMMU':     {'first': (1.5, 2*k), 'second': (0.75, 2*k)},
        'RJH':      {'first': (1.5, 3.5*k), 'second': (0.75, 3.5*k)},
        'PLA-304F': {'first': (1.9, 3.5*k), 'second': (1.45, 3.0*k)},
        'TMMU-DRF': {'first': (2.6, 2*k), 'second': (0.75, 2*k)},
    }
    if formula not in formulas:
        raise ValueError(f"Unknown formula: {formula}")
    a, b = formulas[formula][phase]
    return float(a) * float(weight) * float(tbsa) + float(b)


# --------- 构建模型（与训练一致）---------

def build_model(args) -> nn.Module:
    if args.model == 'base':
        return BaselineNetwork(args.input_dim_base, args.input_dim_temporal,
                               args.embed_dim, args.num_heads, args.hidden_dim,
                               args.num_classes, args.history_length, args.pred_length)
    if args.model == 'base_1':
        return BaselineNetwork_1(args.input_dim_base, args.input_dim_temporal,
                                 args.embed_dim, args.num_heads, args.hidden_dim,
                                 args.num_classes, args.history_length, args.pred_length)
    if args.model == 'cnn':
        return CNNBaselineNetwork(args.input_dim_base, args.input_dim_temporal,
                                  args.embed_dim, args.hidden_dim, args.num_classes,
                                  args.history_length, args.pred_length)
    if args.model == 'transformer':
        return TransformerBaselineNetwork(
            input_dim_base=args.input_dim_base,
            input_dim_temporal=args.input_dim_temporal,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            dim_ff=args.hidden_dim,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
            history_length=args.history_length,
            pred_length=args.pred_length,
            num_layers=getattr(args, "num_layers", 4),
            dropout=getattr(args, "dropout", 0.1),
            causal=getattr(args, "causal", True),
            use_film=getattr(args, "use_film", True)
        )
    if args.model == 'lstm_new':
        return LSTMBaselineNetwork(
            input_dim_base=args.input_dim_base,
            input_dim_temporal=args.input_dim_temporal,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
            history_length=args.history_length,
            pred_length=args.pred_length,
            rnn_hidden=getattr(args, "rnn_hidden", None),
            num_layers=getattr(args, "num_layers", 2),
            bidirectional=getattr(args, "bidirectional", True),
            dropout=getattr(args, "dropout", 0.1),
            use_film=getattr(args, "use_film", True)
        )
    if args.model == 'flim':
        return build_model_film_ziln(
            input_dim_base=args.input_dim_base,
            input_dim_temporal=args.input_dim_temporal,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
            pred_length=args.pred_length,
            out_dim_regression=getattr(args, "out_dim_regression", 3),
            num_temporal_layers=getattr(args, "num_temporal_layers", 2),
            dropout=getattr(args, "dropout", 0.1),
        )
    if args.model == 'tcn_film':
        return build_model_tcn_film_ziln(
            input_dim_base=args.input_dim_base,
            input_dim_temporal=args.input_dim_temporal,
            embed_dim=args.embed_dim,                 # 若内部不用也无妨
            num_heads=getattr(args, "num_heads", 1),  # 兜底
            hidden_dim=getattr(args, "hidden_dim", 256),
            num_classes=args.num_classes,
            pred_length=args.pred_length,
            dropout=getattr(args, "dropout", 0.1),
            tcn_k=getattr(args, "tcn_k", 3),
            tcn_layers_short=getattr(args, "tcn_layers_short", 3),
            tcn_layers_long=getattr(args, "tcn_layers_long", 5),
            causal=getattr(args, "causal_temporal", True),
        )
    raise ValueError(f"Invalid model: {args.model}")


def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device)
    # 兼容 DataParallel
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    from collections import OrderedDict
    new_state = OrderedDict()
    for k, v in state.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        new_state[new_k] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[INFO] Loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("  missing keys (truncated):", missing[:10])
    if unexpected:
        print("  unexpected keys (truncated):", unexpected[:10])
    model.to(device)
    model.eval()


def load_thresholds(thr_json: str, fallback=(0.5, 0.5, 0.5)) -> List[float]:
    if thr_json and os.path.exists(thr_json):
        with open(thr_json, "r", encoding="utf-8") as f:
            arr = json.load(f)
        arr = [float(x) for x in arr]
        if len(arr) >= 3:
            return arr[:3]
    return list(fallback)


# --------- 选择 test 病人 ----------

def choose_test_ids(Base_data: Dict, Ts_data: Dict, test_ids_json: str = None, test_ratio: float = 0.2) -> List[str]:
    if test_ids_json and os.path.exists(test_ids_json):
        with open(test_ids_json, "r", encoding="utf-8") as f:
            ids = [unify_id(x) for x in json.load(f)]
    else:
        base_ids = {unify_id(k) for k in Base_data.keys()}
        ts_ids   = {unify_id(k) for k in Ts_data.keys()}
        ids = sorted(base_ids & ts_ids)  # 交集，避免选到没有时序或没有基线的病人
    if not ids:
        return []
    n = max(1, int(round(len(ids) * test_ratio)))
    return ids[-n:]


# --------- 单个病人推理并返回逐时刻 DataFrame（不作图）---------

def infer_one_patient(
    pid: str,
    test_loader,
    model: nn.Module,
    device: torch.device,
    history_length: int,
    pred_length: int,
    thresholds: List[float],
    step_ml_scale: float = 1.0,
) -> pd.DataFrame:
    """
    基于 test_loader（仅该 pid 的所有滑窗样本）做推理。
    DataLoader 的每个 batch 形如:
      X: (B, H+P, 13)   时序特征窗口（示例）
      B: (B, Bb)        baseline
      Y: (B, H+P, >=6)  标签: 前3=types(0/1), 后3=speeds

    与训练一致：输入模型时拼接历史段  X[:, :-P, :]  和  Y[:, :-P, :6]  → (B, H, 13+6)
    ZILN 回归分支若给出 (p0_logit, mu, log_sigma) 则用期望值作为速度；再经 softplus 保证非负。
    阈值门控在最后一步（只对最后一步做累计/落表）。

    注意：t_index 是“窗口序号”（0..N-1）。如需绝对时间索引，请在 Dataset 添加并在此读取。
    """

    def _last_step(t: torch.Tensor) -> torch.Tensor:
        """兼容 2D/3D 张量，统一取最后时间步 (bs, C)。"""
        return t if t.dim() == 2 else t[:, -1, :]

    pid = str(pid).replace(".csv", "")
    thr = np.asarray(thresholds, dtype=np.float32).reshape(-1)

    rows = []
    win_idx = 0

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # 解包 batch
            if isinstance(batch, (tuple, list)) and len(batch) >= 3:
                X, B, Y = batch[0], batch[1], batch[2]
            else:
                raise RuntimeError("test_loader must yield (X, B, Y)")

            X = X.to(device)
            B = B.to(device)
            Y = Y.to(device)

            # 历史段拼接（与训练一致）：(bs, H, 13+6)
            X_hist = X[:, :-pred_length, :]
            Y_hist = Y[:, :-pred_length, :6]
            inp_T = torch.cat([X_hist, Y_hist], dim=-1)

            # 兼容 (x,b) 或只接 x 的模型
            try:
                outs = model(inp_T, B)
            except TypeError:
                outs = model(inp_T)

            # 解析输出
            flow_pred = None
            logits = None
            if isinstance(outs, dict):
                logits = outs.get("class_logits", None)
                flow_pred = outs.get("flow_pred", None)
                # ZILN 头 → 期望值当速度
                if flow_pred is None and all(k in outs for k in ("reg_mu", "reg_log_sigma", "p0_logit")):
                    mu = _last_step(outs["reg_mu"])                   # (bs, 3)
                    log_sigma = _last_step(outs["reg_log_sigma"]).clamp(-6, 6)
                    p0 = torch.sigmoid(_last_step(outs["p0_logit"]))  # (bs, 3)
                    sigma2 = torch.exp(2 * log_sigma)
                    flow_pred = (1 - p0) * torch.exp(mu + 0.5 * sigma2)  # (bs, 3)
                    flow_pred = flow_pred.unsqueeze(1)                   # (bs, 1, 3)
            elif isinstance(outs, (tuple, list)):
                if len(outs) >= 1: flow_pred = outs[0]
                if len(outs) >= 2: logits    = outs[1]
            else:
                logits = outs

            assert logits is not None, "model must output class_logits (or outs is logits)"

            probs = torch.sigmoid(logits)                                  # (bs, P, 3) 或 (bs, 3)
            probs_last = probs[:, -1, :] if probs.dim() == 3 else probs    # (bs, 3)

            if flow_pred is not None:
                speeds_last = F.softplus(flow_pred[:, -1, :]) if flow_pred.dim() == 3 else F.softplus(flow_pred)
            else:
                speeds_last = torch.zeros_like(probs_last)

            probs_np  = probs_last.detach().cpu().numpy()   # (bs, 3)
            speeds_np = speeds_last.detach().cpu().numpy()  # (bs, 3) —— raw 速度（未门控）

            # 逐样本落表（仅最后一步）
            Y_np = Y.detach().cpu().numpy()
            bs = probs_np.shape[0]
            for i in range(bs):
                mask = (probs_np[i] > thr).astype(np.float32)
                pred_speeds = speeds_np[i] * mask  # 门控后的速度

                gt_flags  = Y_np[i, -1, 0:3].astype(np.float32)
                gt_speeds = Y_np[i, -1, 3:6].astype(np.float32)

                rows.append([
                    pid, win_idx,
                    gt_flags[0], gt_flags[1], gt_flags[2],
                    gt_speeds[0], gt_speeds[1], gt_speeds[2],
                    probs_np[i, 0], probs_np[i, 1], probs_np[i, 2],
                    pred_speeds[0], pred_speeds[1], pred_speeds[2],
                    speeds_np[i, 0], speeds_np[i, 1], speeds_np[i, 2],  # raw 速度（softplus 后，未门控）
                ])
                win_idx += 1

    # 组装 DataFrame 与（初步）累计量
    cols = [
        "pid", "t_index",
        "gt_crystal", "gt_colloid", "gt_water",
        "gt_speed_crystal", "gt_speed_colloid", "gt_speed_water",
        "prob_crystal", "prob_colloid", "prob_water",
        "pred_speed_crystal", "pred_speed_colloid", "pred_speed_water",
        "pred_speed_raw_crystal", "pred_speed_raw_colloid", "pred_speed_raw_water",
    ]
    df = pd.DataFrame(rows, columns=cols)

    for cname in ["crystal", "colloid", "water"]:
        df[f"gt_cum_{cname}"]   = (df[f"gt_speed_{cname}"]   * step_ml_scale).cumsum()
        df[f"pred_cum_{cname}"] = (df[f"pred_speed_{cname}"] * step_ml_scale).cumsum()
    df["gt_cum_total"]   = df["gt_cum_crystal"] + df["gt_cum_colloid"] + df["gt_cum_water"]
    df["pred_cum_total"] = df["pred_cum_crystal"] + df["pred_cum_colloid"] + df["pred_cum_water"]

    # ===== 【新增】补齐完整时间轴，并将“首次补液”之前的预测清零 =====
    H = int(history_length)
    P = int(pred_length)
    N = len(df)
    # 推回原始总时长：当前 df 每行对应“滑窗最后一步”，其绝对时刻为 H..(T_full-P)
    T_full = N + H + P - 1  # 若数据是 72h，这里通常=72

    # 将 t_index 映射到绝对时刻 t_abs，并与 0..T_full-1 的完整索引对齐
    df["t_abs"] = df["t_index"] + H
    full = pd.DataFrame({"t_abs": np.arange(T_full, dtype=int)})
    df_full = full.merge(df, on="t_abs", how="left").sort_values("t_abs").reset_index(drop=True)

    # 缺失的 GT 列用前后向填充（便于判定“首次补液”）
    gt_cols = [
        "gt_crystal","gt_colloid","gt_water",
        "gt_speed_crystal","gt_speed_colloid","gt_speed_water",
    ]
    for c in gt_cols:
        if c in df_full.columns:
            df_full[c] = df_full[c].ffill().bfill()

    # 计算 GT 的瞬时体积
    gt_flag = np.stack([
        df_full["gt_crystal"].values,
        df_full["gt_colloid"].values,
        df_full["gt_water"].values
    ], axis=1)
    gt_speed = np.stack([
        df_full["gt_speed_crystal"].values,
        df_full["gt_speed_colloid"].values,
        df_full["gt_speed_water"].values
    ], axis=1)
    inst_gt = gt_flag * gt_speed  # (T_full, 3)

    # 找到首次任何一类补液>0 的绝对时刻；若全程无补液，则认为整段前期
    nonzero_any = (inst_gt > 0).any(axis=1)
    start_abs = int(np.argmax(nonzero_any)) if nonzero_any.any() else None

    # 需要处理的预测列（没有就补 NaN）
    pred_cols = [
        "prob_crystal","prob_colloid","prob_water",
        "pred_speed_crystal","pred_speed_colloid","pred_speed_water",
        "pred_speed_raw_crystal","pred_speed_raw_colloid","pred_speed_raw_water",
    ]
    for c in pred_cols:
        if c not in df_full.columns:
            df_full[c] = np.nan

    # 在首次补液之前（t_abs < start_abs），将所有预测清零；若全程无补液，则整段清零
    if start_abs is None:
        mask_early = np.ones(len(df_full), dtype=bool)
    else:
        mask_early = df_full["t_abs"].values < start_abs

    for c in pred_cols:
        arr = df_full[c].astype(float).fillna(0.0).values
        arr[mask_early] = 0.0
        df_full[c] = arr

    # 重新计算累计（pred/gt），把 NaN 当 0
    for cname in ["crystal", "colloid", "water"]:
        g_inst = (df_full[f"gt_speed_{cname}"].fillna(0.0).values) * float(step_ml_scale)
        p_inst = (df_full[f"pred_speed_{cname}"].fillna(0.0).values) * float(step_ml_scale)
        df_full[f"gt_cum_{cname}"]   = np.cumsum(g_inst)
        df_full[f"pred_cum_{cname}"] = np.cumsum(p_inst)

    df_full["gt_cum_total"]   = df_full["gt_cum_crystal"] + df_full["gt_cum_colloid"] + df_full["gt_cum_water"]
    df_full["pred_cum_total"] = df_full["pred_cum_crystal"] + df_full["pred_cum_colloid"] + df_full["pred_cum_water"]

    # t_index 统一成 0..T_full-1，去掉临时列
    df_full["t_index"] = np.arange(T_full, dtype=int)
    df_full = df_full.drop(columns=["t_abs"])

    # 返回补齐且清零后的结果
    return df_full


# --------- 画图（单独 mode 使用）---------

def plot_from_saved_csv(
    timeseries_csv: str,
    out_dir: str,
    pid: str,
    formula: str = "TMMU-DRF",
    formula_phase: str = "first",
    tbsa: float = None,
    weight: float = None,
    thresholds=None,               # float 或 长度为3的list/tuple；None→自动加载 used_thresholds.json 或 0.5
    step_ml_scale: float = 1.0,    # 累计时的步长缩放（与 infer 时保持一致）
    save_pdf: bool = False,
):
    df = pd.read_csv(timeseries_csv)

    # ---- 1) 解析阈值 ----
    if thresholds is None:
        used_thr = os.path.join(os.path.dirname(timeseries_csv), "used_thresholds.json")
        if os.path.exists(used_thr):
            with open(used_thr, "r", encoding="utf-8") as f:
                thresholds = json.load(f)
        else:
            print("[WARN] thresholds not given and used_thresholds.json not found; using 0.5")
            thresholds = [0.5, 0.5, 0.5]
    if not isinstance(thresholds, (list, tuple, np.ndarray)):
        thresholds = [float(thresholds)] * 3
    thr = np.asarray(thresholds, dtype=float).reshape(1, 3)  # 方便广播

    # ---- 2) 取基础列 ----
    x = df["t_index"].values

    gt_flag = np.stack([df["gt_crystal"].values,
                        df["gt_colloid"].values,
                        df["gt_water"].values], axis=1)                # (N,3)

    gt_speed = np.stack([df["gt_speed_crystal"].values,
                         df["gt_speed_colloid"].values,
                         df["gt_speed_water"].values], axis=1)        # (N,3)

    prob = np.stack([df["prob_crystal"].values,
                     df["prob_colloid"].values,
                     df["prob_water"].values], axis=1)                # (N,3)

    # 预测速度优先用 raw 列；无则回退用 pred_speed_*（已被旧阈值门控过）
    def pick_col(df_, prefer, fallback):
        return df_[prefer].values if prefer in df_.columns else df_[fallback].values

    pred_speed_cr = pick_col(df, "pred_speed_raw_crystal", "pred_speed_crystal")
    pred_speed_co = pick_col(df, "pred_speed_raw_colloid", "pred_speed_colloid")
    pred_speed_wa = pick_col(df, "pred_speed_raw_water",   "pred_speed_water")

    pred_speed = np.stack([pred_speed_cr, pred_speed_co, pred_speed_wa], axis=1)  # (N,3)

    # ---- 3) 计算“存在性 × 速度”的瞬时量 ----
    inst_gt   = gt_flag * gt_speed                                  # (N,3)
    pred_mask = (prob > thr).astype(float)                           # (N,3)
    inst_pred = pred_speed * pred_mask                               # (N,3)

    # ---- 4) 按步长缩放后累计 ----
    gt_cum   = (inst_gt   * float(step_ml_scale)).cumsum(axis=0)     # (N,3)
    pred_cum = (inst_pred * float(step_ml_scale)).cumsum(axis=0)     # (N,3)
    gt_total   = gt_cum.sum(axis=1)                                  # (N,)
    pred_total = pred_cum.sum(axis=1)                                # (N,)

    # ---- 5) 作图：总量累计曲线 ----
    plt.figure(figsize=(8, 5))
    plt.plot(x, gt_total,  label="GT Total")
    plt.plot(x, pred_total, label="Pred Total")
    plt.xlabel("Time Index")
    plt.ylabel("Cumulative Volume (mL)")
    plt.title(f"PID {pid} - Cumulative Total (GT vs Pred)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"{pid}_cum_total.png")
    plt.savefig(out_png, dpi=150)
    if save_pdf:
        plt.savefig(os.path.join(out_dir, f"{pid}_cum_total.pdf"))
    plt.close()

    # ---- 6) 作图：三类别累计曲线 ----
    cname = ["crystal", "colloid", "water"]
    plt.figure(figsize=(10, 6))
    for i, c in enumerate(cname):
        plt.plot(x, gt_cum[:, i],   label=f"GT {c}")
        plt.plot(x, pred_cum[:, i], linestyle="--", label=f"Pred {c}")
    plt.xlabel("Time Index")
    plt.ylabel("Cumulative Volume (mL)")
    plt.title(f"PID {pid} - Cumulative by Class")
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"{pid}_cum_by_class.png")
    plt.savefig(out_png, dpi=150)
    if save_pdf:
        plt.savefig(os.path.join(out_dir, f"{pid}_cum_by_class.pdf"))
    plt.close()

    # ---- 7) 作图：总量柱状对比（可加公式） ----
    actual_total = float(gt_total[-1])  if len(gt_total)  else 0.0
    pred_total_  = float(pred_total[-1]) if len(pred_total) else 0.0
    labels = ["Actual", "Pred"]
    vals   = [actual_total, pred_total_]

    if tbsa is not None and weight is not None:
        try:
            formula_total = calculate_formula_fluid(formula, float(tbsa), float(weight), formula_phase)
            labels.append("Formula"); vals.append(formula_total)
        except Exception:
            pass

    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, vals)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h, f"{h:.0f}", ha="center", va="bottom", fontsize=10)
    plt.ylabel("Total Volume (mL)")
    plt.title(f"PID {pid} - Total Comparison")
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"{pid}_total_bar.png")
    plt.savefig(out_png, dpi=150)
    if save_pdf:
        plt.savefig(os.path.join(out_dir, f"{pid}_total_bar.pdf"))
    plt.close()


# =============================
# I/O & 主流程
# =============================

def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser("Per-patient inference & plotting (integrated)")
    parser.add_argument("--mode", type=str, default="infer", choices=["infer", "plot"],
                        help="infer: 逐病人推理保存 CSV；plot: 基于 CSV 出图")

    # 数据
    parser.add_argument("--data_pkl", type=str, default="data/output_data.pkl")
    parser.add_argument("--base_pkl", type=str, default="data/baseline.pkl")

    # 输出
    parser.add_argument("--output_dir", type=str, default="./per_patient_outputs")

    # 与训练一致的关键参数
    parser.add_argument("--model", type=str, default="tcn_film",
                        choices=['base', 'transformer', 'lstm_new', 'cnn', 'base_1', 'flim','tcn_film'])
    parser.add_argument("--history_length", type=int, default=4)
    parser.add_argument("--pred_length", type=int, default=1)
    parser.add_argument("--input_dim_base", type=int, default=10)
    parser.add_argument("--input_dim_temporal", type=int, default=19)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=3)

    # 设备/权重/阈值
    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        default_dev = "cuda:0"
    else:
        default_dev = "cpu"
    parser.add_argument("--device", type=str, default=default_dev,
                        help="e.g., cuda:0 / cpu")
    parser.add_argument("--ckpt", type=str, default="./ckpts/tcn_film_e33_loss12.2983_20250928_141551.pth")
    parser.add_argument("--thresholds_json", type=str, default="./ckpts/best_thresholds.json")

    # 选择 test 病人
    parser.add_argument("--test_ids_json", type=str, default=None, help="可选，显式指定测试病人列表 JSON")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="若未指定 JSON，则取 Ts∩Base 的后 20%")

    # 累计步长换算（每步对应分钟/小时比例）
    parser.add_argument("--step_ml_scale", type=float, default=1.0,
                        help="每时间步的毫升换算系数：若速度单位为 mL/h 且每步=Δ分钟，应设 Δ/60")

    # 仅 plot 模式会用到：公式设置
    parser.add_argument("--formula", type=str, default="TMMU-DRF")
    parser.add_argument("--formula_phase", type=str, default="first", choices=["first", "second"])
    parser.add_argument("--save_pdf", action="store_true", help="额外导出 PDF 图（论文友好）")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "infer":
        # ---------- 阶段1：推理并保存 ----------
        Ts_data = load_pickle(args.data_pkl)
        Base_data = load_pickle(args.base_pkl)
        # 统一 ID
        Ts_data, Base_data = normalize_keys_inplace(Ts_data, Base_data)
        print(f"[INFO] Loaded Ts_data keys: {list(Ts_data.keys())[:5]}{'...' if len(Ts_data)>5 else ''}")
        print(f"[INFO] Loaded Base_data keys: {list(Base_data.keys())[:5]}{'...' if len(Base_data)>5 else ''}")
        assert isinstance(Ts_data, dict) and len(Ts_data) > 0, "Ts_data should be a non-empty dict"

        device = torch.device(args.device)
        model = build_model(args)
        load_checkpoint(model, args.ckpt, device)

        thresholds = load_thresholds(args.thresholds_json, fallback=(0.5, 0.5, 0.5))
        with open(os.path.join(args.output_dir, "used_thresholds.json"), "w", encoding="utf-8") as f:
            json.dump(list(map(float, thresholds)), f)

        # 选择测试病人（交集）
        test_ids = choose_test_ids(Base_data, Ts_data, args.test_ids_json, args.test_ratio)
        print(f"[INFO] Test patients ({len(test_ids)}): {test_ids[:8]}{'...' if len(test_ids)>8 else ''}")

        summary_rows = []
        for pid in test_ids:
            try:
                Ts_data_subset = {pid: Ts_data[pid]}
                Base_data_subset = {pid: Base_data[pid]}

                # 全部进 test，train/val 为空会返回 None（见 data_loader 内处理）
                _, _, test_loader = data_loader(
                    Ts_data_subset, Base_data_subset,
                    history_length=args.history_length,
                    pred_length=args.pred_length,
                    classes=args.num_classes,
                    batch_size=getattr(args, "batch_size", 32),
                    split_mode="patient",
                    splits=(0.0, 0.0, 1.0)
                )

                # 空 loader 直接跳过
                if (test_loader is None) or (hasattr(test_loader, "dataset") and len(test_loader.dataset) == 0):
                    print(f"[WARN] {pid}: no windows generated (T < H+P? baseline missing/NaN?). Skipped.")
                    continue

                df = infer_one_patient(
                    pid, test_loader, model,
                    device=device,
                    history_length=args.history_length,
                    pred_length=args.pred_length,
                    thresholds=thresholds,
                    step_ml_scale=args.step_ml_scale,
                )

                out_csv = os.path.join(args.output_dir, f"{pid}_timeseries.csv")
                df.to_csv(out_csv, index=False)
                print(f"[INFO] Saved {out_csv}")

                # 汇总（末时刻累计）
                actual_total = float(df["gt_cum_total"].iloc[-1]) if len(df) else 0.0
                pred_total   = float(df["pred_cum_total"].iloc[-1]) if len(df) else 0.0

                # baseline 中可能含 tbsa/weight
                tbsa = None
                weight = None
                base = Base_data.get(pid, None)
                if isinstance(base, dict):
                    tbsa = base.get("tbsa", None)
                    weight = base.get("weight", None)

                summary_rows.append({
                    "pid": pid,
                    "actual_total_ml": actual_total,
                    "pred_total_ml": pred_total,
                    "tbsa": None if tbsa is None else float(tbsa),
                    "weight": None if weight is None else float(weight),
                })

            except Exception as e:
                print(f"[ERR] {pid}: {e}")
                continue

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(args.output_dir, "patients_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        if summary_rows:
            print(f"[DONE] Saved per-patient timeseries & summary to: {args.output_dir}")
        else:
            print("[WARN] No patients produced outputs. Check baseline keys / window lengths / thresholds.")

    else:
        def _normalize_pid(v):
            # 转成字符串后做清洗：去空格、去.csv、去尾部的 .0 / .000...
            s = str(v).strip()
            s = s.replace(".csv", "")
            # 若形如 7.000e+09 或 7000...0.0
            try:
                f = float(s)
                if f.is_integer():
                    return str(int(f))
            except Exception:
                pass
            s = re.sub(r"\.0+$", "", s)
            return s

        # ---------- 阶段2：根据保存的 CSV 再画图 ----------
        summary_csv = os.path.join(args.output_dir, "patients_summary.csv")
        if not os.path.exists(summary_csv):
            raise FileNotFoundError(f"{summary_csv} not found. Please run with --mode infer first.")
        summary = pd.read_csv(summary_csv)

        for _, row in summary.iterrows():
            pid = _normalize_pid(row["pid"]) if "pid" in row else _normalize_pid(row[0])

            timeseries_csv = os.path.join(args.output_dir, f"{pid}_timeseries.csv")
            if not os.path.exists(timeseries_csv):
                print(f"[WARN] {timeseries_csv} not found, skip plotting for {pid}.")
                continue

            tbsa = row["tbsa"] if "tbsa" in row and pd.notna(row["tbsa"]) else None
            weight = row["weight"] if "weight" in row and pd.notna(row["weight"]) else None

            plot_from_saved_csv(
                timeseries_csv=timeseries_csv,
                out_dir=args.output_dir,
                pid=pid,
                formula=args.formula,
                formula_phase=args.formula_phase,
                tbsa=tbsa,
                weight=weight,
                thresholds=[0.45, 0.45, 0.45],            # 若想覆盖阈值，可传入 [t1,t2,t3]
                step_ml_scale=args.step_ml_scale,
                save_pdf=args.save_pdf,
            )
        print(f"[DONE] Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
