#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fluid Resuscitation Inference + Visualization (KEEP PLOT, MINIMIZE OTHERS)

- infer: 极简推理（输出 plot 所需字段）
- plot : 保留你原来的高质量 2x2 可视化逻辑（几乎不改）

输出文件（与原 plot 兼容）:
  {output_dir}/{pid}_timeseries.csv
  {output_dir}/{pid}_summary.csv
  {output_dir}/patients_summary.csv
  {output_dir}/used_thresholds.json
"""

import re
import json
import argparse
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ===== 模型 & 数据加载（保持你项目结构不变）=====
from utils.data_loader import *
from models.Baseline import BaselineNetwork, BaselineNetwork_1
from models.cnn import CNNBaselineNetwork
from models.transformer import TransformerBaselineNetwork
from models.Lstm import LSTMBaselineNetwork
from models.Flim import build_model_film_ziln
from models.tcn_film import build_model_tcn_film_ziln


# ============================================================
#                 MINIMAL INFERENCE (简化部分)
# ============================================================

def norm_pid(x) -> str:
    x = str(x)
    if x.endswith(".csv"):
        x = x[:-4]
    x = re.sub(r"\.0+$", "", x).strip()
    try:
        f = float(x)
        if f.is_integer():
            x = str(int(f))
    except Exception:
        pass
    return x

def norm_keys(d: Dict) -> Dict:
    return {norm_pid(k): v for k, v in d.items()}

def load_thresholds(path: Optional[str], default=(0.5, 0.5, 0.5)) -> List[float]:
    if path and Path(path).exists():
        try:
            t = json.load(open(path, "r", encoding="utf-8"))
            t = [float(x) for x in t[:3]]
            if len(t) == 3:
                return t
        except Exception:
            pass
    return list(default)

def build_model(name: str, cfg: Dict) -> torch.nn.Module:
    if name == "base":
        return BaselineNetwork(cfg["input_dim_base"], cfg["input_dim_temporal"],
                               cfg["embed_dim"], cfg["num_heads"], cfg["hidden_dim"],
                               cfg["num_classes"], cfg["history_length"], cfg["pred_length"])
    if name == "base_1":
        return BaselineNetwork_1(cfg["input_dim_base"], cfg["input_dim_temporal"],
                                 cfg["embed_dim"], cfg["num_heads"], cfg["hidden_dim"],
                                 cfg["num_classes"], cfg["history_length"], cfg["pred_length"])
    if name == "cnn":
        return CNNBaselineNetwork(cfg["input_dim_base"], cfg["input_dim_temporal"],
                                  cfg["embed_dim"], cfg["hidden_dim"], cfg["num_classes"],
                                  cfg["history_length"], cfg["pred_length"])
    if name == "transformer":
        return TransformerBaselineNetwork(
            input_dim_base=cfg["input_dim_base"],
            input_dim_temporal=cfg["input_dim_temporal"],
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            dim_ff=cfg["hidden_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_classes=cfg["num_classes"],
            history_length=cfg["history_length"],
            pred_length=cfg["pred_length"],
            num_layers=cfg.get("num_layers", 4),
            dropout=cfg.get("dropout", 0.1),
            causal=cfg.get("causal", True),
            use_film=cfg.get("use_film", True),
        )
    if name == "lstm_new":
        return LSTMBaselineNetwork(
            input_dim_base=cfg["input_dim_base"],
            input_dim_temporal=cfg["input_dim_temporal"],
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_classes=cfg["num_classes"],
            history_length=cfg["history_length"],
            pred_length=cfg["pred_length"],
            num_layers=cfg.get("num_layers", 2),
            bidirectional=cfg.get("bidirectional", True),
            dropout=cfg.get("dropout", 0.1),
            use_film=cfg.get("use_film", True),
        )
    if name == "flim":
        return build_model_film_ziln(
            input_dim_base=cfg["input_dim_base"],
            input_dim_temporal=cfg["input_dim_temporal"],
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            hidden_dim=cfg["hidden_dim"],
            num_classes=cfg["num_classes"],
            pred_length=cfg["pred_length"],
            out_dim_regression=cfg.get("out_dim_regression", 3),
            num_temporal_layers=cfg.get("num_temporal_layers", 2),
            dropout=cfg.get("dropout", 0.1),
        )
    if name == "tcn_film":
        return build_model_tcn_film_ziln(
            input_dim_base=cfg["input_dim_base"],
            input_dim_temporal=cfg["input_dim_temporal"],
            embed_dim=cfg["embed_dim"],
            num_heads=cfg.get("num_heads", 1),
            hidden_dim=cfg.get("hidden_dim", 256),
            num_classes=cfg["num_classes"],
            pred_length=cfg["pred_length"],
            dropout=cfg.get("dropout", 0.1),
            tcn_k=cfg.get("tcn_k", 3),
            tcn_layers_short=cfg.get("tcn_layers_short", 3),
            tcn_layers_long=cfg.get("tcn_layers_long", 5),
            causal=cfg.get("causal_temporal", True),
        )
    raise ValueError(f"unknown model: {name}")

def load_ckpt(model: torch.nn.Module, ckpt: str, device: torch.device) -> torch.nn.Module:
    s = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(s, dict) and "state_dict" in s:
        s = s["state_dict"]
    s = {k.replace("module.", ""): v for k, v in s.items()}
    model.load_state_dict(s, strict=False)
    return model.to(device).eval()

def infer_patient(
    pid: str,
    ts_data: Dict,
    base_data: Dict,
    model: torch.nn.Module,
    device: torch.device,
    thresholds: List[float],
    history_length: int,
    pred_length: int,
    num_classes: int,
    batch_size: int,
    step_ml_scale: float
):
    history_length = 1
    if pid not in ts_data or pid not in base_data:
        return None, None, None, None
    _, _, loader = data_loader(
        {pid: ts_data[pid]},
        {pid: base_data[pid]},
        history_length=history_length,
        pred_length=pred_length,
        classes=num_classes,
        batch_size=1,
        split_mode="patient",
        splits=(0.0, 0.0, 1.0),
    )
    if loader is None or len(loader.dataset) == 0:
        return None, None, None, None

    weight = max(float(base_data[pid][3]), 35.0)
    tbsa = float(base_data[pid][5]) if base_data[pid][5] is not None else 1.0
    thr = np.asarray(thresholds, dtype=np.float32)

    rows = []
    for i in range(history_length):
        rows.append([
            pid, i,
            0, 0, 0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ])
    for bidx, batch in enumerate(loader):
        X, B, Y = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        with torch.no_grad():
            X_hist = X[:, :-pred_length, :]
            Y_hist = Y[:, :-pred_length, :6]
            inp_T = torch.cat([X_hist, Y_hist], dim=-1)

            try:
                outs = model(inp_T, B)
            except TypeError:
                outs = model(inp_T)

            logits = outs.get("class_logits")
            flow = outs.get("flow_pred")

            probs = torch.sigmoid(logits)
            probs = probs[:, -1, :] if probs.dim() == 3 else probs  # (bs,3)

            speeds = torch.zeros_like(probs)
            if flow is not None:
                speeds = F.softplus(flow[:, -1, :]) if flow.dim() == 3 else F.softplus(flow)

        probs_np = probs.cpu().numpy()
        speeds_np = speeds.cpu().numpy()
        Y_np = Y.cpu().numpy()

        for i in range(len(probs_np)):
            mask = (probs_np[i] > thr).astype(np.float32)
            gated = speeds_np[i] * mask
            gt_flags = Y_np[i, -1, 0:3].astype(np.float32)
            gt_speed = Y_np[i, -1, 3:6].astype(np.float32)

            t_index = bidx * len(probs_np) + i + history_length
            rows.append([
                pid, t_index,
                *gt_flags, *gt_speed,
                *probs_np[i].tolist(),
                *gated.tolist(),
                *speeds_np[i].tolist()
            ])

    df = pd.DataFrame(rows, columns=[
        "pid", "t_index",
        "gt_crystal", "gt_colloid", "gt_water",
        "gt_speed_crystal", "gt_speed_colloid", "gt_speed_water",
        "prob_crystal", "prob_colloid", "prob_water",
        "pred_speed_crystal", "pred_speed_colloid", "pred_speed_water",
        "pred_speed_raw_crystal", "pred_speed_raw_colloid", "pred_speed_raw_water"
    ])

    # 首次真实输液前清零（plot 依赖 prob/raw）
    first = np.where(df[["gt_crystal", "gt_colloid", "gt_water"]].sum(axis=1).to_numpy() > 0)[0]
    if len(first):
        idx0 = first[0]
        zero_cols = [
            "pred_speed_crystal", "pred_speed_colloid", "pred_speed_water",
            "pred_speed_raw_crystal", "pred_speed_raw_colloid", "pred_speed_raw_water",
            "prob_crystal", "prob_colloid", "prob_water"
        ]
        df.loc[df.index < idx0, zero_cols] = 0.0

    # plot 依赖的累积字段
    for f in ["crystal", "colloid", "water"]:
        df[f"gt_cum_{f}"] = np.cumsum(df[f"gt_speed_{f}"].to_numpy() * step_ml_scale)
        df[f"pred_cum_{f}"] = np.cumsum(df[f"pred_speed_raw_{f}"].to_numpy() * step_ml_scale)

    df["gt_cum_total"] = df["gt_cum_crystal"] + df["gt_cum_colloid"] + df["gt_cum_water"]
    df["pred_cum_total"] = df["pred_cum_crystal"] + df["pred_cum_colloid"] + df["pred_cum_water"]

    # 简单 summary（你 plot 会从 patients_summary.csv 读 tbsa/weight）
    summary = pd.DataFrame([{
        "pid": pid,
        "weight": float(weight),
        "tbsa": float(tbsa),
        "final_gt_total_ml": float(df["gt_cum_total"].iloc[-1]),
        "final_pred_total_ml": float(df["pred_cum_total"].iloc[-1]),
    }])

    return df, summary, weight, tbsa

def run_inference(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    thresholds = load_thresholds(args.thresholds_json)
    json.dump(thresholds, open(out / "used_thresholds.json", "w", encoding="utf-8"), indent=2)

    ts_data = norm_keys(pickle.load(open(args.data_pkl, "rb")))
    base_data = norm_keys(pickle.load(open(args.base_pkl, "rb")))

    if args.test_ids_json and Path(args.test_ids_json).exists():
        test_ids = [norm_pid(x) for x in json.load(open(args.test_ids_json, "r", encoding="utf-8"))]
    else:
        common = sorted(set(ts_data) & set(base_data))
        n_test = max(1, int(len(common) * args.test_ratio))
        test_ids = common[-n_test:]

    cfg = dict(
        input_dim_base=args.input_dim_base,
        input_dim_temporal=args.input_dim_temporal,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        history_length=args.history_length,
        pred_length=args.pred_length,
    )
    model = load_ckpt(build_model(args.model, cfg), args.ckpt, device)

    summary_rows = []
    for pid in test_ids:
        df, stats_df, weight, tbsa = infer_patient(
            pid, ts_data, base_data, model, device,
            thresholds, args.history_length, args.pred_length,
            args.num_classes, args.batch_size, args.step_ml_scale
        )
        if df is None:
            continue

        df.to_csv(out / f"{pid}_timeseries.csv", index=False)
        stats_df.to_csv(out / f"{pid}_summary.csv", index=False)

        summary_rows.append({
            "pid": pid,
            "actual_total_ml": float(df["gt_cum_total"].iloc[-1]),
            "pred_total_ml": float(df["pred_cum_total"].iloc[-1]),
            "tbsa": float(tbsa),
            "weight": float(weight),
        })

    pd.DataFrame(summary_rows).to_csv(out / "patients_summary.csv", index=False)
    print(f"[infer] done -> {out}")


# ============================================================
#                 PLOT (保留你原画图逻辑)
# ============================================================

@dataclass
class PlotConfig:
    formula: str = "TMMU-DRF"
    formula_phase: str = "first"
    save_pdf: bool = False
    plot_dpi: int = 300
    colors: Dict[str, str] = None

    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'gt': '#2E4053',
                'pred_orig': '#E74C3C',
                'pred_gated': '#3498DB',
                'formula': '#27AE60',
                'background': '#F8F9F9'
            }

class ClinicalFormulaCalculator:
    FORMULA_COEFFICIENTS = {
        "Evans": {"first": (2.0, 2000.0), "second": (1.0, 2000.0)},
        "Brooke": {"first": (2.0, 2000.0), "second": (1.0, 2000.0)},
        "Parkland": {"first": (4.0, 0.0), "second": (0.0, 2000.0)},
        "Monafo": {"first": (2.0, 0.0), "second": (1.0, 0.0)},
        "TMMU": {"first": (1.5, 2000.0), "second": (0.75, 2000.0)},
        "RJH": {"first": (1.5, 3500.0), "second": (0.75, 3500.0)},
        "PLA-304F": {"first": (1.9, 3500.0), "second": (1.45, 3000.0)},
        "TMMU-DRF": {"first": (2.6, 2000.0), "second": (0.75, 2000.0)},
    }

    def calculate(self, formula: str, tbsa: float, weight: float, phase: str = "first") -> float:
        if tbsa <= 0 or weight <= 0:
            return 0.0
        a, b = self.FORMULA_COEFFICIENTS[formula][phase]
        return a * weight * tbsa + b


class VisualizationPipeline:
    def __init__(self, config: PlotConfig, output_dir: str, step_ml_scale: float = 1.0, thresholds: List[float] = None):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.formula_calculator = ClinicalFormulaCalculator()
        self.step_ml_scale = step_ml_scale
        self.thresholds = thresholds if thresholds is not None else [0.5, 0.5, 0.5]

        self.fluid_types = ['crystal', 'colloid', 'water']
        self.fluid_names = ['Crystalloid', 'Colloid', 'Water']
        self.fluid_colors = {
            'crystal': '#3498DB',
            'colloid': '#E74C3C',
            'water': '#2ECC71',
            'total': '#9B59B6',
            'gt': '#2C3E50',
            'pred': '#F39C12',
            'formula': '#27AE60',
        }
        self._setup_plot_style()

    def _setup_plot_style(self):
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'legend.fontsize': 9,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'figure.dpi': self.config.plot_dpi,
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.linewidth': 0.8,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'grid.linestyle': '-',
            'grid.color': '#CCCCCC',
            'xtick.bottom': True,
            'ytick.left': True,
            'xtick.color': 'black',
            'ytick.color': 'black',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True,

            # ✅ 仍然使用 constrained_layout
            'figure.constrained_layout.use': True,
            'figure.constrained_layout.h_pad': 0.1,
            'figure.constrained_layout.w_pad': 0.1,
            'figure.constrained_layout.hspace': 0.1,
            'figure.constrained_layout.wspace': 0.1,

            'lines.linewidth': 2.0,
            'lines.markersize': 6,
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'legend.edgecolor': 'black',
        })

    def _prepare_data_for_plotting(self, df: pd.DataFrame) -> Dict:
        t = df["t_index"].to_numpy().astype(np.float64)

        thr = np.asarray(self.thresholds, dtype=np.float64).reshape(1, -1)
        prob = df[["prob_crystal", "prob_colloid", "prob_water"]].to_numpy().astype(np.float64)
        raw = df[["pred_speed_raw_crystal", "pred_speed_raw_colloid", "pred_speed_raw_water"]].to_numpy().astype(np.float64)
        mask = (prob > thr).astype(np.float64)
        gated = raw * mask
        print(raw,gated)
        pred_cum_regated = {}
        for i, fluid in enumerate(self.fluid_types):
            pred_cum_regated[fluid] = np.cumsum(gated[:, i] * self.step_ml_scale)
        pred_cum_regated['total'] = sum(pred_cum_regated.values())

        cum_data = {'gt': {}, 'pred_orig': {}, 'pred_gated': {}}
        for fluid in self.fluid_types:
            cum_data['gt'][fluid] = df[f"gt_cum_{fluid}"].to_numpy().astype(np.float64)
            cum_data['pred_orig'][fluid] = df[f"pred_cum_{fluid}"].to_numpy().astype(np.float64)
            cum_data['pred_gated'][fluid] = pred_cum_regated[fluid]

        cum_data['gt']['total'] = df["gt_cum_total"].to_numpy().astype(np.float64)
        cum_data['pred_orig']['total'] = df["pred_cum_total"].to_numpy().astype(np.float64)
        cum_data['pred_gated']['total'] = pred_cum_regated['total']

        if len(t) > 1:
            speed_data = {'gt': {}, 'pred_gated': {}}
            for i, fluid in enumerate(self.fluid_types):
                speed_data['gt'][fluid] = df[f"gt_speed_{fluid}"].to_numpy().astype(np.float64)
                speed_data['pred_gated'][fluid] = gated[:, i]
            speed_data['gt']['total'] = speed_data['gt']['crystal'] + speed_data['gt']['colloid'] + speed_data['gt']['water']
            speed_data['pred_gated']['total'] = speed_data['pred_gated']['crystal'] + speed_data['pred_gated']['colloid'] + speed_data['pred_gated']['water']
        else:
            speed_data = None

        return {'time': t, 'cum_data': cum_data, 'speed_data': speed_data, 'pred_cum_regated': pred_cum_regated}

    def _create_comprehensive_plot(self, df: pd.DataFrame, pid: str, formula_total: Optional[float] = None) -> Figure:
        data = self._prepare_data_for_plotting(df)
        t = data['time']
        cum_data = data['cum_data']
        speed_data = data['speed_data']

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        ax1, ax2, ax3, ax4 = axes.flatten()

        patient_info = f"Patient {pid}"
        if formula_total is not None:
            patient_info += f" | {self.config.formula} Formula: {formula_total:,.0f} mL"

        # ✅ suptitle 稍微上移，避免 constrained_layout 挤压
        fig.suptitle(patient_info, fontsize=14, fontweight='bold', y=1.02)

        # 1) 三液体累积
        for i, fluid in enumerate(self.fluid_types):
            ax1.plot(t, cum_data['gt'][fluid],
                     label=f'GT {self.fluid_names[i]}',
                     color=self.fluid_colors[fluid],
                     linewidth=2.5, alpha=0.9, zorder=3)
            ax1.plot(t, cum_data['pred_gated'][fluid],
                     label=f'Pred {self.fluid_names[i]}',
                     color=self.fluid_colors[fluid],
                     linewidth=2.0, linestyle='--', alpha=0.9, zorder=2)

        ax1.set_xlabel('Time Index (hours)')
        ax1.set_ylabel('Cumulative Volume (mL)')
        ax1.set_title('Individual Fluids: Cumulative Volume', fontweight='bold')
        ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax1.legend(loc='upper left', fontsize=9, frameon=True, framealpha=0.95, edgecolor='black', fancybox=True)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

        # 2) 总累积
        ax2.plot(t, cum_data['gt']['total'], label='Ground Truth',
                 color=self.fluid_colors['gt'], linewidth=3.0, alpha=0.9, zorder=3)
        ax2.plot(t, cum_data['pred_orig']['total'], label='Model (Original)',
                 color=self.fluid_colors['pred'], linewidth=2.0, linestyle=':', alpha=0.7, zorder=2)
        ax2.plot(t, cum_data['pred_gated']['total'], label='Model (Threshold-gated)',
                 color=self.fluid_colors['pred'], linewidth=2.5, linestyle='--', alpha=0.9, zorder=2)

        if formula_total is not None:
            ax2.axhline(formula_total, linestyle='-.',
                        color=self.fluid_colors['formula'],
                        linewidth=2.0, alpha=0.8,
                        label=f'{self.config.formula} Formula', zorder=1)

        ax2.set_xlabel('Time Index (hours)')
        ax2.set_ylabel('Cumulative Volume (mL)')
        ax2.set_title('Total: Cumulative Volume', fontweight='bold')
        ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax2.legend(loc='upper left', fontsize=9, frameon=True, framealpha=0.95, edgecolor='black', fancybox=True)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

        # 3) 三液体流速堆叠
        if speed_data is not None:
            display_step = max(1, len(t) // 100)
            t_display = t[::display_step]

            gt_speed_display = {}
            pred_speed_display = {}
            for fluid in self.fluid_types:
                gt_speed_display[fluid] = speed_data['gt'][fluid][::display_step].astype(np.float64)
                pred_speed_display[fluid] = speed_data['pred_gated'][fluid][::display_step].astype(np.float64)

            bottom = np.zeros_like(t_display, dtype=np.float64)
            for i, fluid in enumerate(self.fluid_types):
                ax3.fill_between(t_display, bottom, bottom + gt_speed_display[fluid],
                                 alpha=0.6, color=self.fluid_colors[fluid],
                                 label=f'GT {self.fluid_names[i]}',
                                 edgecolor='white', linewidth=0.5)
                bottom += gt_speed_display[fluid]

            bottom = np.zeros_like(t_display, dtype=np.float64)
            for i, fluid in enumerate(self.fluid_types):
                ax3.fill_between(t_display, bottom, bottom + pred_speed_display[fluid],
                                 alpha=0.3, color=self.fluid_colors[fluid],
                                 hatch='//', edgecolor=self.fluid_colors[fluid], linewidth=0.5,
                                 label=f'Pred {self.fluid_names[i]}')
                bottom += pred_speed_display[fluid]

        ax3.set_xlabel('Time Index (hours)')
        ax3.set_ylabel('Flow Rate (mL/hour)')
        ax3.set_title('Individual Fluids: Flow Rate (Stacked)', fontweight='bold')
        ax3.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax3.legend(loc='upper right', fontsize=9, frameon=True, framealpha=0.95, edgecolor='black', fancybox=True)
        ax3.set_ylim(bottom=0)

        # 4) 总流速
        if speed_data is not None:
            gt_total_speed = speed_data['gt']['total'][::display_step].astype(np.float64)
            pred_total_speed = speed_data['pred_gated']['total'][::display_step].astype(np.float64)

            ax4.plot(t_display, gt_total_speed, label='Ground Truth',
                     color=self.fluid_colors['gt'], linewidth=2.5, alpha=0.9,
                     marker='o', markersize=4,
                     markevery=max(1, len(t_display)//20), zorder=3)

            ax4.plot(t_display, pred_total_speed, label='Model (Threshold-gated)',
                     color=self.fluid_colors['pred'], linewidth=2.5, alpha=0.9,
                     marker='s', markersize=4,
                     markevery=max(1, len(t_display)//20),
                     linestyle='--', zorder=2)

            ax4.fill_between(t_display, gt_total_speed, pred_total_speed,
                             where=(pred_total_speed > gt_total_speed),
                             alpha=0.2, color='red', label='Over-prediction')
            ax4.fill_between(t_display, gt_total_speed, pred_total_speed,
                             where=(pred_total_speed <= gt_total_speed),
                             alpha=0.2, color='blue', label='Under-prediction')

        ax4.set_xlabel('Time Index (hours)')
        ax4.set_ylabel('Flow Rate (mL/hour)')
        ax4.set_title('Total: Flow Rate Comparison', fontweight='bold')
        ax4.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax4.legend(loc='upper right', fontsize=9, frameon=True, framealpha=0.95, edgecolor='black', fancybox=True)
        ax4.set_ylim(bottom=0)

        final_gt_total = cum_data['gt']['total'][-1] if len(cum_data['gt']['total']) > 0 else 0
        final_pred_total = cum_data['pred_gated']['total'][-1] if len(cum_data['pred_gated']['total']) > 0 else 0
        mae_total = np.mean(np.abs(cum_data['pred_gated']['total'] - cum_data['gt']['total'])) if len(cum_data['gt']['total']) > 0 else 0

        info_text = (f"Final GT Total: {final_gt_total:,.0f} mL\n"
                     f"Final Pred Total: {final_pred_total:,.0f} mL\n"
                     f"MAE: {mae_total:,.0f} mL")

        if formula_total is not None and formula_total != 0:
            gt_diff = final_gt_total - formula_total
            pred_diff = final_pred_total - formula_total
            info_text += (f"\nFormula Diff (GT): {gt_diff:+,.0f} mL ({gt_diff/formula_total*100:+.1f}%)\n"
                          f"Formula Diff (Pred): {pred_diff:+,.0f} mL ({pred_diff/formula_total*100:+.1f}%)")

        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                           edgecolor='gray', linewidth=0.5))

        # ✅ 关键修改：不再调用 plt.tight_layout(...)，避免和 constrained_layout 冲突
        # plt.tight_layout(rect=[0, 0, 1, 0.96])

        return fig

    def plot_patient(self, timeseries_csv: str, pid: str, tbsa: Optional[float] = None, weight: Optional[float] = None) -> Dict[str, Path]:
        df = pd.read_csv(timeseries_csv)
        if df.empty:
            return {}

        formula_total = None
        if tbsa is not None and weight is not None:
            formula_total = self.formula_calculator.calculate(
                self.config.formula, float(tbsa), float(weight), self.config.formula_phase
            )

        fig = self._create_comprehensive_plot(df, pid, formula_total)

        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"patient_{pid}_{self.config.formula}_{timestamp}"
        out_files = {}

        png_path = self.output_dir / f"{base_name}.png"
        fig.savefig(png_path, dpi=self.config.plot_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        out_files['png'] = png_path

        preview_path = self.output_dir / f"{base_name}_preview.png"
        fig.savefig(preview_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')

        if self.config.save_pdf:
            pdf_path = self.output_dir / f"{base_name}.pdf"
            fig.savefig(pdf_path, bbox_inches='tight', facecolor='white', transparent=False)
            out_files['pdf'] = pdf_path

        plt.close(fig)
        return out_files

    def run_visualization(self, thresholds: List[float] = None, step_ml_scale: float = None):
        if thresholds is not None:
            self.thresholds = thresholds
        if step_ml_scale is not None:
            self.step_ml_scale = step_ml_scale

        summary_path = self.output_dir / "patients_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing {summary_path}")

        summary_df = pd.read_csv(summary_path)

        thresholds_path = self.output_dir / "used_thresholds.json"
        if thresholds_path.exists():
            try:
                loaded = json.load(open(thresholds_path, "r", encoding="utf-8"))
                if loaded and len(loaded) >= 3:
                    self.thresholds = loaded[:3]
            except Exception:
                pass

        ok = 0
        for _, row in summary_df.iterrows():
            pid = norm_pid(row["pid"])
            ts_path = self.output_dir / f"{pid}_timeseries.csv"
            if not ts_path.exists():
                continue
            tbsa = row.get("tbsa")
            weight = row.get("weight")
            res = self.plot_patient(str(ts_path), pid, tbsa, weight)
            if res:
                ok += 1

        print(f"[plot] done. {ok}/{len(summary_df)}")


# ============================================================
#                           MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fluid Resuscitation Inference + Visualization")

    parser.add_argument("--mode", type=str, default="plot", choices=["infer", "plot"])

    # data
    parser.add_argument("--data_pkl", type=str, default="data/output_data.pkl")
    parser.add_argument("--base_pkl", type=str, default="data/baseline.pkl")
    parser.add_argument("--output_dir", type=str, default="./per_patient_outputs")

    # model
    parser.add_argument("--model", type=str, default="tcn_film",
                        choices=["base", "transformer", "lstm_new", "cnn", "base_1", "flim", "tcn_film"])
    parser.add_argument("--history_length", type=int, default=4)
    parser.add_argument("--pred_length", type=int, default=1)
    parser.add_argument("--input_dim_base", type=int, default=10)
    parser.add_argument("--input_dim_temporal", type=int, default=19)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)

    # inference
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default="./ckpts/tcn_film_best_f1_0.7203_mae_21.3136.pth")
    parser.add_argument("--thresholds_json", type=str, default="./ckpts/best_thresholds.json")
    parser.add_argument("--test_ids_json", type=str, default=None)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--step_ml_scale", type=float, default=1.0)

    # plot
    parser.add_argument("--formula", type=str, default="TMMU-DRF")
    parser.add_argument("--formula_phase", type=str, default="first", choices=["first", "second"])
    parser.add_argument("--save_pdf", action="store_true")
    parser.add_argument("--plot_dpi", type=int, default=300)

    args = parser.parse_args()

    if args.mode == "infer":
        run_inference(args)
    else:
        # plot 模式：优先读取 output_dir/used_thresholds.json
        thresholds = None
        used_thr = Path(args.output_dir) / "used_thresholds.json"
        if used_thr.exists():
            try:
                thresholds = json.load(open(used_thr, "r", encoding="utf-8"))
            except Exception:
                thresholds = None
        if thresholds is None:
            thresholds = load_thresholds(args.thresholds_json)

        plot_config = PlotConfig(
            formula=args.formula,
            formula_phase=args.formula_phase,
            save_pdf=args.save_pdf,
            plot_dpi=args.plot_dpi
        )
        vis = VisualizationPipeline(
            config=plot_config,
            output_dir=args.output_dir,
            step_ml_scale=args.step_ml_scale,
            thresholds=thresholds
        )
        vis.run_visualization()

if __name__ == "__main__":
    main()
