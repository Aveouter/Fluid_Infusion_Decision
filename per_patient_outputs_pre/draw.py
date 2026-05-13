#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def infer_pid_from_filename(csv_path: Path) -> str:
    stem = csv_path.stem
    if stem.endswith("_timeseries"):
        stem = stem[:-len("_timeseries")]
    return stem


def load_timeseries(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    pid_guess = infer_pid_from_filename(csv_path)

    # 补齐/创建 pid
    if "pid" not in df.columns:
        df["pid"] = pid_guess
    else:
        s = df["pid"].astype(str).replace({"nan": "", "None": ""})
        df["pid"] = s
        df.loc[df["pid"].str.strip().eq(""), "pid"] = pid_guess

    # t_index 变成 int
    df["t_index"] = pd.to_numeric(df.get("t_index", 0), errors="coerce").fillna(0).astype(int)

    # 必要列
    required = [
        "gt_speed_crystal", "gt_speed_colloid", "gt_speed_water",
        "prob_crystal", "prob_colloid", "prob_water",
        "pred_speed_raw_crystal", "pred_speed_raw_colloid", "pred_speed_raw_water",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} 缺少必要列: {missing}")

    # 如果 cum 列不存在就自己算（更稳）
    for fluid in ["crystal", "colloid", "water"]:
        gt_c = f"gt_cum_{fluid}"
        pr_c = f"pred_cum_{fluid}"
        if gt_c not in df.columns:
            df[gt_c] = np.cumsum(df[f"gt_speed_{fluid}"].to_numpy())
        if pr_c not in df.columns:
            df[pr_c] = np.cumsum(df[f"pred_speed_raw_{fluid}"].to_numpy())

    if "gt_cum_total" not in df.columns:
        df["gt_cum_total"] = df["gt_cum_crystal"] + df["gt_cum_colloid"] + df["gt_cum_water"]
    if "pred_cum_total" not in df.columns:
        df["pred_cum_total"] = df["pred_cum_crystal"] + df["pred_cum_colloid"] + df["pred_cum_water"]

    return df


def plot_2x2(df: pd.DataFrame, out_png: Path, thresholds=(0.5, 0.5, 0.5), step_ml_scale=1.0):
    """
    CVPR-style clean figure:
      - No MAE / stats text box
      - Subplot 2 shows ONLY original (ungated) prediction (no gated curve)
      - No panel labels/annotations
      - Bottom-left replaced by 3 mini subplots (one per fluid) in the same cell
      - Unified y-limit across the 3 mini subplots (cleaner comparison)
    """
    pid = str(df["pid"].iloc[0])
    t = df["t_index"].to_numpy(dtype=float)

    thr = np.asarray(thresholds, dtype=float).reshape(1, 3)
    prob = df[["prob_crystal", "prob_colloid", "prob_water"]].to_numpy(dtype=float)
    raw  = df[["pred_speed_raw_crystal", "pred_speed_raw_colloid", "pred_speed_raw_water"]].to_numpy(dtype=float)

    # Probability-gated prediction (used in subplot 1 / mini-plots / subplot 4)
    mask  = (prob > thr).astype(float)
    gated = raw * mask

    pred_cum_gated = {
        "crystal": np.cumsum(gated[:, 0] * step_ml_scale),
        "colloid": np.cumsum(gated[:, 1] * step_ml_scale),
        "water":   np.cumsum(gated[:, 2] * step_ml_scale),
    }
    pred_cum_gated["total"] = (
        pred_cum_gated["crystal"]
        + pred_cum_gated["colloid"]
        + pred_cum_gated["water"]
    )

    cum_gt = {
        "crystal": df["gt_cum_crystal"].to_numpy(dtype=float),
        "colloid": df["gt_cum_colloid"].to_numpy(dtype=float),
        "water":   df["gt_cum_water"].to_numpy(dtype=float),
        "total":   df["gt_cum_total"].to_numpy(dtype=float),
    }
    cum_pred_orig_total = df["pred_cum_total"].to_numpy(dtype=float)

    gt_speed = np.vstack([
        df["gt_speed_crystal"].to_numpy(dtype=float),
        df["gt_speed_colloid"].to_numpy(dtype=float),
        df["gt_speed_water"].to_numpy(dtype=float),
    ]).T

    gt_total_speed   = gt_speed.sum(axis=1)
    pred_total_speed = gated.sum(axis=1)

    fluid_types = ["crystal", "colloid", "water"]
    fluid_names = ["Crystalloid", "Colloid", "Glucose"]

    # Clean palette
    colors = {
        "crystal": "#4C78A8",
        "colloid": "#F58518",
        "water":   "#54A24B",
        "gt":      "#111111",
        "pred":    "#7F7F7F",
    }

    # Global style
    plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.3,
    })

    # --- Layout with GridSpec (bottom-left split into 3 mini axes) ---
    fig = plt.figure(figsize=(14.5, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    gs_bl = gs[1, 0].subgridspec(3, 1, hspace=0.12)
    ax3a = fig.add_subplot(gs_bl[0, 0])
    ax3b = fig.add_subplot(gs_bl[1, 0], sharex=ax3a)
    ax3c = fig.add_subplot(gs_bl[2, 0], sharex=ax3a)

    ax4 = fig.add_subplot(gs[1, 1])

    fig.suptitle(f"Patient {pid}", y=1.02, fontsize=13, fontweight="bold")

    # ========= 1) Cumulative by fluid type (GT vs gated pred) =========
    for i, f in enumerate(fluid_types):
        ax1.plot(t, cum_gt[f], color=colors[f], alpha=0.95,
                 label=f"Ground Truth: {fluid_names[i]}")
        ax1.plot(t, pred_cum_gated[f], color=colors[f], alpha=0.65,
                 linestyle="--", label=f"Prediction: {fluid_names[i]}")

    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Cumulative Infused Volume (mL)")
    ax1.set_title("Cumulative Volume by Fluid Type", fontweight="bold")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    ax1.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.95)

    # ========= 2) Total cumulative (GT vs ORIGINAL ONLY) =========
    ax2.plot(t, cum_gt["total"], color=colors["gt"], linewidth=2.8,
             label="Ground Truth")
    ax2.plot(t, cum_pred_orig_total, color=colors["pred"],
             linestyle=":", linewidth=2.4,
             label="Model Prediction")

    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Cumulative Infused Volume (mL)")
    ax2.set_title("Total Cumulative Infused Volume", fontweight="bold")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95)

    # ========= 3) Bottom-left: 3 mini subplots (one per fluid) =========
    display_step = max(1, len(t) // 160)
    t_disp  = t[::display_step]
    gt_disp = gt_speed[::display_step, :]
    pr_disp = gated[::display_step, :]

    # unified y-limit across mini plots
    y_max = float(np.nanmax(np.vstack([gt_disp, pr_disp])) if gt_disp.size else 0.0)
    y_top = max(1.0, y_max * 1.05)

    ax3_list = [ax3a, ax3b, ax3c]
    for i, ax in enumerate(ax3_list):
        f = fluid_types[i]
        name = fluid_names[i]

        ax.plot(t_disp, gt_disp[:, i], color=colors[f], alpha=0.95, label="Ground Truth")
        ax.plot(t_disp, pr_disp[:, i], color=colors["pred"], alpha=0.95, linestyle="--",
                label="Prediction")

        ax.set_title(name, loc="left", fontweight="bold", pad=2)
        ax.set_ylabel("Rate\n(mL·h$^{-1}$)")
        ax.set_ylim(0, y_top)

        if i == 0:
            ax.legend(loc="upper right", frameon=True, framealpha=0.95, borderpad=0.3)

        if i < 2:
            plt.setp(ax.get_xticklabels(), visible=False)

    ax3c.set_xlabel("Time (hours)")

    # ========= 4) Total infusion rate (GT vs gated pred) =========
    gt_tot_disp   = gt_total_speed[::display_step]
    pred_tot_disp = pred_total_speed[::display_step]

    ax4.plot(t_disp, gt_tot_disp,
             color=colors["gt"], linewidth=2.6,
             label="Ground Truth")
    ax4.plot(t_disp, pred_tot_disp,
             color=colors["pred"], linewidth=2.6,
             linestyle="--",
             label="Model Prediction")

    ax4.fill_between(t_disp, gt_tot_disp, pred_tot_disp, alpha=0.12)

    ax4.set_xlabel("Time (hours)")
    ax4.set_ylabel("Total Infusion Rate (mL·h$^{-1}$)")
    ax4.set_title("Comparison of Total Infusion Rate", fontweight="bold")
    ax4.set_ylim(bottom=0)
    ax4.legend(loc="upper right", frameon=True, framealpha=0.95)

    # Save
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="./per_patient_outputs_pre", help="folder containing timeseries csv files")
    ap.add_argument("--out_dir", default="./results_plus/figs", help="folder to save png results")
    ap.add_argument("--pattern", default="*_timeseries.csv", help="glob pattern, default: *_timeseries.csv")
    ap.add_argument("--thr", nargs=3, type=float, default=[0.5, 0.5, 0.5], help="3 thresholds")
    ap.add_argument("--step_ml_scale", type=float, default=1.0)
    ap.add_argument("--recursive", action="store_true", help="search files recursively")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        raise FileNotFoundError(in_dir)

    files = sorted(in_dir.rglob(args.pattern)) if args.recursive else sorted(in_dir.glob(args.pattern))
    if not files:
        print(f"[warn] no files matched: {in_dir} / {args.pattern}")
        return

    ok, fail = 0, 0
    for csv_path in files:
        try:
            df = load_timeseries(csv_path)
            pid = str(df["pid"].iloc[0])
            out_png = out_dir / f"{pid}_2x2.png"
            plot_2x2(df, out_png=out_png, thresholds=tuple(args.thr), step_ml_scale=float(args.step_ml_scale))
            ok += 1
            print(f"[ok] {csv_path.name} -> {out_png.name}")
        except Exception as e:
            fail += 1
            print(f"[fail] {csv_path}: {e}")

    print(f"[done] ok={ok}, fail={fail}, total={len(files)}")


if __name__ == "__main__":
    main()
