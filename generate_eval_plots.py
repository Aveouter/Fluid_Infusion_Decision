# -*- coding: utf-8 -*-
# generate_eval_plots_standalone.py — DISTINCT & VALUABLE PLOTS ONLY (fixed & polished)
#
# 与已有 AE/RE 柱状/ECDF/Bland–Altman 等重复内容不同，此脚本生成一组“非重复、信息量高”的图。
# 修复点：
#   - 修正 _efficiency_scatter 的布尔索引优先级（必须加括号）。
#   - 更健壮的空值处理与边界保护。
#
# Modes
# 1) SAMPLE 模式（逐样本 CSV，含 GT/PRED）：
#    - Calibration by predicted deciles（按预测分位分箱，箱均值 True vs Pred + SD 误差棒）
#    - Error vs True magnitude（按真值分箱，中位 AE + IQR 带）
#    - Pareto curve of cumulative AE（累计贡献曲线）
#    - Residual QQ-plot vs Normal（残差分布正态性）
#    - AE by TBSA bins（若有 TBSA 列；或按已有分组）
#    - Calibration slope per model（多模型时的 y~a+b·pred 斜率）
#
# 2) SUMMARY 模式（effectiveness_summary_mean_std.csv）：
#    - Heatmaps (AE/RE) per metric：行=Formula，列=Day；单元格标注均值
#    - Win-rate tiles：统计 Model 相对各公式在 4×3=12 个格子中的胜出次数
#    - Efficiency scatter：总体 AE_Total_mean vs RE_Total_mean（越靠左下越优）
#
# 输出：每图单独 PDF（Matplotlib）。
# 用法：
#   python generate_eval_plots_standalone.py --csv <file.csv> --dataset_name <Name> --out_dir ./figs
#
import argparse
import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Global style configuration
# -----------------------------
plt.rcParams["font.size"] = 22           # tick/label font size
plt.rcParams["axes.titlesize"] = 18      # title size
plt.rcParams["axes.labelsize"] = 22
plt.rcParams["legend.fontsize"] = 18
plt.rcParams["figure.dpi"] = 150

# color palette (colorblind-friendly leaning, blue/purple/cyan)
BLUE        = (31/255, 119/255, 180/255, 0.95)
PURPLE      = (148/255, 103/255, 189/255, 0.95)
CYAN        = (23/255, 190/255, 207/255, 0.95)
DARK_BLUE   = (44/255,  62/255,  80/255,  0.95)
LIGHT_GRAY  = (0.92, 0.92, 0.95, 1.0)
ACCENT_GRAY = (0.45, 0.45, 0.50, 0.95)

DAYS_ORDER = ["Day1", "Day2", "Total"]
METRICS = ["Electrolyte","Colloid","Water","Total"]

# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def savefig_pdf(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def detect_column(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in cols:
            return cols[name]
    for c in df.columns:
        lc = c.lower()
        for name in candidates:
            if name in lc:
                return c
    return None


def compute_basic_arrays(df):
    y_col = detect_column(df, ["y", "label", "gt", "true", "actual","ground_truth","target"])
    p_col = detect_column(df, ["y_pred", "pred", "prediction","predicted","output"])
    if y_col is None or p_col is None:
        raise ValueError("Could not auto-detect ground truth and/or prediction columns.")
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
    p = pd.to_numeric(df[p_col], errors="coerce").to_numpy()
    ae = np.abs(p - y)
    with np.errstate(divide="ignore", invalid="ignore"):
        re = np.where(np.abs(y) > 0, ae / np.abs(y), np.nan)
    return y, p, ae, re, y_col, p_col

# -----------------------------
# DISTINCT plots for SAMPLE mode
# -----------------------------

def plot_calibration_by_pred(y, p, dataset_name, out_dir, bins=10):
    valid = ~(np.isnan(y) | np.isnan(p))
    y = y[valid]; p = p[valid]
    if len(y) == 0: return
    bins = max(3, int(bins))
    q = np.linspace(0, 1, bins+1)
    qs = np.quantile(p, q)
    if not np.all(np.isfinite(qs)):
        return
    idx = np.digitize(p, qs[1:-1], right=False)
    mean_pred, mean_true, std_true = [], [], []
    for b in range(bins):
        m = (idx == b)
        if not np.any(m):
            mean_pred.append(np.nan); mean_true.append(np.nan); std_true.append(np.nan)
        else:
            mean_pred.append(np.nanmean(p[m]))
            mean_true.append(np.nanmean(y[m]))
            std_true.append(np.nanstd(y[m]))
    mean_pred = np.array(mean_pred); mean_true = np.array(mean_true); std_true = np.array(std_true)

    lo = np.nanmin([np.nanmin(mean_pred), np.nanmin(mean_true)])
    hi = np.nanmax([np.nanmax(mean_pred), np.nanmax(mean_true)])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return

    fig = plt.figure(figsize=(8,6))
    ax = plt.gca(); ax.set_facecolor(LIGHT_GRAY)
    for spine in ax.spines.values(): spine.set_visible(False)
    plt.plot([lo, hi], [lo, hi], '--', c=ACCENT_GRAY, lw=2, label='Ideal')
    plt.errorbar(mean_pred, mean_true, yerr=std_true, fmt='o-', lw=2, ms=6, c=BLUE, ecolor=DARK_BLUE, capsize=4, label='Mean±SD True')
    plt.title("Calibration by Predicted Deciles")
    plt.xlabel("Mean Pred in Bin")
    plt.ylabel("Mean True in Bin")
    plt.legend(loc='best')
    savefig_pdf(fig, os.path.join(out_dir, f"Calibration_by_Pred_{dataset_name}.pdf"))


def plot_error_vs_true_binned(y, ae, dataset_name, out_dir, bins=12):
    valid = ~(np.isnan(y) | np.isnan(ae)); y = y[valid]; ae = ae[valid]
    if len(y) == 0: return
    bins = max(3, int(bins))
    qs = np.quantile(y, np.linspace(0,1,bins+1))
    idx = np.digitize(y, qs[1:-1], right=False)
    x_mid, med, q1, q3 = [], [], [], []
    for b in range(bins):
        m = (idx==b)
        if not np.any(m):
            continue
        yb = y[m]; aeb = ae[m]
        if yb.size==0: continue
        x_mid.append(np.nanmedian(yb))
        med.append(np.nanmedian(aeb))
        q1.append(np.nanpercentile(aeb, 25))
        q3.append(np.nanpercentile(aeb, 75))
    if not x_mid:
        return
    x_mid = np.array(x_mid); med = np.array(med); q1=np.array(q1); q3=np.array(q3)

    fig = plt.figure(figsize=(10,6))
    ax = plt.gca(); ax.set_facecolor(LIGHT_GRAY)
    for spine in ax.spines.values(): spine.set_visible(False)
    plt.fill_between(x_mid, q1, q3, color=CYAN, alpha=0.3, label='IQR of AE')
    plt.plot(x_mid, med, '-o', c=PURPLE, lw=2, ms=6, label='Median AE')
    plt.title("Error vs True Magnitude (binned)")
    plt.xlabel("True (bin median)")
    plt.ylabel("Absolute Error (mL)")
    plt.legend(loc='best')
    savefig_pdf(fig, os.path.join(out_dir, f"AE_vs_True_Binned_{dataset_name}.pdf"))


def plot_pareto_cumulative_ae(ae, dataset_name, out_dir):
    vals = ae[~np.isnan(ae)]
    if len(vals)==0: return
    vals = np.sort(vals)[::-1]
    if vals.size==0 or vals[-1]==0:
        return
    cum = np.cumsum(vals)
    share = cum / cum[-1]
    x = np.arange(1, len(vals)+1) / len(vals)
    fig = plt.figure(figsize=(8,6))
    ax = plt.gca(); ax.set_facecolor(LIGHT_GRAY)
    for spine in ax.spines.values(): spine.set_visible(False)
    plt.plot(x*100, share*100, '-', c=BLUE, lw=3)
    plt.title("Pareto of Absolute Error Contribution")
    plt.xlabel("Top-k% Samples")
    plt.ylabel("Cumulative % of Total AE")
    savefig_pdf(fig, os.path.join(out_dir, f"Pareto_AE_{dataset_name}.pdf"))


def plot_residual_qq(residuals, dataset_name, out_dir):
    r = residuals[~np.isnan(residuals)]
    if len(r)==0: return
    r = np.sort(r)
    n = len(r)
    probs = (np.arange(1, n+1) - 0.5)/n
    # Prefer numpy.erfinv if available; else approximate via large-sample quantiles of normal
    if hasattr(np, 'erfinv'):
        z = np.sqrt(2)*np.erfinv(2*probs-1)
    else:
        # deterministic fallback using percent point function approximation via poly (not perfect but stable)
        # Here we use numpy's percentile on a fixed grid to approximate normal quantiles
        grid = np.linspace(0,1,10001)[1:-1]
        normal = np.sort(np.random.default_rng(12345).normal(size=100000))
        z = np.interp(probs, np.linspace(1/(len(normal)+1), len(normal)/(len(normal)+1), len(normal)), normal)
    fig = plt.figure(figsize=(8,6))
    ax = plt.gca(); ax.set_facecolor(LIGHT_GRAY)
    for spine in ax.spines.values(): spine.set_visible(False)
    plt.scatter(z, r, s=20, c=[PURPLE], alpha=0.7)
    lo = min(np.nanmin(z), np.nanmin(r)); hi = max(np.nanmax(z), np.nanmax(r))
    plt.plot([lo,hi],[lo,hi],'--', c=ACCENT_GRAY, lw=2)
    plt.title("Residual QQ-Plot vs Normal")
    plt.xlabel("Normal Quantiles")
    plt.ylabel("Residuals")
    savefig_pdf(fig, os.path.join(out_dir, f"QQ_Residuals_{dataset_name}.pdf"))


def plot_ae_by_tbsa_bins(df, ae, dataset_name, out_dir):
    tbsa_col = detect_column(df, ["tbsa","burn_area","bsa","total_bsa"])
    if tbsa_col is None: return
    tbsa = pd.to_numeric(df[tbsa_col], errors='coerce').to_numpy()
    valid = ~(np.isnan(tbsa)|np.isnan(ae))
    tbsa = tbsa[valid]; ae = ae[valid]
    if len(tbsa)==0: return
    bins = [0,10,20,30,40,50,60,100]
    labels = ["0-10","10-20","20-30","30-40","40-50","50-60","60+"]
    idx = np.digitize(tbsa, bins, right=False)-1
    data, tick = [], []
    for i,l in enumerate(labels):
        arr = ae[idx==i]
        arr = arr[~np.isnan(arr)]
        if len(arr)>0:
            data.append(arr); tick.append(l)
    if not data: return
    fig = plt.figure(figsize=(10,6))
    bp = plt.boxplot(data, labels=tick, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor(CYAN); patch.set_alpha(0.85); patch.set_edgecolor('white')
    for med in bp['medians']:
        med.set_color(DARK_BLUE); med.set_linewidth(2)
    plt.title("AE by TBSA Bins")
    plt.xlabel("TBSA (%)")
    plt.ylabel("Absolute Error (mL)")
    savefig_pdf(fig, os.path.join(out_dir, f"AE_by_TBSA_{dataset_name}.pdf"))


def plot_calibration_slope_per_model(df, y, p, dataset_name, out_dir):
    model_col = detect_column(df, ["model","algo","method"])
    if model_col is None: return
    rows = []
    for name, sub in df.assign(_y=y, _p=p).groupby(model_col):
        yy = pd.to_numeric(sub['_y'], errors='coerce').to_numpy()
        pp = pd.to_numeric(sub['_p'], errors='coerce').to_numpy()
        m = ~(np.isnan(yy)|np.isnan(pp))
        yy=yy[m]; pp=pp[m]
        if len(yy)<2: continue
        x = np.vstack([np.ones_like(pp), pp]).T
        beta, *_ = np.linalg.lstsq(x, yy, rcond=None)
        a,b = beta[0], beta[1]
        rows.append((str(name), a, b))
    if not rows: return
    labels = [r[0] for r in rows]
    slopes = [r[2] for r in rows]
    fig = plt.figure(figsize=(10,6))
    plt.bar(np.arange(len(slopes)), slopes, color=BLUE, edgecolor='white')
    plt.xticks(np.arange(len(slopes)), labels, rotation=0)
    plt.axhline(1.0, linestyle='--', c=ACCENT_GRAY, lw=2)
    plt.title("Calibration Slope per Model (y ~ a + b·pred)")
    plt.ylabel("Slope b (ideal=1)")
    savefig_pdf(fig, os.path.join(out_dir, f"CalibrationSlope_byModel_{dataset_name}.pdf"))

# -----------------------------
# SUMMARY mode: detection and DISTINCT plots
# -----------------------------

def is_summary_table(df: pd.DataFrame) -> bool:
    has_formula = "Formula" in df.columns
    has_day = "Day" in df.columns
    has_any_ae = any(col.startswith("AE_") and col.endswith("_mean") for col in df.columns)
    has_any_re = any(col.startswith("RE_") and col.endswith("_mean") for col in df.columns)
    return has_formula and has_day and (has_any_ae or has_any_re)


def _heatmap_metric(df, kind, metric, dataset_name, out_dir):
    assert kind in ("AE","RE")
    if f"{kind}_{metric}_mean" not in df.columns: return
    formulas = list(df['Formula'].unique())
    M = np.full((len(formulas), len(DAYS_ORDER)), np.nan)
    for i,f in enumerate(formulas):
        for j,d in enumerate(DAYS_ORDER):
            row = df[(df['Formula']==f)&(df['Day']==d)]
            if not row.empty:
                M[i,j] = float(row[f"{kind}_{metric}_mean"].values[0])
    fig = plt.figure(figsize=(10, max(6, 0.6*len(formulas))))
    ax = plt.gca()
    im = ax.imshow(M, aspect='auto', cmap='Blues' if kind=='AE' else 'Purples')
    ax.set_xticks(np.arange(len(DAYS_ORDER))); ax.set_xticklabels(DAYS_ORDER)
    ax.set_yticks(np.arange(len(formulas)));   ax.set_yticklabels(formulas)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i,j]):
                ax.text(j, i, f"{M[i,j]:.0f}", ha='center', va='center', color='black')
    ax.set_title(f"{kind} Heatmap — {metric}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    savefig_pdf(fig, os.path.join(out_dir, f"{kind}_Heatmap_{metric}_{dataset_name}.pdf"))


def _winrate_tiles(df, dataset_name, out_dir):
    if 'Model' not in df['Formula'].unique(): return
    formulas = [f for f in df['Formula'].unique() if f != 'Model']
    wins_AE, wins_RE = [], []
    for f in formulas:
        cnt_ae = 0; cnt_re = 0
        for metric in METRICS:
            for d in DAYS_ORDER:
                rM = df[(df['Formula']=='Model') & (df['Day']==d)]
                rF = df[(df['Formula']==f)      & (df['Day']==d)]
                if rM.empty or rF.empty: continue
                if f"AE_{metric}_mean" in df.columns:
                    mM = float(rM[f"AE_{metric}_mean"].values[0])
                    mF = float(rF[f"AE_{metric}_mean"].values[0])
                    if mM < mF: cnt_ae += 1
                if f"RE_{metric}_mean" in df.columns:
                    rMv = float(rM[f"RE_{metric}_mean"].values[0])
                    rFv = float(rF[f"RE_{metric}_mean"].values[0])
                    if rMv < rFv: cnt_re += 1
        wins_AE.append(cnt_ae); wins_RE.append(cnt_re)
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    W = np.vstack([wins_AE, wins_RE])  # 2 × F
    im = ax.imshow(W, aspect='auto', cmap='Greens')
    ax.set_yticks([0,1]); ax.set_yticklabels(['AE wins','RE wins'])
    ax.set_xticks(np.arange(len(formulas))); ax.set_xticklabels(formulas, rotation=20, ha='right')
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            ax.text(j, i, str(W[i,j]), ha='center', va='center', color='black')
    ax.set_title('Model win counts vs each formula (lower is better) — 12 max')
    fig.colorbar(im, ax=ax, shrink=0.8)
    savefig_pdf(fig, os.path.join(out_dir, f"WinRateTiles_{dataset_name}.pdf"))


def _efficiency_scatter(df, dataset_name, out_dir):
    # For each formula: overall AE_Total_mean and RE_Total_mean averaged over metrics
    points = []
    for f in df['Formula'].unique():
        if 'Day' in df.columns:
            r = df[(df['Formula'] == f) & (df['Day'].isin(DAYS_ORDER))]
        else:
            r = df[(df['Formula'] == f)]
        ae_vals = []; re_vals = []
        for metric in METRICS:
            colA = f"AE_{metric}_mean"; colR = f"RE_{metric}_mean"
            if colA in df.columns:
                v = r[colA].to_numpy(dtype=float)
                v = v[~np.isnan(v)]
                if v.size>0: ae_vals.append(np.nanmean(v))
            if colR in df.columns:
                v = r[colR].to_numpy(dtype=float)
                v = v[~np.isnan(v)]
                if v.size>0: re_vals.append(np.nanmean(v))
        if ae_vals and re_vals:
            points.append((f, np.nanmean(ae_vals), np.nanmean(re_vals)))
    if not points: return
    fig = plt.figure(figsize=(10,7))
    for name, x, y in points:
        plt.scatter(x, y, s=120, c=[BLUE] if name!='Model' else [PURPLE], alpha=0.9, edgecolor='white', linewidth=1.0)
        plt.text(x, y, f" {name}", va='center', fontsize=18)
    plt.xlabel('Overall AE_Total_mean (mL) ↓')
    plt.ylabel('Overall RE_Total_mean (%) ↓')
    plt.title('Efficiency Frontier (lower-left is better)')
    savefig_pdf(fig, os.path.join(out_dir, f"EfficiencyScatter_{dataset_name}.pdf"))


def run_summary_mode(df: pd.DataFrame, dataset_name: str, out_dir: str):
    for metric in METRICS:
        _heatmap_metric(df, 'AE', metric, dataset_name, out_dir)
        _heatmap_metric(df, 'RE', metric, dataset_name, out_dir)
    _winrate_tiles(df, dataset_name, out_dir)
    _efficiency_scatter(df, dataset_name, out_dir)

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate DISTINCT evaluation plots from a CSV (sample or summary mode).")
    parser.add_argument("--csv", required=True, help="Path to input CSV file.")
    parser.add_argument("--dataset_name", required=True, help="Dataset name used in output filenames.")
    parser.add_argument("--out_dir", default="./figs", help="Directory to save figures (PDF).")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    df = pd.read_csv(args.csv)

    # SUMMARY first
    if is_summary_table(df):
        run_summary_mode(df, args.dataset_name, args.out_dir)
        print("[ALL DONE] Summary mode: heatmaps, win-tiles, efficiency scatter saved.")
        return

    # SAMPLE mode
    y, p, ae, re, y_col, p_col = compute_basic_arrays(df)
    plot_calibration_by_pred(y, p, args.dataset_name, args.out_dir, bins=10)
    plot_error_vs_true_binned(y, ae, args.dataset_name, args.out_dir, bins=12)
    plot_pareto_cumulative_ae(ae, args.dataset_name, args.out_dir)
    plot_residual_qq(p - y, args.dataset_name, args.out_dir)
    plot_ae_by_tbsa_bins(df, ae, args.dataset_name, args.out_dir)
    plot_calibration_slope_per_model(df, y, p, args.dataset_name, args.out_dir)

    print("[ALL DONE] Sample mode: distinct plots saved.")


if __name__ == "__main__":
    main()
