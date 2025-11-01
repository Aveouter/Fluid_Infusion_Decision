#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effectiveness Validation (Formulas + Model)
- 输出：总超图（四宫格）+ 可选单指标分图 + 汇总表格（CSV / XLSX / Markdown / 终端友好打印）
- 科研级主题：
    * 字体：Cambria（自动回退 Times New Roman / DejaVu Serif / SimSun）
    * 颜色：提供多套 **色盲友好** 科研配色可选（Okabe-Ito / Tableau 10 子集 / ColorBrewer Blues/PuBu 等）
    * 细节：半透明柱、深色误差线、浅灰网格、白色边框、优雅对比

运行示例：
python EffectivenessValidation_Styled.py \
  --ts_pkl /baksv/CIGIT/GXN_Liuxy/fluid/data/output_data.pkl \
  --base_pkl /baksv/CIGIT/GXN_Liuxy/fluid/data/baseline.pkl \
  --pred_dir /baksv/CIGIT/GXN_Liuxy/fluid/per_patient_outputs \
  --out_dir /baksv/CIGIT/GXN_Liuxy/fluid/results_plus \
  --formula_water_strategy midpoint \
  --model_thresholds 0.5 0.5 0.5 \
  --palette okabe \
  --plot_metric_figs
"""

import os
import json
import pickle
import argparse
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 配置：经验公式（Day1/Day2）
# =========================
K = 1000.0
FORMULAS = OrderedDict({
    "Evans":    {"Day1": (1.0, 1.0, 2*K),    "Day2": (0.5, 0.5, 2*K)},
    "Brooke":   {"Day1": (1.5, 0.5, 2*K),    "Day2": (0.75,0.25,2*K)},
    "Parkland": {"Day1": (4.0, 0.0, 0.0),    "Day2": (0.0,  1.25, (0.0, 2*K))},
    "Monafo":   {"Day1": (2.0, 0.0, 0.0),    "Day2": (1.0,  0.0,  0.0)},
    "TMMU":     {"Day1": (1.0, 0.5, 2*K),    "Day2": (0.5,  0.25, 2*K)},
    "RJH":      {"Day1": (0.75,0.75,(3*K,4*K)),"Day2":(0.375,0.375,(3*K,4*K))},
    "PLA-304F": {"Day1": (0.95,0.95,(3*K,4*K)),"Day2":(0.725,0.725,3*K)},
    "TMMU-DRF": {"Day1": (1.3, 1.3, 2*K),    "Day2": (0.5,  0.25, 2*K)},
})

# =========================
# 公式计算
# =========================
def formula_day_volumes(
    formula: str,
    day: str,                # "Day1" / "Day2" / "Total"
    weight: float,
    tbsa: float,
    water_strategy: str = "midpoint",
    anchor: dict | None = None
) -> dict:
    if day == "Total":
        d1 = formula_day_volumes(formula, "Day1", weight, tbsa, water_strategy, anchor)
        d2 = formula_day_volumes(formula, "Day2", weight, tbsa, water_strategy, anchor)
        out = {
            "Electrolyte": d1["Electrolyte"] + d2["Electrolyte"],
            "Colloid":     d1["Colloid"]     + d2["Colloid"],
            "Water":       d1["Water"]       + d2["Water"],
        }
        out["Total"] = out["Electrolyte"] + out["Colloid"] + out["Water"]
        return out

    if formula not in FORMULAS or day not in FORMULAS[formula]:
        raise KeyError(f"Unknown formula/day: {formula}/{day}")

    a_e, a_c, w_rng = FORMULAS[formula][day]
    elec = float(a_e) * float(weight) * float(tbsa)
    coll = float(a_c) * float(weight) * float(tbsa)

    if isinstance(w_rng, (tuple, list)) and len(w_rng) == 2:
        low, high = float(w_rng[0]), float(w_rng[1])
        if water_strategy == "lower":
            water = low
        elif water_strategy == "upper":
            water = high
        elif water_strategy == "anchor":
            if anchor and formula in anchor and day in anchor[formula]:
                water = float(anchor[formula][day])
            else:
                water = 0.5 * (low + high)
        else:  # midpoint
            water = 0.5 * (low + high)
    else:
        water = float(w_rng)

    total = elec + coll + water
    return {"Electrolyte": elec, "Colloid": coll, "Water": water, "Total": total}

# =========================
# IO & 基础工具
# =========================
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def norm_pid(pid):
    pid = str(pid)
    return pid[:-4] if pid.endswith(".csv") else pid

def get_weight_tbsa(base_entry):
    w = tbsa = None
    if isinstance(base_entry, dict):
        for k in ["weight","Weight","wt","WT"]:
            if k in base_entry:
                w = base_entry[k]
                break
        for k in ["tbsa","TBSA","burn_area","BSA"]:
            if k in base_entry:
                tbsa = base_entry[k]
                break
    elif isinstance(base_entry, (list, tuple)):
        try:
            w = base_entry[3]
            tbsa = base_entry[5]
        except Exception:
            pass
    return (float(w) if w is not None else None,
            float(tbsa) if tbsa is not None else None)

# =========================
# 统计 GT & Model（日量）
# =========================
def compute_gt_day_volumes(ts_entry, step_hours=1.0):
    if hasattr(ts_entry["label"], "values"):
        arr = np.asarray(ts_entry["label"].values, dtype=float)
    else:
        arr = np.asarray(ts_entry["label"], dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 6:
        return None
    T = arr.shape[0]
    inst = arr[:, 0:3] * arr[:, 3:6] * float(step_hours)

    def seg_sum(start, end):
        if T == 0 or start >= T:
            return np.zeros(3, dtype=float)
        end = min(end, T-1)
        if end < start:
            return np.zeros(3, dtype=float)
        return inst[start:end+1, :].sum(axis=0)

    d1 = seg_sum(0, 23)
    d2 = seg_sum(24, 47)
    tot = d1 + d2
    out = {
        "Day1": {"Electrolyte": d1[0], "Colloid": d1[1], "Water": d1[2]},
        "Day2": {"Electrolyte": d2[0], "Colloid": d2[1], "Water": d2[2]},
        "Total":{"Electrolyte": tot[0], "Colloid": tot[1], "Water": tot[2]},
    }
    for k in out:
        x = out[k]
        x["Total"] = x["Electrolyte"] + x["Colloid"] + x["Water"]
    return out

def compute_model_day_volumes(csv_path, step_hours=1.0, thresholds=(0.5,0.5,0.5)):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)

    def pick(name_raw, name_old):
        return df[name_raw].values if name_raw in df.columns else df[name_old].values

    pr = np.stack([
        df["prob_crystal"].values,
        df["prob_colloid"].values,
        df["prob_water"].values
    ], axis=1)
    sp = np.stack([
        pick("pred_speed_raw_crystal","pred_speed_crystal"),
        pick("pred_speed_raw_colloid","pred_speed_colloid"),
        pick("pred_speed_raw_water","pred_speed_water"),
    ], axis=1)

    thr = np.asarray(thresholds, dtype=float).reshape(1,3)
    inst = (pr > thr).astype(float) * sp * float(step_hours)

    N = inst.shape[0]
    def seg_sum(start, end):
        if N == 0 or start >= N:
            return np.zeros(3, dtype=float)
        end = min(end, N-1)
        if end < start:
            return np.zeros(3, dtype=float)
        return inst[start:end+1, :].sum(axis=0)

    d1 = seg_sum(0, 23)
    d2 = seg_sum(24, 47)
    tot = d1 + d2
    out = {
        "Day1": {"Electrolyte": d1[0], "Colloid": d1[1], "Water": d1[2]},
        "Day2": {"Electrolyte": d2[0], "Colloid": d2[1], "Water": d2[2]},
        "Total":{"Electrolyte": tot[0], "Colloid": tot[1], "Water": tot[2]},
    }
    for k in out:
        x = out[k]
        x["Total"] = x["Electrolyte"] + x["Colloid"] + x["Water"]
    return out

# =========================
# 科研主题：字体 & 配色
# =========================

def mean_std(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return (0.0, 0.0)
    return (float(np.mean(a)), float(np.std(a)))


def _matplotlib_theme():
    # 字体：Cambria 优先，自动回退
    plt.rcParams.update({
        'font.family': ['Cambria', 'Times New Roman', 'DejaVu Serif', 'SimSun'],
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20
    })


def get_palette(name: str):
    """返回 3 色条形图调色板（色盲友好优先）。
    name ∈ {okabe, tableau, brewer_blues, brewer_pubu, mono_blue}
    """
    name = (name or '').lower()
    if name == 'okabe':
        # Okabe–Ito（色盲友好），选 3 色：蓝、橙、绿（但保持蓝系协调，把橙替换为紫，见下）
        # 原色：#0072B2(蓝) #D55E00(橙) #009E73(绿)
        return [
            (0/255, 114/255, 178/255, 0.95),  # 蓝 Day1
            (130/255,  99/255, 170/255, 0.95),# 紫 Day2（自定义，与蓝协调且区分）
            (0/255, 158/255, 115/255, 0.95),  # 绿 Day3
        ]
    if name == 'tableau':
        # Tableau 10 子集（色盲友好）：蓝、紫、青
        return [
            (31/255, 119/255, 180/255, 0.95),  # 蓝
            (148/255, 103/255, 189/255, 0.95), # 紫
            (23/255, 190/255, 207/255, 0.95),  # 青
        ]
    if name == 'brewer_blues':
        # ColorBrewer Blues(7) 中挑 3 个分段（深→中→浅），加高透明以显层次
        return [
            (8/255,  48/255, 107/255, 0.95),   # 深蓝
            (49/255,130/255, 189/255, 0.95),   # 中蓝
            (158/255,202/255,225/255,0.95),    # 浅蓝
        ]
    if name == 'brewer_pubu':
        # ColorBrewer PuBu(7)
        return [
            (2/255,  56/255, 88/255,  0.95),   # 深青蓝
            (44/255,123/255,182/255, 0.95),    # 中蓝
            (158/255,188/255,218/255,0.95),    # 浅蓝紫
        ]
    if name == 'mono_blue':
        # 单一蓝系（深 / 中 / 浅），风格克制
        return [
            (0.16, 0.34, 0.64, 0.95),
            (0.28, 0.54, 0.86, 0.95),
            (0.60, 0.78, 1.00, 0.95),
        ]
    # 默认：okabe
    return get_palette('okabe')

# =========================
# 绘图：总超图 & 分图
# =========================

def ultra_plot_allinone(
    ae_dict,
    re_dict,
    out_dir,
    fig_name_prefix="ULTRA",
    y_label="Absolute Error (mL)",
    title_suffix="Formulas + Model vs GT (Day1/Day2/Total)",
    palette_name='okabe',
):
    os.makedirs(out_dir, exist_ok=True)
    formula_names = list(FORMULAS.keys()) + ["Model"]
    metrics = ["Electrolyte", "Colloid", "Water", "Total"]
    days = ["Day1", "Day2", "Total"]

    _matplotlib_theme()
    palette = get_palette(palette_name)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    axes = axes.ravel()

    width = 0.24
    x = np.arange(len(formula_names))

    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        ax.set_facecolor((0.97, 0.97, 0.97))
        for spine in ax.spines.values():
            spine.set_visible(False)
        for di, day in enumerate(days):
            means, stds = [], []
            for fn in formula_names:
                src = ae_dict if y_label.lower().startswith("absolute") else re_dict
                vals = src.get(fn, {}).get(day, {}).get(metric, [])
                m, s = mean_std(vals)
                means.append(m)
                stds.append(s)
            ax.bar(
                x + di * width, means, yerr=stds, width=width,
                color=palette[di % len(palette)], edgecolor='white', linewidth=1.0,
                error_kw={'elinewidth': 1.1, 'ecolor': 'black', 'capsize': 3}, label=day
            )
        ax.set_title(metric, color='#2B4C7E')
        ax.set_xticks(x + width)
        ax.set_xticklabels(formula_names, rotation=40, ha="right")
        ax.set_ylabel(y_label)
        ax.grid(axis="y", alpha=0.25, linestyle='--')
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', framealpha=0.85)

    fig.suptitle(f"{y_label} — {title_suffix}", fontweight='bold', color='#2B4C7E')
    suffix = 'AE' if y_label.lower().startswith('absolute') else 'RE'
    png_path = os.path.join(out_dir, f"{fig_name_prefix}_{suffix}_AllInOne.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[PLOT SAVED] {png_path}")


def plot_metric_figure(
    src_dict,
    out_dir,
    metric: str,
    fig_name_prefix: str,
    y_label: str,
    palette_name='okabe',
):
    os.makedirs(out_dir, exist_ok=True)
    formula_names = list(FORMULAS.keys()) + ["Model"]
    days = ["Day1", "Day2", "Total"]

    _matplotlib_theme()
    palette = get_palette(palette_name)

    width = 0.24
    x = np.arange(len(formula_names))

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.set_facecolor((0.97, 0.97, 0.97))
    for spine in ax.spines.values():
        spine.set_visible(False)

    for di, day in enumerate(days):
        means, stds = [], []
        for fn in formula_names:
            vals = src_dict.get(fn, {}).get(day, {}).get(metric, [])
            m, s = mean_std(vals)
            means.append(m)
            stds.append(s)
        ax.bar(
                x + di * width, means, yerr=stds, width=width,
                color=palette[di % len(palette)], edgecolor='white', linewidth=1.0,
                error_kw={'elinewidth': 1.1, 'ecolor': 'black', 'capsize': 3}, label=day
            )

    ax.set_title(metric, color='#2B4C7E')
    ax.set_xticks(x + width)
    ax.set_xticklabels(formula_names, rotation=40, ha="right")
    ax.set_ylabel(y_label)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', framealpha=0.85)

    suffix = 'AE' if y_label.lower().startswith('absolute') else 'RE'
    png_path = os.path.join(out_dir, f"{fig_name_prefix}_{suffix}_{metric}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[PLOT SAVED] {png_path}")

# =========================
# 表格：保存 + 友好打印
# =========================

def _format_mean_std(mean: float, std: float, decimals: int = 1, unit: str = "") -> str:
    if unit:
        return f"{mean:.{decimals}f}±{std:.{decimals}f} {unit}"
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"


def build_pretty_table(df_summary: pd.DataFrame) -> pd.DataFrame:
    """将 mean/std 合并为单列，便于人类阅读与论文粘贴。"""
    pretty_rows = []
    metrics = ["Electrolyte","Colloid","Water","Total"]
    for _, row in df_summary.iterrows():
        rec = {
            "Formula": row["Formula"],
            "Day": row["Day"],
        }
        for m in metrics:
            rec[f"AE_{m}"] = _format_mean_std(row[f"AE_{m}_mean"], row[f"AE_{m}_std"], decimals=1, unit="mL")
            rec[f"RE_{m}"] = _format_mean_std(row[f"RE_{m}_mean"], row[f"RE_{m}_std"], decimals=2, unit="%")
        pretty_rows.append(rec)
    pretty_df = pd.DataFrame(pretty_rows)
    # 排序以保持 Day1, Day2, Total 顺序
    day_order = {"Day1":0, "Day2":1, "Total":2}
    pretty_df = pretty_df.sort_values(by=["Formula","Day"], key=lambda s: s.map(lambda x: day_order.get(x, 99)))
    return pretty_df


def save_and_print_summary(df_summary: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # 1) 原始均值±标准差数值表（CSV）
    out_csv = os.path.join(out_dir, "effectiveness_summary_mean_std.csv")
    df_summary.to_csv(out_csv, index=False)
    print(f"[DONE] Saved summary table (numeric): {out_csv}")

    # 2) 终端友好打印 + Markdown + XLSX（合并显示 mean±std）
    pretty_df = build_pretty_table(df_summary)

    # 终端打印（限制列宽可读）
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 180):
        print("\n========== Effectiveness Summary (mean±std) ==========")
        print(pretty_df.to_string(index=False))
        print("====================================================\n")

    # 保存 Markdown
    out_md = os.path.join(out_dir, "effectiveness_summary_mean_std.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(pretty_df.to_markdown(index=False))
    print(f"[DONE] Saved pretty summary (Markdown): {out_md}")

    # 保存 Excel
    out_xlsx = os.path.join(out_dir, "effectiveness_summary_mean_std.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        df_summary.to_excel(writer, index=False, sheet_name="numeric")
        pretty_df.to_excel(writer, index=False, sheet_name="pretty")
    print(f"[DONE] Saved summary workbook (XLSX): {out_xlsx}")


# =========================
# 主流程
# =========================

def main():
    ap = argparse.ArgumentParser("Effectiveness Validation + Styled ULTRA Plots")

    # 默认路径
    ap.add_argument("--ts_pkl",   type=str, default="/baksv/CIGIT/GXN_Liuxy/fluid/data/output_data.pkl",
                    help="path to output_data.pkl")
    ap.add_argument("--base_pkl", type=str, default="/baksv/CIGIT/GXN_Liuxy/fluid/data/baseline.pkl",
                    help="path to baseline.pkl")
    ap.add_argument("--pred_dir", type=str, default="/baksv/CIGIT/GXN_Liuxy/fluid/per_patient_outputs",
                    help="dir of {pid}_timeseries.csv")
    ap.add_argument("--out_dir",  type=str, default="/baksv/CIGIT/GXN_Liuxy/fluid/results_plus",
                    help="output dir")

    ap.add_argument("--eval_tail_ratio", type=float, default=1.0,
                    help="仅用后 N% 病人评估（0.2 → 后 20%）")
    ap.add_argument("--step_hours", type=float, default=1.0,
                    help="每步小时数（速率单位mL/h时：1步=1h→1.0；1步=10min→10/60）")

    ap.add_argument("--formula_water_strategy", type=str, default="midpoint",
                    choices=["lower","midpoint","upper","anchor"],
                    help="区间水量处理策略")
    ap.add_argument("--anchor_json", type=str, default=None,
                    help="可选：{formula:{Day1:xx, Day2:xx}} 每天水量锚点（mL）")

    ap.add_argument("--denom_floor", type=float, default=500.0,
                    help="相对误差(%)分母地板，避免 GT 极小导致夸大")
    ap.add_argument("--model_thresholds", type=float, nargs=3, default=[0.5,0.5,0.5],
                    help="模型门控阈值 [crystal, colloid, water]")

    # 主题参数
    ap.add_argument("--palette", type=str, default='okabe',
                    choices=['okabe','tableau','brewer_blues','brewer_pubu','mono_blue'],
                    help="科研调色板：okabe/tableau/brewer_blues/brewer_pubu/mono_blue")

    # 是否输出单指标分图
    ap.add_argument("--plot_metric_figs", action="store_true",
                    help="额外输出每个指标的分图（AE/RE 各一张）")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "Fig")
    os.makedirs(fig_dir, exist_ok=True)

    # 读数据
    Ts_data = load_pickle(args.ts_pkl)
    Base_data = load_pickle(args.base_pkl)

    # 统一 pid 键
    Ts_data = {norm_pid(k): v for k, v in Ts_data.items()}
    Base_data = {norm_pid(k): v for k, v in Base_data.items()}

    # 选择评估病人：交集 + tail_ratio
    ids = sorted(set(Ts_data.keys()) & set(Base_data.keys()))
    if not ids:
        print("[ERR] No patient ID intersection between Ts and Base.")
        return
    n_eval = max(1, int(round(len(ids) * float(args.eval_tail_ratio))))
    eval_ids = ids[-n_eval:]
    print(f"[INFO] Eval patients: {len(eval_ids)}")

    # anchor
    anchor = None
    if args.anchor_json and os.path.exists(args.anchor_json):
        with open(args.anchor_json, "r", encoding="utf-8") as f:
            anchor = json.load(f)

    # 误差池：公式 + Model
    ae_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    re_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    missing_model = 0
    for pid in eval_ids:
        w, tbsa = get_weight_tbsa(Base_data[pid])
        if w is None or tbsa is None:
            continue
        gt = compute_gt_day_volumes(Ts_data[pid], step_hours=args.step_hours)
        if gt is None:
            continue
        model_csv = os.path.join(args.pred_dir, f"{pid}_timeseries.csv")
        model_day = compute_model_day_volumes(model_csv, step_hours=args.step_hours, thresholds=args.model_thresholds)
        if model_day is None:
            missing_model += 1
            continue

        day_keys = ["Day1","Day2","Total"]
        est_cache = {"Model": model_day}
        for fn in FORMULAS:
            est_cache[fn] = {}
            for day in day_keys:
                est_cache[fn][day] = formula_day_volumes(fn, day, w, tbsa, args.formula_water_strategy, anchor)

        for fn in list(FORMULAS.keys()) + ["Model"]:
            for day in day_keys:
                for m in ["Electrolyte","Colloid","Water","Total"]:
                    est = est_cache[fn][day][m]
                    gtv = gt[day][m]
                    ae = abs(est - gtv)
                    denom = max(gtv, args.denom_floor)
                    re = 100.0 * ae / denom
                    ae_dict[fn][day][m].append(ae)
                    re_dict[fn][day][m].append(re)

    if missing_model > 0:
        print(f"[WARN] Missing model CSVs for {missing_model} patients (no *_timeseries.csv).")

    # 导出均值±标准差的表（数值）
    out_rows = []
    for fn in list(FORMULAS.keys()) + ["Model"]:
        for day in ["Day1","Day2","Total"]:
            row = {"Formula": fn, "Day": day}
            for m in ["Electrolyte","Colloid","Water","Total"]:
                ae_m, ae_s = mean_std(ae_dict[fn][day][m])
                re_m, re_s = mean_std(re_dict[fn][day][m])
                row[f"AE_{m}_mean"] = ae_m
                row[f"AE_{m}_std"]  = ae_s
                row[f"RE_{m}_mean"] = re_m
                row[f"RE_{m}_std"]  = re_s
            out_rows.append(row)
    df_summary = pd.DataFrame(out_rows)

    # —— 新增：多格式保存 + 终端打印 ——
    save_and_print_summary(df_summary, args.out_dir)

    # ===== 绘图 =====
    # 总超图（AE）
    ultra_plot_allinone(
        ae_dict, re_dict, fig_dir,
        fig_name_prefix="ULTRA",
        y_label="Absolute Error (mL)",
        title_suffix="Formulas + Model vs GT (Day1/Day2/Total)",
        palette_name=args.palette,
    )

    # 总超图（RE）
    ultra_plot_allinone(
        ae_dict, re_dict, fig_dir,
        fig_name_prefix="ULTRA",
        y_label="Relative Error (%)",
        title_suffix="Formulas + Model vs GT (Day1/Day2/Total)",
        palette_name=args.palette,
    )

    # 分图（可选）
    if args.plot_metric_figs:
        metrics = ["Electrolyte", "Colloid", "Water", "Total"]
        for metric in metrics:
            plot_metric_figure(ae_dict, fig_dir, metric, fig_name_prefix="ULTRA", y_label="Absolute Error (mL)", palette_name=args.palette)
        for metric in metrics:
            plot_metric_figure(re_dict, fig_dir, metric, fig_name_prefix="ULTRA", y_label="Relative Error (%)", palette_name=args.palette)

    print("[ALL DONE]")


if __name__ == "__main__":
    main()
