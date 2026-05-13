#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effectiveness Validation (Formulas + Model)
- 输出：总超图（四宫格）+ 可选单指标分图 + 汇总表格（CSV / XLSX / Markdown / 终端友好打印）
- 科研级主题：
    * 字体：Cambria（自动回退 Times New Roman / DejaVu Serif / SimSun）
    * 颜色：提供多套 色盲友好 科研配色（Okabe-Ito / Tableau 10 子集 / ColorBrewer 等）
    * 细节：半透明柱、深色误差线、浅灰网格、白色边框

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

from __future__ import annotations

import os
import json
import pickle
import argparse
from collections import defaultdict, OrderedDict
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.compat import pickle_compat

# =========================
# 配置：经验公式（Day1/Day2）
# =========================
K = 1000.0
FORMULAS = OrderedDict({
    "Evans":    {"Day1": (1.0, 1.0, 2*K),      "Day2": (0.5, 0.5, 2*K)},
    "Brooke":   {"Day1": (1.5, 0.5, 2*K),      "Day2": (0.75,0.25,2*K)},
    "Parkland": {"Day1": (4.0, 0.0, 0.0),      "Day2": (0.0, 1.25, 2*K)},  # 修复：Day2水改为2*K
    "Monafo":   {"Day1": (2.0, 0.0, 0.0),      "Day2": (1.0, 0.0, 0.0)},
    "TMMU":     {"Day1": (1.0, 0.5, 2*K),      "Day2": (0.5, 0.25, 2*K)},
    "RJH":      {"Day1": (0.75,0.75,(3*K,4*K)),"Day2":(0.375,0.375,(3*K,4*K))},
    "PLA-304F": {"Day1": (0.95,0.95,(3*K,4*K)),"Day2":(0.725,0.725,3*K)},
    "TMMU-DRF": {"Day1": (1.3, 1.3, 2*K),      "Day2": (0.5, 0.25, 2*K)},
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
    anchor: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """计算公式的液体量，跳过结果为0的计算"""
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
    
    # 初始化所有成分
    elec = 0.0
    coll = 0.0
    water = 0.0
    
    # 计算电解质（如果系数不为0）
    if a_e != 0:
        elec = float(a_e) * float(weight) * float(tbsa)
    
    # 计算胶体（如果系数不为0）
    if a_c != 0:
        coll = float(a_c) * float(weight) * float(tbsa)
    
    # 计算水
    if isinstance(w_rng, (tuple, list)) and len(w_rng) == 2:
        low, high = float(w_rng[0]), float(w_rng[1])
        if low == 0 and high == 0:
            water = 0.0
        elif water_strategy == "lower":
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
def norm_pid(pid: Any) -> str:
    pid = str(pid)
    return pid[:-4] if pid.endswith(".csv") else pid

def get_weight_tbsa(base_entry: Any) -> Tuple[Optional[float], Optional[float]]:
    w = tbsa = None
    if isinstance(base_entry, dict):
        for k in ["weight","Weight","wt","WT"]:
            if k in base_entry:
                w = base_entry[k]; break
        for k in ["tbsa","TBSA","burn_area","BSA"]:
            if k in base_entry:
                tbsa = base_entry[k]; break
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

def compute_gt_day_volumes(ts_entry: Dict[str, Any], step_hours: float = 1.0):
    if ts_entry is None or "label" not in ts_entry:
        return None
    if hasattr(ts_entry["label"], "values"):
        arr = np.asarray(ts_entry["label"].values, dtype=float)
    else:
        arr = np.asarray(ts_entry["label"], dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 6:
        return None

    T = arr.shape[0]
    inst = arr[:, 0:3] * arr[:, 3:6] * float(step_hours)  # (bit * speed * hours) -> 当步体积

    def seg_sum(start: int, end: int):
        if T == 0 or start >= T:
            return np.zeros(3, dtype=float)
        end = min(end, T - 1)
        if end < start:
            return np.zeros(3, dtype=float)
        return inst[start:end + 1, :].sum(axis=0)

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

def _pick_col(df: pd.DataFrame, new_name: str, old_name: str, default: Optional[np.ndarray] = None) -> np.ndarray:
    if new_name in df.columns:
        return df[new_name].to_numpy()
    if old_name in df.columns:
        return df[old_name].to_numpy()
    if default is not None:
        return default
    raise KeyError(f"缺失列: {new_name} 或 {old_name}")

def compute_model_day_volumes(csv_path: str, step_hours: float = 1.0, thresholds: Tuple[float,float,float]=(0.5,0.5,0.5)):
    if not os.path.exists(csv_path):
        return None
    # 读取 CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, engine="python")

    # 准备概率与速率
    # 概率缺失时默认 0；速率缺失时默认 0，保证健壮
    zeros = np.zeros(len(df), dtype=float)
    pr = np.stack([
        df["prob_crystal"].to_numpy() if "prob_crystal" in df.columns else zeros,
        df["prob_colloid"].to_numpy() if "prob_colloid" in df.columns else zeros,
        df["prob_water"].to_numpy()   if "prob_water"   in df.columns else zeros
    ], axis=1)

    sp = np.stack([
        _pick_col(df, "pred_speed_raw_crystal", "pred_speed_crystal", default=zeros),
        _pick_col(df, "pred_speed_raw_colloid", "pred_speed_colloid", default=zeros),
        _pick_col(df, "pred_speed_raw_water",   "pred_speed_water",   default=zeros),
    ], axis=1)

    thr = np.asarray(thresholds, dtype=float).reshape(1, 3)
    inst = (pr > thr).astype(float) * sp * float(step_hours)

    N = inst.shape[0]
    def seg_sum(start: int, end: int):
        if N == 0 or start >= N:
            return np.zeros(3, dtype=float)
        end = min(end, N - 1)
        if end < start:
            return np.zeros(3, dtype=float)
        return inst[start:end + 1, :].sum(axis=0)

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

def mean_std(a: List[float]) -> Tuple[float, float]:
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
        return [
            (0/255, 114/255, 178/255, 0.95),   # 蓝
            (130/255, 99/255, 170/255, 0.95), # 紫
            (0/255, 158/255, 115/255, 0.95),  # 绿
        ]
    if name == 'tableau':
        return [
            (31/255, 119/255, 180/255, 0.95),
            (148/255, 103/255, 189/255, 0.95),
            (23/255, 190/255, 207/255, 0.95),
        ]
    if name == 'brewer_blues':
        return [
            (8/255, 48/255, 107/255, 0.95),
            (49/255,130/255,189/255, 0.95),
            (158/255,202/255,225/255,0.95),
        ]
    if name == 'brewer_pubu':
        return [
            (2/255, 56/255, 88/255,  0.95),
            (44/255,123/255,182/255, 0.95),
            (158/255,188/255,218/255,0.95),
        ]
    if name == 'mono_blue':
        return [
            (0.16, 0.34, 0.64, 0.95),
            (0.28, 0.54, 0.86, 0.95),
            (0.60, 0.78, 1.00, 0.95),
        ]
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
    """绘制总超图，自动跳过空数据"""
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
                # 跳过空数据
                if not vals:
                    means.append(0)
                    stds.append(0)
                    continue
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
    """绘制单指标图，自动跳过空数据"""
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
            # 跳过空数据
            if not vals:
                means.append(0)
                stds.append(0)
                continue
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

def split_patient_ids_short(Ts_data: Dict[str, Any], splits=(0.7,0.15,0.15), seed=12, min_hours=48.0, orig_step_hours=1.0):
    """不依赖 DataLoader 的轻量划分；验证/测试过滤出 ≥48h 的病人"""
    import random
    rng = random.Random(seed)

    def getpid(k: str, r: Any) -> str:
        if isinstance(r, dict) and 'patient_id' in r:
            return str(r['patient_id'])
        return k.split('_')[0].split('.')[0]

    p2k: Dict[str, List[str]] = {}
    for k in Ts_data:
        p2k.setdefault(getpid(k, Ts_data[k]), []).append(k)

    ps = list(p2k)
    rng.shuffle(ps)
    n = len(ps)
    a = int(round(splits[0] * n))
    b = int(round(splits[1] * n))
    tr, va, te = ps[:a], ps[a:a+b], ps[a+b:]

    need = int(min_hours / orig_step_hours + 1e-9)

    def ok(pid: str) -> bool:
        for k in p2k[pid]:
            try:
                tlen = min(len(Ts_data[k]['tdata']), len(Ts_data[k]['label']))
            except Exception:
                tlen = 0
            if tlen >= need:
                return True
        return False

    return tr, [p for p in va if ok(p)], [p for p in te if ok(p)]

import numpy as np

def detect_weight_outliers(weights):
    """
    检测和处理体重异常值
    """
    weights_array = np.array(weights)
    
    print("=== 异常值检测 ===")
    
    # 临床合理范围
    clinical_min = 2.5  # 成人最低合理体重
    clinical_max = 120 # 成人最高合理体重
    
    # 检测异常值
    outliers_low = weights_array[weights_array < clinical_min]
    outliers_high = weights_array[weights_array > clinical_max]
    
    print(f"临床合理范围: {clinical_min}-{clinical_max} kg")
    print(f"过低体重 (<{clinical_min}kg): {len(outliers_low)} 个,{outliers_low}")
    print(f"过高体重 (>{clinical_max}kg): {len(outliers_high)} 个")
    
    if len(outliers_low) > 0:
        print(f"  具体值: {sorted(outliers_low)}")
    if len(outliers_high) > 0:
        print(f"  具体值: {sorted(outliers_high)}")
    
    # 处理异常值（过滤）
    valid_weights = weights_array[(weights_array >= clinical_min) & (weights_array <= clinical_max)]
    
    if len(valid_weights) > 0:
        print(f"\n=== 处理后结果 ===")
        print(f"有效数据: {len(valid_weights)}/{len(weights)}")
        print(f"体重范围: {valid_weights.min():.1f} - {valid_weights.max():.1f} kg")
        print(f"平均值: {valid_weights.mean():.2f} kg")
        print(f"中位数: {np.median(valid_weights):.2f} kg")
    
    return valid_weights

def comprehensive_anomaly_detection(Base_data):
    """
    综合异常检测，包含数据逻辑验证
    """
    fields = {
        0: "性别", 1: "体重", 2: "身高", 3: "年龄", 4: "BMI",
        5: "TBSA", 6: "三度烧伤", 7: "深二度烧伤", 8: "浅二度烧伤", 9: "吸入性损伤"
    }
    
    # 收集数据
    all_data = {i: [] for i in range(10)}
    patient_ids = list(Base_data.keys())
    
    for pid, patient_data in Base_data.items():
        for i in range(min(10, len(patient_data))):
            all_data[i].append(patient_data[i])
    
    print("=== 综合异常检测 ===\n")
    
    total_anomalies = 0
    
    # 1. 范围异常检测
    print("1. 范围异常:")
    ranges = {
        1: (1.0, 120), 2: (30, 220), 3: (0, 120), 4: (10, 60),
        5: (0, 100), 6: (0, 100), 7: (0, 100), 8: (0, 100), 9: (0, 1)
    }
    
    for field, (min_val, max_val) in ranges.items():
        data = all_data[field]
        outliers = [val for val in data if val < min_val or val > max_val]
        if outliers:
            print(f"   {fields[field]}: {len(outliers)}个异常")
            total_anomalies += len(outliers)
    
    # 2. 逻辑一致性检查
    print("\n2. 逻辑一致性检查:")
    
    # 检查烧伤面积总和是否等于TBSA
    burn_mismatch = 0
    for i, pid in enumerate(patient_ids):
        data = Base_data[pid]
        if len(data) >= 9:
            tbsa = data[5]
            burn_sum = data[6] + data[7] + data[8]
            if abs(tbsa - burn_sum) > 1:  # 允许1%的误差
                burn_mismatch += 1
    if burn_mismatch > 0:
        print(f"   烧伤面积不匹配: {burn_mismatch}例")
        total_anomalies += burn_mismatch
    
    # 3. BMI计算验证
    print("\n3. BMI计算验证:")
    bmi_errors = 0
    for i, pid in enumerate(patient_ids):
        data = Base_data[pid]
        if len(data) >= 5:
            weight = data[1]
            height = data[2] / 100  # cm转m
            calculated_bmi = weight / (height ** 2) if height > 0 else 0
            recorded_bmi = data[4]
            if abs(calculated_bmi - recorded_bmi) > 1:  # 允许1的误差
                bmi_errors += 1
    if bmi_errors > 0:
        print(f"   BMI计算错误: {bmi_errors}例")
        total_anomalies += bmi_errors
    
    print(f"\n=== 总结 ===")
    print(f"总异常数: {total_anomalies}")
    print(f"总记录数: {len(patient_ids)}")
    print(f"数据质量: {(1 - total_anomalies/len(patient_ids)/10) * 100:.1f}%")



def main():
    ap = argparse.ArgumentParser("Effectiveness Validation + Styled ULTRA Plots")
    ap.add_argument("--ts_pkl",   type=str, required=False, default="/baksv/CIGIT/GXN_Liuxy/Fluid_Infusion_Decision/data/output_data.pkl")
    ap.add_argument("--base_pkl", type=str, required=False, default="/baksv/CIGIT/GXN_Liuxy/Fluid_Infusion_Decision/data/baseline.pkl")
    ap.add_argument("--pred_dir", type=str, required=False, default="/baksv/CIGIT/GXN_Liuxy/Fluid_Infusion_Decision/per_patient_outputs")
    ap.add_argument("--out_dir",  type=str, required=False, default="/baksv/CIGIT/GXN_Liuxy/Fluid_Infusion_Decision/results_plus")
    ap.add_argument("--eval_tail_ratio", type=float, default=1.0)
    ap.add_argument("--step_hours", type=float, default=1.0)  # 每步小时数
    ap.add_argument("--formula_water_strategy", type=str, default="midpoint", choices=["lower","midpoint","upper","anchor"])
    ap.add_argument("--anchor_json", type=str, default=None)  # 可选锚点 json
    ap.add_argument("--denom_floor", type=float, default=500.0)
    ap.add_argument("--model_thresholds", type=float, nargs=3, default=[0.5,0.5,0.5])
    ap.add_argument("--palette", type=str, default='okabe', choices=['okabe','tableau','brewer_blues','brewer_pubu','mono_blue'])
    ap.add_argument("--plot_metric_figs", action="store_true")
    ap.add_argument("--seed", type=int, default=12)
    args = ap.parse_args()

    # 读取数据（PKL）
    if not os.path.exists(args.ts_pkl):
        raise FileNotFoundError(f"时间序列 pkl 未找到: {args.ts_pkl}")
    if not os.path.exists(args.base_pkl):
        raise FileNotFoundError(f"基线特征 pkl 未找到: {args.base_pkl}")
    with open(args.ts_pkl, 'rb') as f:
        Ts_data = pickle.load(f)
    with open(args.base_pkl, 'rb') as f:
        Base_data = pickle.load(f)
    comprehensive_anomaly_detection(Base_data)
    
    weights = []    
    urine_ml_h = []
    path =  '/baksv/CIGIT/GXN_Liuxy/Fluid_Infusion_Decision/data/data_speed/vital_signs'
    for patient_id, patient_data in Base_data.items():
        weights.append(patient_data[3])
        import pandas as pd
        pd = pd.read_csv(os.path.join(path, f"{patient_id}.csv"))
        urine_data = pd['尿量'].mean()
        if urine_data == None:
            print(f"{patient_id} 无尿量数据")
            continue
        urine_ml_h.append(urine_data)
    _ = detect_weight_outliers(weights)
    weights_array = np.array(weights)   

    print(len(urine_ml_h),len(weights))
    # 计算标准化尿量
    urine_ml_kg_h =[]
    for urine, weight in zip(urine_ml_h, weights_array):
        print(urine, weight)
        urine_ml_kg_h.append(urine / weight if weight > 0 else 0.0) 
        # 删除0
    urine_ml_kg_h = np.array([v for v in urine_ml_kg_h if v > 0])

    if urine_ml_kg_h.size > 0:
        print(f"Urine Output: {np.median(urine_ml_kg_h):.2f}, "
            f"({np.percentile(urine_ml_kg_h, 25):.2f}, {np.percentile(urine_ml_kg_h, 75):.2f})")
    else:
        print("⚠️ No valid weight values (>0), cannot calculate urine output.")

    print("=== 详细体重分析 ===")
    print(f"总患者数: {len(weights)}")
    print(f"体重范围: {weights_array.min():.1f} - {weights_array.max():.1f} kg")
    print(f"平均值: {weights_array.mean():.2f} kg")
    print(f"中位数: {np.median(weights_array):.2f} kg")
    print(f"标准差: {weights_array.std():.2f} kg")
    print(f"25%分位数: {np.percentile(weights_array, 25):.2f} kg")
    print(f"75%分位数: {np.percentile(weights_array, 75):.2f} kg")    
    # --- 读取 anchor（如指定且存在） ---
    exit()
    anchor = None
    if args.anchor_json:
        if not os.path.exists(args.anchor_json):
            raise FileNotFoundError(f"anchor_json 不存在：{args.anchor_json}")
        with open(args.anchor_json, "r", encoding="utf-8") as f:
            anchor = json.load(f)

    # --- 病人划分；验证/测试集都保证 ≥48h ---
    train_ids, val_ids, test_ids = split_patient_ids_short(
        Ts_data, splits=(0.7,0.15,0.15), seed=args.seed,
        min_hours=48.0, orig_step_hours=args.step_hours
    )
    print(f"[INFO] Total patients: {len(train_ids)+len(val_ids)+len(test_ids)} | "
          f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    # --- 评估ID：用"测试集"按尾部比例抽取（不足则并入验证集尾部） ---
    pool = test_ids + val_ids
    n_eval = max(1, int(round(len(pool) * float(args.eval_tail_ratio))))
    eval_ids = pool[-n_eval:] if pool else []

    # --- 输出与作图目录 ---
    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    # --- 误差池 ---
    ae_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    re_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    missing_model = 0
    for pid in eval_ids:
        base = Base_data.get(pid) or Base_data.get(norm_pid(pid)) or Base_data.get(str(pid))
        if base is None:
            # 尝试：有些数据主键是文件名，需要遍历一次
            candidates = [k for k in Base_data.keys() if norm_pid(k) == norm_pid(pid)]
            base = Base_data[candidates[0]] if candidates else None
        if base is None:
            continue

        w, tbsa = get_weight_tbsa(base)
        if w is None or tbsa is None:
            continue

        ts = Ts_data.get(pid) or Ts_data.get(norm_pid(pid)) or Ts_data.get(str(pid))
        if ts is None:
            candidates = [k for k in Ts_data.keys() if norm_pid(k) == norm_pid(pid)]
            ts = Ts_data[candidates[0]] if candidates else None
        if ts is None:
            continue

        gt = compute_gt_day_volumes(ts, step_hours=args.step_hours)
        if gt is None:
            continue

        model_csv = os.path.join(args.pred_dir, f"{norm_pid(pid)}_timeseries.csv")
        model_day = compute_model_day_volumes(model_csv, step_hours=args.step_hours, thresholds=tuple(args.model_thresholds))
        if model_day is None:
            missing_model += 1
            continue

        day_keys = ["Day1","Day2","Total"]
        est_cache = {"Model": model_day}
        for fn in FORMULAS:
            est_cache[fn] = {day: formula_day_volumes(fn, day, w, tbsa, args.formula_water_strategy, anchor) for day in day_keys}

        for fn in list(FORMULAS.keys()) + ["Model"]:
            for day in day_keys:
                for m in ["Electrolyte","Colloid","Water","Total"]:
                    est = est_cache[fn][day][m]
                    gtv = gt[day][m]
                    # 只有当估计值和真实值都不为0时才计算误差
                    if est != 0 and gtv != 0:
                        ae = abs(est - gtv)
                        denom = max(gtv, args.denom_floor)
                        re = 100.0 * ae / denom
                        ae_dict[fn][day][m].append(ae)
                        re_dict[fn][day][m].append(re)

    if missing_model > 0:
        print(f"[WARN] Missing model CSVs for {missing_model} patients (no *_timeseries.csv).")

    # --- 汇总表 ---
    out_rows = []
    for fn in list(FORMULAS.keys()) + ["Model"]:
        for day in ["Day1","Day2","Total"]:
            row = {"Formula": fn, "Day": day}
            for m in ["Electrolyte","Colloid","Water","Total"]:
                ae_vals = ae_dict[fn][day][m]
                re_vals = re_dict[fn][day][m]
                # 只有当有有效数据时才计算统计量
                if ae_vals:
                    ae_m, ae_s = mean_std(ae_vals)
                    re_m, re_s = mean_std(re_vals)
                else:
                    ae_m, ae_s, re_m, re_s = 0, 0, 0, 0
                row[f"AE_{m}_mean"] = ae_m; row[f"AE_{m}_std"]  = ae_s
                row[f"RE_{m}_mean"] = re_m; row[f"RE_{m}_std"]  = re_s
            out_rows.append(row)
    df_summary = pd.DataFrame(out_rows)
    save_and_print_summary(df_summary, args.out_dir)

    # --- 绘图（AE/RE 总超图 + 可选分图） ---
    ultra_plot_allinone(ae_dict, re_dict, fig_dir,
                        fig_name_prefix="ULTRA", y_label="Absolute Error (mL)",
                        title_suffix="Formulas + Model vs GT (Day1/Day2/Total)",
                        palette_name=args.palette)
    ultra_plot_allinone(ae_dict, re_dict, fig_dir,
                        fig_name_prefix="ULTRA", y_label="Relative Error (%)",
                        title_suffix="Formulas + Model vs GT (Day1/Day2/Total)",
                        palette_name=args.palette)

    if args.plot_metric_figs:
        for metric in ["Electrolyte","Colloid","Water","Total"]:
            plot_metric_figure(ae_dict, fig_dir, metric, fig_name_prefix="ULTRA", y_label="Absolute Error (mL)",  palette_name=args.palette)
        for metric in ["Electrolyte","Colloid","Water","Total"]:
            plot_metric_figure(re_dict, fig_dir, metric, fig_name_prefix="ULTRA", y_label="Relative Error (%)",   palette_name=args.palette)

    print("[ALL DONE]")

if __name__ == "__main__":
    main()