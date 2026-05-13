#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import numpy as np

# ================== 配置 ==================
input_ts = "/baksv/CIGIT/GXN_Liuxy/Fluid_Infusion_Decision/data/output_data.pkl"
input_base = "/baksv/CIGIT/GXN_Liuxy/Fluid_Infusion_Decision/data/baseline.pkl"
output_excel = "Fluid_AE_RE_and_Balance_full.xlsx"

ADULT_MIN_AGE = 18
K = 1000.0

# ====== RE 控制参数（你主要调这里） ======
# 是否用 predicted 做 RE 的分母（True 更稳定，RE 会更小更平滑）
USE_PRED_FOR_DENOM = True

# mL·kg^-1 的最小参考值（避免分母太小导致 RE 爆炸）
FLOOR_ML_KG = 30.0

# mL·kg^-1·TBSA^-1 的最小参考值
FLOOR_ML_KG_TBSA = 3.0

# RE 是否以百分比形式输出（True => ×100 后的百分数；False => 原始比值）
RE_AS_PERCENT = True

# 尿量列名，如果不一样在这里改
URINE_COL = "尿量"

# 经验公式： (晶体系数, 胶体系数, 水量[mL 或 (下限,上限)])
FORMULAS = {
    "Evans":      {"Day1": (1.0,   1.0,   2 * K), "Day2": (0.5,   0.5,   2 * K)},
    "Brooke":     {"Day1": (1.5,   0.5,   2 * K), "Day2": (0.75,  0.25,  2 * K)},
    "Parkland":   {"Day1": (4.0,   0.0,   0.0),   "Day2": (0.0,   1.25,  2 * K)},
    "Monafo":     {"Day1": (2.0,   0.0,   0.0),   "Day2": (1.0,   0.0,   0.0)},
    "TMMU":       {"Day1": (1.0,   0.5,   2 * K), "Day2": (0.5,   0.25,  2 * K)},
    "RJH":        {"Day1": (0.75,  0.75,  (3 * K, 4 * K)),
                   "Day2": (0.375, 0.375, (3 * K, 4 * K))},
    "PLA-304F":   {"Day1": (0.95,  0.95,  (3 * K, 4 * K)),
                   "Day2": (0.725, 0.725, 3 * K)},
    "TMMU-DRF":   {"Day1": (1.3,   1.3,   2 * K), "Day2": (0.5,   0.25,  2 * K)},
}

DAYS    = ["Day1", "Day2", "Total"]
METRICS = ["Crystalloid", "Colloid", "Water"]

# ================== 读取数据 ==================

ts_data = pickle.load(open(input_ts, "rb"))
base_data = pickle.load(open(input_base, "rb"))

# 记录：
records_ae_re = []      # AE / RE 明细（按液体、按天）
records_balance = []    # 出入量明细（按天）

# ================== 主循环 ==================

for pid_raw, patient in ts_data.items():
    pid = str(pid_raw).replace(".csv", "")

    if pid not in base_data:
        continue

    base_entry = base_data[pid]
    # 假设 baseline.pkl 结构：0性别,1体重,2身高,3年龄,4BMI,5TBSA
    age    = float(base_entry[3])
    weight = float(base_entry[1])
    tbsa   = float(base_entry[5])

    if age < ADULT_MIN_AGE:
        continue

    # TBSA：如果 <5，当作比例 0.45 → 45
    if tbsa < 5:
        tbsa = tbsa * 100.0

    # 防止极端情况 TBSA<=0
    if tbsa <= 0 or weight <= 0:
        continue

    # 分组标签
    if tbsa < 20:
        tbsa_group = "<20%"
    elif tbsa < 40:
        tbsa_group = "20-39%"
    else:
        tbsa_group = ">=40%"

    if weight < 60:
        weight_group = "<60kg"
    elif weight < 80:
        weight_group = "60-79kg"
    else:
        weight_group = ">=80kg"

    if age < 40:
        age_group = "<40y"
    elif age < 60:
        age_group = "40-59y"
    else:
        age_group = ">=60y"

    df = pd.DataFrame(patient["label"])

    # ================== 实际输入量 ==================
    actual = {"Day1":{}, "Day2":{}, "Total":{}}
    c1 = c2 = o1 = o2 = w1 = w2 = 0.0

    # 跳过第 0 行，从第 1 行起按小时
    for hour, (idx, row) in enumerate(df.iloc[1:].iterrows(), start=1):
        # 假设晶体/胶体/水速度列已经是每小时体积(mL/h)，按 1 小时积分
        vc = float(row["晶体速度"])
        vo = float(row["胶体速度"])
        vw = float(row["水速度"])

        if hour <= 24:
            c1 += vc; o1 += vo; w1 += vw
        elif hour <= 48:
            c2 += vc; o2 += vo; w2 += vw
        else:
            pass

    actual["Day1"]  = {"Crystalloid": c1,         "Colloid": o1,         "Water": w1}
    actual["Day2"]  = {"Crystalloid": c2,         "Colloid": o2,         "Water": w2}
    actual["Total"] = {"Crystalloid": c1 + c2,    "Colloid": o1 + o2,    "Water": w1 + w2}

    # ================== 实际尿量 ==================
    u1 = u2 = 0.0
    if URINE_COL in df.columns:
        for hour, (idx, row) in enumerate(df.iloc[1:].iterrows(), start=1):
            u = float(row[URINE_COL])
            if hour <= 24:
                u1 += u
            elif hour <= 48:
                u2 += u
    else:
        # 没有尿量列，全部置 0
        pass

    urine = {
        "Day1":  u1,
        "Day2":  u2,
        "Total": u1 + u2
    }

    # ================== 逐公式逐天计算 ==================
    for formula in FORMULAS:

        for day in DAYS:
            # ----- 公式预测液体量（mL） -----
            if day == "Total":
                pred = {m: 0.0 for m in METRICS}
                for d in ["Day1", "Day2"]:
                    cryst_coef, coll_coef, water_cfg = FORMULAS[formula][d]
                    pred["Crystalloid"] += cryst_coef * weight * tbsa
                    pred["Colloid"]     += coll_coef  * weight * tbsa
                    if isinstance(water_cfg, (tuple, list)):
                        pred["Water"] += (water_cfg[0] + water_cfg[1]) / 2.0
                    else:
                        pred["Water"] += float(water_cfg)
            else:
                cryst_coef, coll_coef, water_cfg = FORMULAS[formula][day]
                if isinstance(water_cfg, (tuple, list)):
                    water_val = (water_cfg[0] + water_cfg[1]) / 2.0
                else:
                    water_val = float(water_cfg)
                pred = {
                    "Crystalloid": cryst_coef * weight * tbsa,
                    "Colloid":     coll_coef  * weight * tbsa,
                    "Water":       water_val
                }

            # ----- AE/RE 记录（分液体，以 mL·kg^-1 和 mL·kg^-1·TBSA^-1 为单位） -----
            for metric in METRICS:
                a_val = actual[day][metric]   # mL
                p_val = pred[metric]          # mL

                # 跳过公式给 0 的情况（无此液体）
                if p_val == 0:
                    continue

                # ====== 单位转换 ======
                # 1) mL·kg^-1
                a_ml_kg = a_val / weight
                p_ml_kg = p_val / weight

                # 2) mL·kg^-1·TBSA^-1
                a_ml_kg_tbsa = a_val / (weight * tbsa)
                p_ml_kg_tbsa = p_val / (weight * tbsa)

                # ====== 误差 (mL·kg^-1) ======
                ae_ml_kg = abs(p_ml_kg - a_ml_kg)

                if USE_PRED_FOR_DENOM:
                    base_ml_kg = abs(p_ml_kg)
                else:
                    base_ml_kg = abs(a_ml_kg)

                denom_ml_kg = max(base_ml_kg, FLOOR_ML_KG)

                re_ml_kg_raw = ae_ml_kg / denom_ml_kg
                re_ml_kg = re_ml_kg_raw * 100.0 if RE_AS_PERCENT else re_ml_kg_raw

                # ====== 误差 (mL·kg^-1·TBSA^-1) ======
                ae_ml_kg_tbsa = abs(p_ml_kg_tbsa - a_ml_kg_tbsa)

                if USE_PRED_FOR_DENOM:
                    base_ml_kg_tbsa = abs(p_ml_kg_tbsa)
                else:
                    base_ml_kg_tbsa = abs(a_ml_kg_tbsa)

                denom_ml_kg_tbsa = max(base_ml_kg_tbsa, FLOOR_ML_KG_TBSA)

                re_ml_kg_tbsa_raw = ae_ml_kg_tbsa / denom_ml_kg_tbsa
                re_ml_kg_tbsa = re_ml_kg_tbsa_raw * 100.0 if RE_AS_PERCENT else re_ml_kg_tbsa_raw

                records_ae_re.append([
                    pid, formula, day, metric.lower(),
                    age, weight, tbsa, age_group, weight_group, tbsa_group,
                    a_val, p_val,                  # 原始 mL
                    a_ml_kg, p_ml_kg,              # mL·kg^-1
                    ae_ml_kg, re_ml_kg,            # AE/RE (mL·kg^-1)
                    a_ml_kg_tbsa, p_ml_kg_tbsa,    # mL·kg^-1·TBSA^-1
                    ae_ml_kg_tbsa, re_ml_kg_tbsa   # AE/RE (mL·kg^-1·TBSA^-1)
                ])

            # ----- 出入量记录（总量，仍用 mL） -----
            actual_input = actual[day]["Crystalloid"] + actual[day]["Colloid"] + actual[day]["Water"]
            formula_input = pred["Crystalloid"] + pred["Colloid"] + pred["Water"]
            actual_urine = urine[day]

            actual_balance  = actual_input  - actual_urine
            formula_balance = formula_input - actual_urine
            diff_balance    = formula_balance - actual_balance

            records_balance.append([
                pid, formula, day,
                age, weight, tbsa, age_group, weight_group, tbsa_group,
                actual_input, formula_input, actual_urine,
                actual_balance, formula_balance, diff_balance
            ])

# ================== 转成 DataFrame ==================

df_ae_re = pd.DataFrame(
    records_ae_re,
    columns=[
        "patient","formula","day","metric",
        "age","weight","tbsa","age_group","weight_group","tbsa_group",
        "actual_ml","predicted_ml",                 # 原始 mL
        "actual_ml_kg","predicted_ml_kg",           # mL·kg^-1
        "AE_ml_kg","RE_ml_kg",                      # 误差（mL·kg^-1）
        "actual_ml_kg_tbsa","predicted_ml_kg_tbsa", # mL·kg^-1·TBSA^-1
        "AE_ml_kg_tbsa","RE_ml_kg_tbsa"             # 误差（mL·kg^-1·TBSA^-1）
    ]
)

df_balance = pd.DataFrame(
    records_balance,
    columns=[
        "patient","formula","day",
        "age","weight","tbsa","age_group","weight_group","tbsa_group",
        "actual_input","formula_input","urine",
        "actual_balance","formula_balance","diff_balance"
    ]
)

# ================== 汇总：总体 AE/RE ==================
# 分别对 mL·kg^-1 和 mL·kg^-1·TBSA^-1 的 AE/RE 做统计

summary_ae_re = df_ae_re.groupby(["formula","day","metric"]).agg(
    AE_ml_kg_mean=("AE_ml_kg","mean"),
    AE_ml_kg_std=("AE_ml_kg","std"),
    RE_ml_kg_mean=("RE_ml_kg","mean"),
    RE_ml_kg_std=("RE_ml_kg","std"),

    AE_ml_kg_tbsa_mean=("AE_ml_kg_tbsa","mean"),
    AE_ml_kg_tbsa_std=("AE_ml_kg_tbsa","std"),
    RE_ml_kg_tbsa_mean=("RE_ml_kg_tbsa","mean"),
    RE_ml_kg_tbsa_std=("RE_ml_kg_tbsa","std"),

    n=("patient","count")
).round(2)

# ================== 汇总：总体 Fluid Balance ==================

summary_balance = df_balance.groupby(["formula","day"]).agg(
    actual_input_mean=("actual_input","mean"),
    formula_input_mean=("formula_input","mean"),
    urine_mean=("urine","mean"),
    actual_balance_mean=("actual_balance","mean"),
    formula_balance_mean=("formula_balance","mean"),
    diff_balance_mean=("diff_balance","mean"),
    diff_balance_std=("diff_balance","std"),
    n=("patient","count")
).round(2)

# ================== TBSA 分层：AE/RE & Balance ==================

summary_ae_re_tbsa = df_ae_re.groupby(["tbsa_group","formula","day","metric"]).agg(
    AE_ml_kg_mean=("AE_ml_kg","mean"),
    AE_ml_kg_std=("AE_ml_kg","std"),
    RE_ml_kg_mean=("RE_ml_kg","mean"),
    RE_ml_kg_std=("RE_ml_kg","std"),

    AE_ml_kg_tbsa_mean=("AE_ml_kg_tbsa","mean"),
    AE_ml_kg_tbsa_std=("AE_ml_kg_tbsa","std"),
    RE_ml_kg_tbsa_mean=("RE_ml_kg_tbsa","mean"),
    RE_ml_kg_tbsa_std=("RE_ml_kg_tbsa","std"),

    n=("patient","count")
).round(2)

summary_balance_tbsa = df_balance.groupby(["tbsa_group","formula","day"]).agg(
    diff_balance_mean=("diff_balance","mean"),
    diff_balance_std=("diff_balance","std"),
    n=("patient","count")
).round(2)

# ================== 体重分层 ==================

summary_ae_re_weight = df_ae_re.groupby(["weight_group","formula","day","metric"]).agg(
    AE_ml_kg_mean=("AE_ml_kg","mean"),
    AE_ml_kg_std=("AE_ml_kg","std"),
    RE_ml_kg_mean=("RE_ml_kg","mean"),
    RE_ml_kg_std=("RE_ml_kg","std"),

    AE_ml_kg_tbsa_mean=("AE_ml_kg_tbsa","mean"),
    AE_ml_kg_tbsa_std=("AE_ml_kg_tbsa","std"),
    RE_ml_kg_tbsa_mean=("RE_ml_kg_tbsa","mean"),
    RE_ml_kg_tbsa_std=("RE_ml_kg_tbsa","std"),

    n=("patient","count")
).round(2)

summary_balance_weight = df_balance.groupby(["weight_group","formula","day"]).agg(
    diff_balance_mean=("diff_balance","mean"),
    diff_balance_std=("diff_balance","std"),
    n=("patient","count")
).round(2)

# ================== 年龄分层 ==================

summary_ae_re_age = df_ae_re.groupby(["age_group","formula","day","metric"]).agg(
    AE_ml_kg_mean=("AE_ml_kg","mean"),
    AE_ml_kg_std=("AE_ml_kg","std"),
    RE_ml_kg_mean=("RE_ml_kg","mean"),
    RE_ml_kg_std=("RE_ml_kg","std"),

    AE_ml_kg_tbsa_mean=("AE_ml_kg_tbsa","mean"),
    AE_ml_kg_tbsa_std=("AE_ml_kg_tbsa","std"),
    RE_ml_kg_tbsa_mean=("RE_ml_kg_tbsa","mean"),
    RE_ml_kg_tbsa_std=("RE_ml_kg_tbsa","std"),

    n=("patient","count")
).round(2)

summary_balance_age = df_balance.groupby(["age_group","formula","day"]).agg(
    diff_balance_mean=("diff_balance","mean"),
    diff_balance_std=("diff_balance","std"),
    n=("patient","count")
).round(2)

# ================== 写入 Excel ==================

with pd.ExcelWriter(output_excel) as writer:
    df_ae_re.to_excel(writer, sheet_name="Raw_AE_RE", index=False)
    summary_ae_re.to_excel(writer, sheet_name="Summary_AE_RE")
    df_balance.to_excel(writer, sheet_name="Fluid_Balance_detail", index=False)
    summary_balance.to_excel(writer, sheet_name="Fluid_Balance_summary")

    summary_ae_re_tbsa.to_excel(writer, sheet_name="AE_RE_by_TBSA")
    summary_balance_tbsa.to_excel(writer, sheet_name="Balance_by_TBSA")

    summary_ae_re_weight.to_excel(writer, sheet_name="AE_RE_by_weight")
    summary_balance_weight.to_excel(writer, sheet_name="Balance_by_weight")

    summary_ae_re_age.to_excel(writer, sheet_name="AE_RE_by_age")
    summary_balance_age.to_excel(writer, sheet_name="Balance_by_age")

print("\n✔ 完成！已生成：", output_excel)
print("Sheets 包含：")
print("  Raw_AE_RE           : 每病人×公式×天×液体的 AE/RE（mL·kg^-1 和 mL·kg^-1·TBSA^-1）")
print("  Summary_AE_RE       : 总体 AE/RE（公式×天×液体，已分两种单位）")
print("  Fluid_Balance_detail: 每病人出入量明细（mL）")
print("  Fluid_Balance_summary: 总体出入量差值（公式×天，mL）")
print("  AE_RE_by_TBSA / Balance_by_TBSA")
print("  AE_RE_by_weight / Balance_by_weight")
print("  AE_RE_by_age / Balance_by_age")
