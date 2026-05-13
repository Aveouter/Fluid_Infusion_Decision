import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import chardet

#################################
# 0️⃣ 读取 CSV → 生成 fluid_data（入量） & vign_data（尿量）
#################################

fluid_path = "/baksv/CIGIT/GXN_Liuxy/Fluid_Infusion_Decision/data/data_speed/fluid_data"
urine_path = "/baksv/CIGIT/GXN_Liuxy/Fluid_Infusion_Decision/data/data_speed/vital_signs"   # ← 如果目录不同请改

fluid_data = {}
vign_data = {}

def read_csv_auto(path):
    """ 自动识别编码并读取 CSV """
    with open(path, 'rb') as f:
        encoding = chardet.detect(f.read(2000))['encoding']
    return pd.read_csv(path, encoding=encoding)

# ---- 读取入量 ----
for filename in os.listdir(fluid_path):
    if filename.endswith(".csv"):
        patient_id = filename.replace(".csv", "")
        df = read_csv_auto(os.path.join(fluid_path, filename))

        # 自动寻找列名（与前面统一）
        time_col = [c for c in df.columns if "时间" in c][0]
        volume_col = [c for c in df.columns if "入量" in c or "晶体" in c][0]

        fluid_data[patient_id] = list(zip(df[time_col], df[volume_col]))

# ---- 读取尿量 ----
for filename in os.listdir(urine_path):
    if filename.endswith(".csv"):
        patient_id = filename.replace(".csv", "")
        df = read_csv_auto(os.path.join(urine_path, filename))
        print(df)
        time_col = [c for c in df.columns if "时间" in c][0]
        urine_col = [c for c in df.columns if "尿" in c or "出量" in c][0]

        vign_data[patient_id] = list(zip(df[time_col], df[urine_col]))


#################################
# 1️⃣ 合并 fluid_data（入量） + vign_data（尿量）
#################################

merged_data = defaultdict(list)

for patient_id, records in fluid_data.items():
    for time, input_volume in records:
        merged_data[patient_id].append({
            "time": time,
            "input": input_volume,
            "urine": None
        })

for patient_id, records in vign_data.items():
    existing_times = [item["time"] for item in merged_data[patient_id]]
    
    for time, urine_volume in records:
        if time in existing_times:
            idx = existing_times.index(time)
            merged_data[patient_id][idx]["urine"] = urine_volume
        else:
            merged_data[patient_id].append({
                "time": time,
                "input": None,
                "urine": urine_volume
            })


#################################
# 2️⃣ 生成镜像柱状图 + 汇总结果
#################################

save_dir = "./fluid_balance_plots"
os.makedirs(save_dir, exist_ok=True)

summary = []

for patient_id, rows in merged_data.items():

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    df["input"] = df["input"].fillna(0)
    df["urine"] = df["urine"].fillna(0)

    df["urine_negative"] = -df["urine"]

    plt.figure(figsize=(14, 6))
    plt.bar(df["time"], df["input"], width=0.03, label="入量 (mL/h)", color="#4CAF50")
    plt.bar(df["time"], df["urine_negative"], width=0.03, label="尿量 (mL/h)", color="#2196F3")

    plt.axhline(0, color="black")
    plt.xticks(rotation=45)
    plt.ylabel("体积 (mL)")
    plt.title(f"患者 {patient_id} 液体平衡趋势图")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(os.path.join(save_dir, f"{patient_id}_balance.png"), bbox_inches="tight")
    plt.close()

    summary.append({
        "患者ID": patient_id,
        "总入量(mL)": df["input"].sum(),
        "总尿量(mL)": df["urine"].sum(),
        "净入量(mL)": df["input"].sum() - df["urine"].sum(),
        "记录小时数": len(df)
    })


#################################
# 3️⃣ 输出分析表
#################################

summary_df = pd.DataFrame(summary).sort_values("净入量(mL)", ascending=False)
summary_df.to_excel("Fluid_Balance_Summary.xlsx", index=False)

print("🎉 完成！")
print(f"📁 所有患者图像保存在: {save_dir}")
print("📄 汇总分析表 → Fluid_Balance_Summary.xlsx")
