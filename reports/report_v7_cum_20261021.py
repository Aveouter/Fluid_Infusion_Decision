"""Render the V7.3-CUM endpoint tables."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/experiments/old-event-v7-20261021"


def f(value, nd=1):
    return "—" if value is None else f"{value:,.{nd}f}"


def main():
    data = json.loads((REPORT / "cum_issue81_seed20261015.json").read_text(encoding="utf-8"))
    lines = ["# V7.3-CUM：患者级 24/48/72 h 累计损失（seed 20261015）", "",
             "fold 0 外折选择 `cumulative_weight=0.2`；下表为同一 seed 20261015 的 5 折 OOF（折1–4）与历史留出。",
             "V7.1/V7.1-M 也在同一 seed 上重算，避免与 3 种子主表混淆；3 种子 V7.1/V7.1-M 主结果见 README §3.1–3.2。",
             "零流、活动门控、动作头与解码未改；只改训练损失和 batch 组织（按患者整段）。", "",
             "| 数据 | 方法 | P1 24h | P1 48h | P1 72h | 设置 MAE | 变化小时 MAE | 严格 F1 | 活动 macro-F1 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ("development_folds1to4", "historical_test"):
        block = data["results"][split]
        for key, label in (("v71", "V7.1"), ("v71_mean", "V7.1-M"), ("v7_cum", "V7.3-CUM")):
            item = block[key]
            summary_key = key + "_summary"
            lines.append(f"| {split} | {label} | {f(item['cumulative']['24']['total_mae'])} | "
                         f"{f(item['cumulative']['48']['total_mae'])} | "
                         f"{f(item['cumulative']['72']['total_mae'])} | "
                         f"{f(item['secondary']['all_hours_mae'], 2)} | "
                         f"{f(block[summary_key]['change_hour_mae'], 2)} | "
                         f"{f(block[summary_key]['strict_action_f1'], 3)} | "
                         f"{f(item['activity']['macro_f1'], 3)} |")
    lines += ["", "| 数据 | V7.3−V7.1 P1 差 | V7.3−V7.1-M P1 差 | V7.3−V7.1 设置 MAE 差 |",
              "|---|---:|---:|---:|"]
    for split in ("development_folds1to4", "historical_test"):
        block = data["results"][split]
        lines.append(f"| {split} | {f(block['paired_vs_v71']['difference'])} "
                     f"[{f(block['paired_vs_v71']['ci_low'])}, {f(block['paired_vs_v71']['ci_high'])}] | "
                     f"{f(block['paired_vs_mean']['difference'])} "
                     f"[{f(block['paired_vs_mean']['ci_low'])}, {f(block['paired_vs_mean']['ci_high'])}] | "
                     f"{f(block['mae_vs_v71']['setting_mae']['difference'], 2)} "
                     f"[{f(block['mae_vs_v71']['setting_mae']['ci_low'], 2)}, "
                     f"{f(block['mae_vs_v71']['setting_mae']['ci_high'], 2)}] |")
    lines += ["", "P1 是患者级三液体总量绝对误差；设置 MAE/严格 F1 来自同一 OOF/留出预测。V7.3 是 P1 导向的 Pareto 点，不替换 V7.1 作为总体/动作主模型；它是在看到 fold 0 后选择的，历史留出是独立检查。", ""]
    (REPORT / "v7_cum.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT / 'v7_cum.md'}")


if __name__ == "__main__":
    main()
