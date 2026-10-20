"""Render the registered old-data V7 aggregate into a public report."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AGGREGATE = ROOT / "docs/experiments/old-event-v7-20261021/aggregate.json"
REPORT = AGGREGATE.parent
PROTOCOL = ROOT / "docs/old_event_v7_protocol_20261021.json"
METHOD_LABELS = {
    "two_branch": "V7 双分支（绝对速率 + 事件/方向）",
    "absolute": "绝对速率对照（同骨干，无事件/动作头）",
    "persistence": "持续性（复制上一小时）",
    "ma3": "因果移动平均 k=3",
    "ma6": "因果移动平均 k=6",
    "zero": "恒 0",
    "train_median": "训练集中位数",
    "V7_two_branch": "V7 双分支",
    "original_BurnResu_rate": "原 BurnResu（epoch-15，rate 口径）",
    "moving_average_k3": "因果移动平均 k=3",
    "joint": "联合（生理 + 输液史）",
    "history_only": "仅输液史（匹配对照）",
    "variant": "V7 变体（hold=上一小时）",
    "prev_base": "hold=上一小时（普通事件 BCE）",
    "prev_neg15": "hold=上一小时（负事件权重 1.5）",
    "hold_absolute": "hold=绝对速率（当前 V7）",
    "V7_prev_base": "V7.1（hold=上一小时）",
    "V7_hold_absolute": "V7（hold=绝对速率）",
}
FLUID_LABELS = {"crystal": "晶体", "colloid": "胶体", "water": "水"}
CLASS_LABELS = {"hold": "保持", "start": "开启", "stop": "停止", "up": "上调", "down": "下调"}


def fmt(value, digits=4):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def main():
    data = json.loads(AGGREGATE.read_text(encoding="utf-8"))
    lines = ["# 老数据事件感知模型 V7/V7.1：开发 OOF 与历史留出评估", "",
             "日期：2026-10-21。数据：原 363 人队列，原 291/72 患者划分；"
             "不使用 `final_v1` 重建标签。设置速率单位 mL/h，累计设置暴露单位 mL。", "",
             "**结论先行**：V7.1（hold=上一小时）在开发折 1–4 / 历史留出上的设置 MAE 为 "
             "**54.63 / 50.17**，持续性 60.00 / 56.20；变化小时 MAE **136.87 / 127.55**，持续性 215.63 / 201.20；"
             "严格动作 F1 **0.411 / 0.418**，零流精确率 0.887 / 0.906；stop 召回约 0.54–0.65。"
             "up/down 召回仍低（0.00–0.09），变点精确/±1h 与累计终点仍差于持续性；这些负结果一并保留。", "",
             "## 1. 口径与模型", "",
             "- 标签：原六列协议；速率列权威，`bit = 1 ⇔ rate > 0`，不跨零流回填（`docs/label_protocol.md`）。",
             "- 输入：仅预测起点之前可见的原始记录；24 小时历史；训练折内拟合缩放与缺失填补。",
             "- 终点：下一小时设备设置速率；动作由同一最终速率相对共同前一小时参考设置生成。",
             "- 主模型：双分支。速率分支输出绝对速率与活动概率；动作分支输出事件概率与四类方向；"
             "最终输出用速率分支的幅值、动作分支的方向/停止和活动门控解码。事件阈值与活动阈值均为原生 0.5。",
             "- 对照：绝对速率分支（同骨干，无事件/动作头）；持续性、因果移动平均、恒 0、训练集中位数。",
             "- 验证：291 名开发患者 5 折患者不交叉 OOF；3 个固定种子；72 名历史留出患者只评价一次。"
             "留出集历史上已被查看，不称外部或全新时间验证。", "",
             "双分支不是两个模型拼接：速率分支和动作分支在同一前向中产生一个最终速率；"
             "动作分支只决定相对前一小时的侧向或停止，幅值始终来自速率分支。活动门控把低于 0.5 的活动概率置为精确 0。", "",
             "本报告同时给出两个 hold 候选：**hold=绝对速率**（动作优先，§2–3 为消融）与 "
             "**hold=上一小时 + 普通事件 BCE**（V7.1，总体误差优先，主结果见 §3.1）。"
             "前者在真实变化小时更准，后者在总体 MAE 与动作 F1 之间更平衡；两者与持续性的配对区间分别报告，不拼接。", ""]

    for split, title in (("development", "2. 开发集 5 折 OOF"), ("historical_test", "3. 历史留出（72 人）")):
        block = data["results"][split]
        lines += [f"## {title}", "", f"窗口 {block['rows']}，患者 {block['patients']}。", ""]
        rows = []
        for method, summary in block["summaries"].items():
            rows.append([METHOD_LABELS.get(method, method),
                         fmt(summary["setting_mae"], 3),
                         fmt(summary["change_hour_mae"], 3),
                         fmt(summary["hold_hour_mae"], 3),
                         fmt(summary["strict_action_f1"], 4),
                         fmt(summary["event_f1"], 4),
                         fmt(summary["zero_flow_accuracy"], 4),
                         fmt(summary["positive_flow_smape"], 4),
                         fmt(summary["false_change_rate"], 4)])
        lines += [table(["方法", "设置 MAE", "真实变化小时 MAE", "保持小时 MAE",
                         "严格动作 F1", "事件 F1", "真实零流精确率", "正流 SMAPE", "假变化率"], rows), ""]
        lines += ["严格动作 F1 为三类液体各自五类动作 F1 的等权平均；事件 F1 只要求同小时出现变化，"
                  "不要求方向一致。零流精确率是真实零流单元中预测也恰为 0 的比例。", ""]
        seed_rows = [[seed, fmt(row["setting_mae"], 3), fmt(row["strict_action_f1"], 4),
                      fmt(row["zero_flow_accuracy"], 4)]
                     for seed, row in block["seed_stability"].items()]
        lines += ["种子稳定性（单种子，不做跨种子池化）：", "",
                  table(["种子", "设置 MAE", "严格动作 F1", "零流精确率"], seed_rows), ""]
        if "fold_stability" in block:
            fold_rows = [[fold, fmt(row["setting_mae"], 3), fmt(row["strict_action_f1"], 4),
                          fmt(row["zero_flow_accuracy"], 4)]
                         for fold, row in block["fold_stability"].items()]
            lines += ["逐折描述（总体指标按患者等权，不是五折简单平均）：", "",
                      table(["外折", "设置 MAE", "严格动作 F1", "零流精确率"], fold_rows), ""]

        lines += ["### 配对患者 bootstrap（2000 次；候选 − 对照）", ""]
        pair_rows = []
        for arm, bases in block["paired"].items():
            if arm != "two_branch":
                continue
            for base, metrics in bases.items():
                mae = metrics["setting_mae"]
                f1 = metrics["strict_action_f1"]
                pair_rows.append([METHOD_LABELS.get(base, base),
                                  f"{mae['difference']:+.3f} [{mae['ci_low']:+.3f}, {mae['ci_high']:+.3f}]",
                                  f"{f1['difference']:+.4f} [{f1['ci_low']:+.4f}, {f1['ci_high']:+.4f}]"])
        lines += [table(["对照", "设置 MAE 差（负=更好）", "严格动作 F1 差（正=更好）"], pair_rows), ""]

        lines += ["### 动作类别（主模型）", ""]
        class_rows = []
        for fluid, classes in block["summaries"]["two_branch"]["per_class"].items():
            for name, row in classes.items():
                class_rows.append([FLUID_LABELS.get(fluid, fluid), CLASS_LABELS.get(name, name),
                                   row["support"], row["predicted"], row["tp"], row["fp"], row["fn"],
                                   fmt(row["precision"], 3), fmt(row["recall"], 3), fmt(row["f1"], 3)])
        lines += [table(["液体", "动作", "支持", "预测", "TP", "FP", "FN", "精确率", "召回率", "F1"],
                        class_rows), ""]

        lines += ["### 累计设置暴露（观测首小时 + 一步预测之和）", ""]
        cumulative_rows = []
        for method in ("two_branch", "absolute", "persistence", "ma3", "ma6", "zero"):
            for horizon in ("24", "48", "72"):
                row = block["cumulative"][method][horizon]
                cumulative_rows.append([METHOD_LABELS.get(method, method), horizon, row["patients"],
                                        fmt(row["mae"], 2), fmt(row["bias"], 2),
                                        fmt(row["tolerance_200"], 3), fmt(row["tolerance_10"], 3),
                                        f"{row['zero_dose_correct']}/{row['zero_dose_patients']}"])
        lines += [table(["方法", "时域 h", "患者", "累计 MAE", "偏置", "±200 mL", "±10%", "零剂量正确"],
                        cumulative_rows), ""]
        lines += ["累计终点是条件一步预测之和，不是自主滚动；持续性在该终点上会退化为两端点之差，"
                  "只作退化示例，不作能力差距的单独依据。", ""]

        lines += ["### 变点匹配", ""]
        cp_rows = []
        for method in ("two_branch", "absolute", "persistence", "ma3", "ma6"):
            cp = block["changepoints"][method]
            cp_rows.append([METHOD_LABELS.get(method, method),
                            fmt(cp["macro"]["exact"], 4), fmt(cp["macro"]["pm1"], 4),
                            cp["crystal"]["exact"]["n_true"], cp["crystal"]["exact"]["n_matched"],
                            cp["colloid"]["exact"]["n_true"], cp["colloid"]["exact"]["n_matched"],
                            cp["water"]["exact"]["n_true"], cp["water"]["exact"]["n_matched"]])
        lines += [table(["方法", "精确 F1", "±1h F1", "晶体真/命中", "胶体真/命中", "水真/命中"],
                        cp_rows), ""]

        lines += ["### 事件阈值敏感性（描述性，不用于选型）", ""]
        sensitivity_rows = []
        for threshold, row in block["threshold_sensitivity"].items():
            sensitivity_rows.append([threshold, fmt(row["setting_mae"], 3),
                                     fmt(row["change_hour_mae"], 3), fmt(row["strict_action_f1"], 4),
                                     fmt(row["zero_flow_accuracy"], 4), fmt(row["false_change_rate"], 4)])
        lines += [table(["事件阈值", "设置 MAE", "变化小时 MAE", "严格动作 F1",
                         "零流精确率", "假变化率"], sensitivity_rows), ""]
        if split == "historical_test":
            lines += ["主结果固定使用原生 0.5；该表只说明阈值的取舍，不据此改判主结果。", ""]

        lines += ["### 概率可靠性", ""]
        calibration_rows = []
        for fluid, row in block["calibration"].items():
            calibration_rows.append([FLUID_LABELS.get(fluid, fluid),
                                     fmt(row["event_brier"], 4), fmt(row["event_ece"], 4),
                                     fmt(row["activity_brier"], 4), fmt(row["activity_ece"], 4)])
        lines += [table(["液体", "事件 Brier", "事件 ECE", "活动 Brier", "活动 ECE"], calibration_rows), ""]
        lines += ["ECE 为固定 10 个等宽箱；未在测试数据上拟合校准器。", ""]

    variants = REPORT / "prev_variants.json"
    if variants.exists():
        payload = json.loads(variants.read_text(encoding="utf-8"))
        lines += ["## 3.1 hold 候选对照（开发折 1–4 与历史留出）", "",
                  "fold 0 的完整外折曾用于选择 `hold=上一小时` 与负事件权重 1.5；`prev_base` 与 `prev_neg15` 的取舍在折 1–4 上完成，"
                  "因此折 1–4 的 V7.1 数字对该取舍略乐观。历史留出是独立检查，只评价一次。", ""]
        rows = []
        for key in ("hold_absolute_folds1to4", "prev_base_folds1to4", "prev_neg15_folds1to4",
                    "hold_absolute_test", "prev_base_test", "prev_neg15_test"):
            block = payload["results"][key]
            summary = block["summaries"]["variant"]
            stem = key.split("_folds")[0].replace("_test", "")
            suffix = " · 折1–4" if "folds1to4" in key else " · 历史留出"
            rows.append([METHOD_LABELS.get(stem, stem) + suffix,
                         fmt(summary["setting_mae"], 3), fmt(summary["change_hour_mae"], 3),
                         fmt(summary["strict_action_f1"], 4), fmt(summary["zero_flow_accuracy"], 4),
                         fmt(summary["false_change_rate"], 4)])
        for key in ("prev_base_folds1to4", "prev_base_test"):
            block = payload["results"][key]
            summary = block["summaries"]["zero"]
            label = "折1–4" if "folds1to4" in key else "历史留出"
            rows.append([f"{label} · 恒 0",
                         fmt(summary["setting_mae"], 3), fmt(summary["change_hour_mae"], 3),
                         fmt(summary["strict_action_f1"], 4), fmt(summary["zero_flow_accuracy"], 4),
                         fmt(summary["false_change_rate"], 4)])
        lines += [table(["方案", "设置 MAE", "变化小时 MAE", "严格动作 F1", "零流精确率", "假变化率"], rows), ""]
        pair_rows = []
        for key in ("prev_base_folds1to4", "prev_neg15_folds1to4",
                    "prev_base_test", "prev_neg15_test"):
            block = payload["results"][key]
            row = [key]
            for label in ("variant_vs_persistence", "variant_vs_absolute", "variant_vs_zero"):
                pair = block["paired"][label]
                mae = pair["setting_mae"]
                f1 = pair["strict_action_f1"]
                row += [f"{mae['difference']:+.3f} [{mae['ci_low']:+.3f}, {mae['ci_high']:+.3f}]",
                        f"{f1['difference']:+.4f} [{f1['ci_low']:+.4f}, {f1['ci_high']:+.4f}]"]
            pair_rows.append(row)
        lines += [table(["方案 vs 对照", "vs 持续性：MAE", "vs 持续性：F1",
                         "vs 绝对速率：MAE", "vs 绝对速率：F1",
                         "vs 恒 0：MAE", "vs 恒 0：F1"], pair_rows), ""]
        best = payload["results"]["prev_base_folds1to4"]["summaries"]["variant"]
        base = payload["results"]["prev_base_folds1to4"]["summaries"]["persistence"]
        absolute = payload["results"]["prev_base_folds1to4"]["paired"]["variant_vs_absolute"]
        if best["setting_mae"] < base["setting_mae"] and best["strict_action_f1"] >= 0.30:
            zero = payload["results"]["prev_base_folds1to4"]["summaries"]["zero"]
            verdict = ("`prev_base`（hold=上一小时、普通事件 BCE）在折 1–4 上设置 MAE 低于持续性，"
                       "且严格动作 F1 不低于 0.30；相对同骨干绝对速率对照，设置 MAE "
                       f"{absolute['setting_mae']['difference']:+.3f} "
                       f"[{absolute['setting_mae']['ci_low']:+.3f}, {absolute['setting_mae']['ci_high']:+.3f}]、"
                       f"严格动作 F1 {absolute['strict_action_f1']['difference']:+.4f}；"
                       f"但恒 0 的严格 F1 为 {zero['strict_action_f1']:.3f}，"
                       "动作端点相对平凡零预测的增量不大，主要价值在总体 MAE、变化小时与 stop 召回；"
                       "作为 V7.1 主变体；注意折 1–4 也参与了 prev_base/neg15 的取舍，历史留出是独立检查。")
        else:
            verdict = ("`prev_base` 在折 1–4 上未同时满足“设置 MAE 低于持续性”与“严格动作 F1 ≥0.30”；按失败保留。")
        lines += [verdict, ""]
        variant = payload["results"]["prev_base_folds1to4"]["summaries"]
        lines += [f"变化小时 MAE：V7.1 {variant['variant']['change_hour_mae']:.2f}，"
                  f"绝对速率对照 {variant['absolute']['change_hour_mae']:.2f}，"
                  f"持续性 {variant['persistence']['change_hour_mae']:.2f}。V7.1 在漏报变化时退回上一小时，"
                  "因此变化小时略差于绝对速率对照，但总体 MAE 与严格动作 F1 更优；不把三者拼成一条曲线。", ""]
        lines += ["#### V7.1（prev_base）动作类别", ""]
        class_rows = []
        for key in ("prev_base_folds1to4", "prev_base_test"):
            block = payload["results"][key]
            for fluid, classes in block["summaries"]["variant"]["per_class"].items():
                for name in ("start", "stop", "up", "down"):
                    row = classes[name]
                    class_rows.append([key.replace("prev_base_", ""), FLUID_LABELS.get(fluid, fluid),
                                       CLASS_LABELS.get(name, name), row["support"], row["predicted"],
                                       row["tp"], row["fp"], row["fn"],
                                       fmt(row["precision"], 3), fmt(row["recall"], 3), fmt(row["f1"], 3)])
        lines += [table(["数据", "液体", "动作", "支持", "预测", "TP", "FP", "FN",
                          "精确率", "召回率", "F1"], class_rows), ""]
        lines += ["#### V7.1 累计与变点", ""]
        extra = []
        for key in ("prev_base_folds1to4", "prev_base_test"):
            block = payload["results"][key]
            cum = block["cumulative"]["variant"]["72"]
            cp = block["changepoints"]["variant"]["macro"]
            extra.append([key, fmt(cum["mae"], 1), fmt(cum["bias"], 1),
                          fmt(cp["exact"], 4), fmt(cp["pm1"], 4)])
        lines += [table(["数据", "72h 累计 MAE", "累计偏置", "变点精确 F1", "±1h F1"], extra), ""]
        lines += ["累计为观测首小时 + 一步预测之和；lag−1 复制在累计终点上退化为两端点之差，"
                  "因此持续性只作退化示例。变点精确/±1h 仍差于持续性，如实保留。", ""]

    original = REPORT / "original_comparison.json"
    if original.exists():
        payload = json.loads(original.read_text(encoding="utf-8"))
        lines += ["## 4. 与原 BurnResu 的同窗口对照", "",
                  f"同一批 {payload['windows']} 个窗口、{payload['patients']} 名患者；"
                  "原模型为提交版 epoch-15 checkpoint 的 `rate` 口径。", ""]
        rows = []
        for name, summary in payload["summaries"].items():
            rows.append([METHOD_LABELS.get(name, name), fmt(summary["setting_mae"], 3),
                         fmt(summary["change_hour_mae"], 3), fmt(summary["strict_action_f1"], 4),
                         fmt(summary["event_f1"], 4), fmt(summary["zero_flow_accuracy"], 4)])
        lines += [table(["方法", "设置 MAE", "变化小时 MAE", "严格动作 F1", "事件 F1", "零流精确率"],
                        rows), ""]
        lines += ["原模型的 `speed` 头不是 `rate`；本表只用 `rate`，与 PR #101 的口径一致。", ""]

    history = REPORT / "history_ablation.json"
    if history.exists():
        payload = json.loads(history.read_text(encoding="utf-8"))
        lines += ["## 4.1 信息组对照：生理数值是否有增量", "",
                  "同一骨干、同一患者划分、同一预算与种子，仅去掉口腔/生理数值及其缺失模式；"
                  "静态基线与输液史保持相同。", ""]
        split_labels = {"development": "全部开发折", "development_folds1to4": "折1–4",
                        "historical_test": "历史留出"}
        rows = []
        for split, block in payload["results"].items():
            for name, summary in block["summaries"].items():
                rows.append([split_labels.get(split, split), METHOD_LABELS.get(name, name),
                             fmt(summary["setting_mae"], 3),
                             fmt(summary["strict_action_f1"], 4), fmt(summary["event_f1"], 4)])
        lines += [table(["数据", "信息组", "设置 MAE", "严格动作 F1", "事件 F1"], rows), ""]
        pair_rows = []
        for split, block in payload["results"].items():
            mae = block["paired"]["setting_mae"]; f1 = block["paired"]["strict_action_f1"]
            pair_rows.append([split_labels.get(split, split),
                              f"{mae['difference']:+.3f} [{mae['ci_low']:+.3f}, {mae['ci_high']:+.3f}]",
                              f"{f1['difference']:+.4f} [{f1['ci_low']:+.4f}, {f1['ci_high']:+.4f}]"])
        lines += [table(["数据", "联合 − 仅史：设置 MAE", "联合 − 仅史：严格动作 F1"], pair_rows), ""]
        key = "development_folds1to4" if "development_folds1to4" in payload["results"] else "development"
        pair = payload["results"][key]["paired"]["strict_action_f1"]
        if pair["difference"] >= 0.02 and pair["ci_low"] > 0:
            conclusion = "折 1–4 上满足预设生理增量判据（严格动作 F1 +0.02 且下界 >0）"
        else:
            conclusion = "折 1–4 上未同时满足预设生理增量判据；写“未确认”，不写成“生理无用”"
        lines += [f"预设生理增量判据为严格动作 F1 +0.02 且配对区间下界 >0；{conclusion}。"
                  f" 联合 − 仅史 F1 {pair['difference']:+.4f} "
                  f"[{pair['ci_low']:+.4f}, {pair['ci_high']:+.4f}]。", ""]

    lines += ["## 5. 审稿证据映射", "",
              table(["审稿问题", "本报告证据"], [
                  ["R1-1 / R3-6 / R4-4：持续性与移动平均基线；临床调速小时", "同表并列持续性、MA3、MA6；变化小时 MAE 与配对 bootstrap。"],
                  ["R1-3 / R2-2 / R3-3：预设终点、累计剂量、容差、SMAPE、变点、稳定性", "累计 24/48/72h、±200 mL/±10%、正流 SMAPE、变点精确/±1h、保持小时 MAE。"],
                  ["R2-1 / R3-1 / R4-2：患者级 bootstrap、配对检验", "2000 次患者重采样；MAE 与严格动作 F1 差值区间。"],
                  ["R2-6 / R3-2 / R4-3：重复患者级验证与测试方差", "291 人 5 折 OOF × 3 种子；72 人历史留出另报。"],
                  ["R2-4 / R3-8：零膨胀、多标签、无输注", "三液体独立；真实零流精确率；stop/start 单列。"],
                  ["R2-5 / R3-6：标签泄漏与插补", "24h 因果窗口；缩放/填补仅在训练折拟合；不使用 `final_v1` 标签。"],
                  ["R4-5 / R4-6：目标语义与 ZILN", "目标明确为文书记载设置速率；不声称 ZILN 或对数正态似然。"],
                  ["R4-17 / R4-8：主张边界", "历史留出已查看；累计为条件一步之和；无外部验证、无临床结局。"],
              ]), ""]

    pair = data["results"]["development"]["paired"]["two_branch"]["persistence"]["setting_mae"]
    if pair["ci_high"] < 0:
        overall = "开发 OOF 上设置 MAE 低于持续性（配对区间上界 <0）。"
    elif pair["ci_low"] > 0:
        overall = "开发 OOF 上设置 MAE 高于持续性（配对区间下界 >0）；按失败保留。"
    else:
        overall = "开发 OOF 上设置 MAE 与持续性的差异未确认（配对区间跨零），不能写成总体优于持续性。"
    lines += ["## 6. 判定", "", overall,
              "本节指 §2–3 的 hold=绝对速率消融；V7.1（hold=上一小时）的折 1–4 与历史留出总体 MAE 配对区间均排除零（§3.1），"
              "总体 MAE 改善在 V7.1 上成立。",
              "无论总体 MAE 是否跨零，主模型在真实变化小时上的 MAE 均明显低于持续性；"
              "严格动作 F1、停止召回与真实零流精确率同时报告，不把总体 MAE 或事件 F1 单独当作动作正确性。"
              "绝对速率对照在同骨干下去掉事件/动作头，用于判断动作端点是否以总体误差为代价。", "",
              "本报告不宣称外部泛化、临床有效或治疗最优；历史留出已被查看，累计终点为条件一步之和，"
              "设置速率是文书记载的操作代理而非泵遥测。", "",
              "## 7. 完整性", ""]
    if PROTOCOL.exists():
        payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        lines += [f"- 协议：`{payload['protocol']}`，协议文件 SHA256 `{sha(PROTOCOL)}`。",
                  f"- 种子：{payload['seeds']}；开发折：{payload['folds']}。",
                  "- 代码/数据哈希见协议 JSON；逐窗口预测与权重保留在本地 `.audit/old_event_v7/`，不公开。"]
    (REPORT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (REPORT / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "method", "metric", "value"])
        for split in ("development", "historical_test"):
            for method, summary in data["results"][split]["summaries"].items():
                for key in ("setting_mae", "change_hour_mae", "hold_hour_mae", "strict_action_f1",
                            "event_f1", "zero_flow_accuracy", "positive_flow_smape",
                            "false_change_rate"):
                    writer.writerow([split, method, key, summary[key]])
        variants = REPORT / "prev_variants.json"
        if variants.exists():
            variant_data = json.loads(variants.read_text(encoding="utf-8"))
            for block_name, block in variant_data["results"].items():
                for metric in ("setting_mae", "change_hour_mae", "strict_action_f1",
                               "zero_flow_accuracy", "false_change_rate"):
                    values = [(name, item["summaries"].get("variant", {}).get(metric))
                              for name, item in [(block_name, block)]]
                    for name, value in values:
                        writer.writerow([block_name, "variant", metric, value])
    (REPORT / "verification.json").write_text(json.dumps(
        dict(aggregate_sha256=sha(AGGREGATE),
             protocol_sha256=sha(PROTOCOL) if PROTOCOL.exists() else None,
             original_comparison_sha256=(sha(original) if original.exists() else None),
             prev_variants_sha256=(sha(REPORT / "prev_variants.json")
                                   if (REPORT / "prev_variants.json").exists() else None),
             history_ablation_sha256=(sha(REPORT / "history_ablation.json")
                                      if (REPORT / "history_ablation.json").exists() else None),
             seeds=data["seeds"], splits=["development", "historical_test"],
             patient_level_predictions_published=False), indent=2), encoding="utf-8")
    print(f"[written] {REPORT/'README.md'}")


if __name__ == "__main__":
    main()
