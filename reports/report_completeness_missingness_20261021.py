"""Render the completeness, interval and missingness tables as Markdown."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/experiments/old-event-v7-20261021"
SPLITS = ("development_folds1to4", "historical_test")
INTERVALS = ("0-24", "0-48", "0-72", "24-48", "48-72", "24-72")
METHODS = ("v71", "v71_mean", "persistence", "zero", "history", "missing", "values")
LABEL = {"v71": "V7.1", "v71_mean": "V7.1-M", "persistence": "持续性",
         "zero": "恒0", "history": "仅史", "missing": "仅缺失指示", "values": "仅生理值"}


def f(value, nd=1):
    if value is None:
        return "—"
    try:
        if value != value:
            return "—"
        return f"{value:,.{nd}f}"
    except (TypeError, ValueError):
        return str(value)


def pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def method(block, name, key, nd=1):
    if not block or not block.get(name):
        return "—"
    return f(block[name].get(key), nd)


def interval_metric(block, name, key, nd=1):
    if not block or not block.get(name):
        return "—"
    return f(block[name].get("interval", {}).get(key), nd)


def main():
    audit = json.loads((REPORT / "data_completeness_audit.json").read_text(encoding="utf-8"))
    result = json.loads((REPORT / "interval_missingness.json").read_text(encoding="utf-8"))
    lines = ["# 原始时序完整性、区间资格与缺失分层", "",
             "数据：老 363 人；只提交脱敏聚合。`时间` 是唯一小时锚；空时间行是未记录，不是零流。",
             "累计口径是文书记载设置速率 × 1 h 的区间暴露，不是实际输注剂量；区间总量不是从 0 点起的累计。", ""]
    lines += ["## 1. 原始时间戳与标签来源", "",
              f"- {audit['cohort']['patients']} 人、{audit['cohort']['rows']} 行；有效时间 {audit['cohort']['valid_rows']} 行、空时间 {audit['cohort']['empty_rows']} 行。",
              f"- 全部从 `时间=1` 开始；重复 {audit['time']['duplicates']}、非整数 {audit['time']['non_integer']}、断档 {audit['time']['gaps']}、内部空行患者 {audit['time']['interior_empty_patients']}。",
              f"- 尾部空时间患者 {audit['time']['trailing_empty_patients']} 人；有效小时 9–72。",
              f"- 标签速率与原始速率列最大差 {audit['labels']['max_abs_diff']:.3g}；bit/rate 不一致 {audit['labels']['bit_rate_mismatch']}；长度不匹配 {audit['labels']['length_mismatch']}。",
              f"- 状态位/速率冲突：status=1 且 rate=0 共 {sum(v['status1_rate0'] for v in audit['labels']['status_conflicts'].values())} 格；status=0 且 rate>0 共 {sum(v['status0_rate_pos'] for v in audit['labels']['status_conflicts'].values())} 格。状态位不用于观测/零流判定。", ""]
    lines += ["## 2. 区间资格", "",
              "每个区间要求区间内每小时恰好出现一次，且三类设置速率可验证；缺 48–72 h 者仍可进入 0–24/0–48，不完整区间不进入对应累计终点。", "",
              "| 数据 | 区间 | 总患者 | 合格 | 排除 | 区间小时 | 含首小时 |", "|---|---|---:|---:|---:|---:|---|"]
    for split in SPLITS:
        for interval in INTERVALS:
            row = result["eligibility"][split][interval]
            lines.append(f"| {split} | {interval} | {row['n_total']} | {row['n_eligible']} | {row['n_excluded']} | {row['interval_hours']} | {row['includes_initial_hour']} |")
    lines += ["", "被排除的 27 名 <72 h 患者与 72 h 合格者在年龄/总面积上相近，但全层烧伤中位更高（35 vs 16，p=0.0009）；72 h 终点只对完整随访者成立，不能无条件外推到全队列。", ""]
    lines += ["## 3. 区间总量与逐小时指标（同一合格集）", "",
              "| 数据 | 区间 | 方法 | n | 区间总量 MAE | 偏置 | 相对 MAE | ±200 mL | ±10% | 设置 MAE | 变化小时 MAE | 严格 F1 | 活动 macro-F1 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        for interval in INTERVALS:
            block = result["methods"][split][interval]
            for name in ("v71", "v71_mean", "persistence"):
                if not block.get(name):
                    continue
                item = block[name]
                lines.append(f"| {split} | {interval} | {LABEL[name]} | {item['interval']['patients']} | "
                             f"{interval_metric(block, name, 'total_mae')} | {interval_metric(block, name, 'total_bias')} | "
                             f"{pct(item['interval'].get('total_relative_mae'))} | {pct(item['interval'].get('total_tolerance_200'))} | "
                             f"{pct(item['interval'].get('total_tolerance_10'))} | {f(item['setting_mae'], 2)} | "
                             f"{f(item['change_hour_mae'], 2)} | {f(item['strict_action_f1'], 3)} | {f(item['activity_macro_f1'], 3)} |")
    lines += ["", "持续性的区间总量误差退化为区间两端点之差，只作退化参照；它的逐小时严格动作 F1 为 0（永不调速）。", ""]
    lines += ["### 3.1 0–72 h 的 P1/P2/P3", "",
              "| 数据 | 方法 | P1 总量 MAE | P2 晶体/胶体/水 MAE | P3 活动 macro-F1 | P3 micro-F1 |", "|---|---|---:|---|---:|---:|"]
    for split in SPLITS:
        block = result["methods"][split]["0-72"]
        for name in ("v71", "v71_mean", "persistence"):
            item = block.get(name)
            if not item:
                continue
            p2 = item["interval"].get("fluid_mae") or [None, None, None]
            lines.append(f"| {split} | {LABEL[name]} | {f(item['interval'].get('total_mae'), 1)} | "
                         f"{f(p2[0], 1)} / {f(p2[1], 1)} / {f(p2[2], 1)} | "
                         f"{f(item.get('activity_macro_f1'), 3)} | {f(item.get('activity_micro_f1'), 3)} |")
    lines += ["", "P1 是患者级三液体总量绝对误差；P2 是逐液体 MAE，二者不是同一估计量，不能互相替代。", ""]
    lines += ["### 3.2 0–72 h 分液体累计偏置", "",
              "| 数据 | 方法 | 晶体偏置 | 胶体偏置 | 水偏置 |", "|---|---|---:|---:|---:|"]
    for split in SPLITS:
        block = result["methods"][split]["0-72"]
        for name in ("v71", "v71_mean"):
            item = block.get(name)
            if not item:
                continue
            bias = item["interval"].get("fluid_bias") or [None, None, None]
            lines.append(f"| {split} | {LABEL[name]} | {f(bias[0], 1)} | {f(bias[1], 1)} | {f(bias[2], 1)} |")
    lines += ["", "V7.1 的负偏置主要来自晶体与水，胶体较小；mean-matching 主要减小晶体/水的负偏置，胶体改善有限。", ""]
    lines += ["## 4. 缺失分层：0–72 h（72 h 合格者；经验四分位由开发折确定）", "",
              "预设的 ≤10% 层在 0–72 h 为 0 人；10–30% 与 >30% 并列，四分位作为可分层的敏感性分析。分层是描述性的，不作因果解释。", "",
              "| 数据 | 层 | n | 缺失中位 | 真实 72 h 总量均值 | P1 V7.1 | P1 V7.1-M | P1 持续 | 设置 MAE V7.1 | 设置 MAE V7.1-M | 严格 F1 V7.1 | stop 召回 V7.1 | 相对 MAE V7.1 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        block = result["strata"][split]["0-72"]
        for label in ("<=0.10", "0.10-0.30", ">0.30", "all-missing", "Q1", "Q2", "Q3", "Q4"):
            group = block["groups"].get(label)
            if not group:
                continue
            desc = group.get("description") or {}
            if not group.get("evaluated"):
                lines.append(f"| {split} | {label} | {group['patients']} | {f(desc.get('missing_median'), 3)} | "
                             f"{f(desc.get('true_interval_total_mean'), 0)} | — | — | — | — | — | — | — | — |")
                continue
            ms = group["methods"]
            stop = (ms["v71"]["stop_recall"].get("crystal") if ms.get("v71") else None)
            lines.append(f"| {split} | {label} | {group['patients']} | {f(desc.get('missing_median'), 3)} | "
                         f"{f(desc.get('true_interval_total_mean'), 0)} | {interval_metric(ms, 'v71', 'total_mae')} | "
                         f"{interval_metric(ms, 'v71_mean', 'total_mae')} | {interval_metric(ms, 'persistence', 'total_mae')} | "
                         f"{method(ms, 'v71', 'setting_mae', 2)} | {method(ms, 'v71_mean', 'setting_mae', 2)} | "
                         f"{method(ms, 'v71', 'strict_action_f1', 3)} | {f(stop, 3)} | "
                         f"{pct(ms['v71']['interval'].get('total_relative_mae'))} |")
    lines += ["", "0–24 h 的 all-missing 核心体征组在开发折有 21 人，真实 0–24 h 三液体总量均为 0（正流率 0），是“无监测且无输液”的独立组，不是可比的生理缺失组；测试折该组 4 人，仅作描述。", ""]
    lines += ["## 5. 各区间缺失分层（四分位；开发折边界）", "",
              "| 数据 | 区间 | 层 | n | 缺失中位 | 真实区间总量均值 | P1 V7.1 | P1 V7.1-M | P1 持续 | 设置 MAE V7.1 | 严格 F1 V7.1 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        for interval in INTERVALS:
            block = result["strata"][split][interval]
            for label in ("Q1", "Q2", "Q3", "Q4"):
                group = block["groups"].get(label)
                if not group or not group.get("evaluated"):
                    continue
                desc = group.get("description") or {}
                ms = group["methods"]
                lines.append(f"| {split} | {interval} | {label} | {group['patients']} | {f(desc.get('missing_median'), 3)} | "
                             f"{f(desc.get('true_interval_total_mean'), 0)} | {interval_metric(ms, 'v71', 'total_mae')} | "
                             f"{interval_metric(ms, 'v71_mean', 'total_mae')} | {interval_metric(ms, 'persistence', 'total_mae')} | "
                             f"{method(ms, 'v71', 'setting_mae', 2)} | {method(ms, 'v71', 'strict_action_f1', 3)} |")
    lines += ["", "低缺失层的真实输液量明显更高（0–24 h 的 10–30% 层真实总量约 12.1 L vs >30% 层约 5.8 L），因此低缺失层的 MAE/P1 绝对值更高是病例构成与输注幅度差异，不能读成“模型在低缺失时更差”。相对 MAE 一并列出供比较。", ""]
    lines += ["## 6. 生理信息组增量（同一患者、折、种子；配对 bootstrap）", "",
              "Δ = 联合 − 对照；设置 MAE 负值表示联合更低，严格 F1 正值表示联合更高。仅作描述性增量，不作因果解释。", ""]

    def physiology_rows(intervals, labels):
        rows = []
        any = False
        for split in SPLITS:
            for interval in intervals:
                block = result.get("physiology", {}).get(split, {}).get(interval, {})
                for label in labels:
                    row = block.get(label)
                    if not row:
                        continue
                    for name in ("history", "missing", "values"):
                        item = row.get(name, {})
                        if not item.get("available"):
                            continue
                        any = True
                        shown = "全部合格" if label == "all" else label
                        rows.append(f"| {split} | {interval} | {shown} | {LABEL[name]} | "
                                    f"{f(item.get('delta_setting_mae'), 2)} [{f(item.get('delta_setting_mae_ci', [None, None])[0], 2)}, {f(item.get('delta_setting_mae_ci', [None, None])[1], 2)}] | "
                                    f"{f(item.get('delta_strict_f1'), 4)} [{f(item.get('delta_strict_f1_ci', [None, None])[0], 4)}, {f(item.get('delta_strict_f1_ci', [None, None])[1], 4)}] |")
        return rows, any

    for title, intervals, labels in (
        ("6.1 整体", ("0-24", "0-48", "24-48", "48-72", "0-72", "24-72"), ("all",)),
        ("6.2 0–72 h 缺失四分位", ("0-72",), ("Q1", "Q2", "Q3", "Q4")),
        ("6.3 0–24 h 缺失四分位", ("0-24",), ("Q1", "Q2", "Q3", "Q4")),
    ):
        rows, any_control = physiology_rows(intervals, labels)
        lines += [f"### {title}", "",
                  "| 数据 | 区间 | 层 | 对照 | Δ设置 MAE | Δ严格 F1 |", "|---|---|---|---|---:|---:|"]
        if any_control:
            lines += rows
        else:
            lines += ["| — | — | — | — | 对照训练未完成或未注册 | — |"]
        lines += [""]
    lines += ["预设的生理增量判据仍是严格动作 F1 +0.02 且配对区间下界 >0；若未满足，写“未确认”，不写“生理无用”。", ""]
    lines += ["## 7. 判定", "",
              "- **P1 不是只在 72 h 才失败**：V7.1 的 0–24 h 总量 MAE 为 2,412.6/2,298.8 mL（相对 41.6%/46.8%），已明显差于持续性的 348.2/253.9 mL；0–48 h 为 4,485.5/3,824.4 mL，0–72 h 为 5,630.9/5,070.3 mL。误差从第一个 24 h 就存在，随后累积。",
              "- **逐小时更好不等于累计更好**：V7.1 的 0–72 h 设置 MAE 为 54.63/50.17，低于持续性的 60.00/56.20；但持续性累计误差只剩区间两端点之差，V7.1 的系统性负偏置逐小时累积。两者不能互相替代。",
              "- **mean-matching 的改善有限且集中在低缺失层**：V7.1-M 把 0–24 h 总量 MAE 降到 1,736.9/1,729.8 mL，0–72 h 降到 4,130.5/3,930.5 mL；0–72 h 的 10–30% 缺失层相对 MAE 从 25.5%/22.9% 降到 17.2%/14.1%，>30% 层仅从 40.5%/46.1% 降到 33.3%/39.9%。相对 MAE 仍远高于持续性的 0.9%/1.0%。",
              "- **0–72 h 的 ≤10% 缺失层为 0 人**；10–30% 与 >30% 的相对 MAE 分别为 25.5%/22.9% 与 40.5%/46.1%。四分位从低缺失到高缺失的相对 MAE 为 24.2%→45.8%（开发）和 22.6%→48.9%（留出），提示缺失越多相对误差越大；但低缺失层真实输注量高约 2–3 倍，绝对 MAE/P1 更高是构成差异，不能读成“低缺失时模型更差”。",
              "- **0–24 h all-missing 不是生理缺失组**：开发折 21 人该段三液体真实总量均为 0，模型也输出 0；它是“无监测且无输液”的独立组，不能进入“少体征时模型更好/更差”的比较。",
              "- **生理增量**：仅史对照在 0–24 h 的严格动作 F1 差为 +0.0186 [+0.0083, +0.0292]（开发），但 0–72 h 为 +0.0059 [−0.0009, +0.0125]，历史留出 0–72 h 为 +0.0002 [−0.0120, +0.0136]；未达到预设 +0.02 且下界 >0 的判据。改用仅缺失指示对照后，0–24 h 的 F1 差降为 +0.0051 [−0.0051, +0.0159]，0–72 h 设置 MAE 反而比仅缺失指示高 +0.75 [+0.08, +1.42]（开发）；仅生理值对照的 0–72 h ΔF1 为 +0.0029 [−0.0033, +0.0093]（开发）、+0.0001 [−0.0111, +0.0113]（留出），0–24 h 为 +0.0104 [−0.0004, +0.0214]。三种信息组在 0–24 h 的点估计为仅史 +0.0186、仅生理值 +0.0104、仅缺失指示 +0.0051，早期弱信号被“是否记录”和“数值”共同分摊，且都不持续到 48/72 h。区间内还有方向不一致的孤立信号：仅生理值 48–72 h 开发 ΔF1 +0.0156 [+0.0031, +0.0281]，但 24–48 h 为 −0.0105 [−0.0192, −0.0012]，留出 48–72 h 为 +0.0160 [−0.0082, +0.0396]。不能把联合模型的早期 F1 增量归因于生理数值。",
              "- **missing-only / values-only** 对照用于区分“缺失模式”与“生理数值”；若缺失模式已覆盖早期信号，则不能把增量归因于生理数值。结果见上表。", ""]
    lines += ["## 8. 边界", "",
              "- 老数据单中心回顾性；`时间` 是业务小时锚，不是伤后/入院时间戳；无法从本表恢复真实的时间零点偏移。",
              "- 空时间行是未记录，不是零流；标签速率与原始速率列逐值一致，状态位与速率冲突，状态位不用于观测判定。",
              "- 72 h 合格者只占 336/363；被排除者的全层烧伤更多，72 h 结论不能无条件外推到全队列。",
              "- 缺失分层是描述性的；低缺失与高输液量、高正流率同时出现，不能把分层差异解释为缺失的因果效应。",
              "- 累计为条件一步预测之和，不是自主滚动；区间总量是设置暴露，不是实际输注剂量。", ""]
    (REPORT / "interval_missingness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT / 'interval_missingness.md'}")


if __name__ == "__main__":
    main()
