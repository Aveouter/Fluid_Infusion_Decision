"""评估协议 v2 的**冻结常量与词表**（issue #81，Gate A）。

为什么需要这个模块
------------------
审计发现同一个指标名在本仓库里有多套算法：F1 至少有 7 处定义；正流 MAE
在同一份权重上能报出 29.0（``speed`` 口径）与 87.2（``rate`` 口径）两个数；
容差有三组参数（本模块的 200 mL/10%、``utils/util.py`` 的 250 mL、
``main.py`` 里从未被读取的 50 mL）。根本原因是**阈值与词表散落在各文件里**，
没有任何一处能回答"这个数字用的是哪个定义"。

本模块把全部会改变指标含义的常量收敛到一处，并提供一个可复算的
:data:`PROTOCOL_HASH`。任何定义变更都会改变 hash，从而让"用了哪版协议"
变成可核对的事实，而不是靠记忆。

单一来源原则
------------
``utils/endpoints.py`` 与 ``utils/changepoints.py`` 的容差/阈值常量改为
**从这里 import**（re-export），不再各自定义。本模块**不得**反向 import
那两个模块，否则形成循环。

与本模块无关的事
----------------
模型结构、ZILN 参数化、训练损失、解码器实现（见 ``utils/decoding.py``）、
具体指标算术（见 ``utils/endpoints.py``）都不在这里。本模块只回答
"口径叫什么、阈值是多少、哪些组合合法"。
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Tuple

#: 协议版本。不兼容变更必须递增，并同步 docs/eval_protocol.md。
PROTOCOL_VERSION = "2.0"

# --------------------------------------------------------------------------
# 阈值与容差（唯一来源）
# --------------------------------------------------------------------------

#: 容差终点的默认参数。稿件方法 §3.6 原文写的是 ±200 mL 与 ±10%。
#: 注意：代码里 ``main.py --tol_abs_ml`` 的默认值 50.0 从未被读取（见 #6）。
TOL_ABS_ML: float = 200.0
TOL_PCT: float = 0.10

#: 变点阈值：绝对变化 >= 50 mL/h，或（基数 >= 50 mL/h 时）相对变化 >= 25%。
CHANGE_ABS_ML: float = 50.0
CHANGE_REL: float = 0.25
REL_BASE_ML: float = 50.0

#: 正流掩码的速率下限 ε（mL/h）。**在查看测试结果之前冻结**。
#: 0.0 表示"严格大于零即为正流"；标签速率为 mL/h、分辨率为 1 mL/h 量级，
#: 故此处取 0.0 不会把真实的小流量误判为零流。
EPS_RATE_ML_PER_H: float = 0.0

#: 事件匹配容差（小时）：主报告用 ±1h，±2h 仅作敏感性。
MATCH_TOLERANCE_PRIMARY: int = 1
MATCH_TOLERANCE_SENSITIVITY: int = 2

#: 累计剂量的评估时域（**记录**小时数）。
#: 注意 72 个记录小时只对应 71 个**模型预测**小时（首小时是观测锚点，见 ESTIMANDS）。
HORIZONS: Tuple[int, ...] = (24, 48, 72)

#: 预测小时数 = 记录小时数 − 1（首小时在决策时刻已观测）。
PREDICTED_HOURS_BY_HORIZON: Dict[int, int] = {h: h - 1 for h in HORIZONS}

# --------------------------------------------------------------------------
# 统计契约（冻结；执行在 Gate B）
# --------------------------------------------------------------------------

#: 患者级外层验证折数。
N_OUTER_FOLDS: int = 5

#: 关键深度模型的最少固定训练种子数。
N_SEEDS_MIN: int = 3

#: bootstrap 抽样次数与置信水平。
N_BOOTSTRAP: int = 2000
CI_LEVEL: float = 0.95

#: 独立性单位。**不是**小时、不是窗口。
INDEPENDENCE_UNIT: str = "patient"

# --------------------------------------------------------------------------
# 词表
# --------------------------------------------------------------------------

#: 评估模式。两种**绝不混排为同一个 estimand**。
EVAL_MODES: Tuple[str, ...] = (
    "one_step_observed_history",          # 教师强制：以真实历史为条件的一步预测
    "action_autoregressive_rollout",      # 动作自回归：回填模型自身输出
    "na",
)

#: 速率口径。语义按源码写死，见 docs/eval_protocol.md §2。
#:
#: - ``pos_mean``           ``exp(mu + sigma^2/2) - 1``，**条件正流均值**
#: - ``flow_pred``          ``(1-p0) * pos_mean``，ZILN **无条件**均值 E[Y]
#: - ``speed``              ``softplus(flow_pred)``，**无条件**均值经 softplus
#: - ``rate``               ``prob_cls * speed``，分类概率**再次**门控
#: - ``action_rate_uncond`` ``1[prob_cls >= tau] * speed``，阈值门控的动作解码
#: - ``action_rate_cond``   ``1[prob_cls >= tau] * pos_mean``，条件强度版动作解码
#: - ``deterministic``      无概率头的基线：``bit = 1 <=> rate > 0``
RATE_DEFINITIONS: Tuple[str, ...] = (
    "pos_mean",
    "flow_pred",
    "speed",
    "rate",
    "action_rate_uncond",
    "action_rate_cond",
    "deterministic",
    "na",
)

#: 需要概率头参与的口径（供 ``predict_test_set``/适配器校验）。
PROBABILISTIC_RATE_DEFINITIONS: Tuple[str, ...] = (
    "rate", "action_rate_uncond", "action_rate_cond",
)

#: 被解码器产出的口径（而不是由模型直接输出）。
DECODED_RATE_DEFINITIONS: Tuple[str, ...] = (
    "action_rate_uncond", "action_rate_cond",
)

#: 确定性解码规则。解码**不是**指标定义的一部分，但属于冻结协议。
DECODERS: Tuple[str, ...] = (
    "threshold_gate_uncond",   # 1[prob >= tau_c] * speed
    "threshold_gate_cond",     # 1[prob >= tau_c] * pos_mean
)

#: 估计对象。每个累计量行必须带其一，且**禁止跨 estimand 汇总或配对**。
#:
#: 数学事实：同窗同锚点下
#:     |(V0 + Vhat_S) - (V0 + V_S)| == |Vhat_S - V_S|
#: 因此 E1 与 E3 的终末累计**绝对误差必然相同**；E3 只在相对容差分母、
#: 累计量绝对水平、时域解释三处提供额外信息，**不能**作为 E1 误差的独立证据。
ESTIMANDS: Tuple[str, ...] = (
    "conditional_cumulative_volume",                        # E1
    "action_autoregressive_rollout_under_true_initial_state",  # E2
    "observed_physiology_conditioned_rollout",              # E2b
    "pure_prediction_window_volume",                        # E3
    "na",
)

#: 时域锚点。``hour_0_anchored`` 用决策时刻已观测的首小时补齐 72 个记录小时；
#: ``first_predicted_hour`` 不补齐，故只有 71 个小时。
HORIZON_ORIGINS: Tuple[str, ...] = (
    "hour_0_anchored",
    "first_predicted_hour",
    "na",
)

#: 掩码。**真实零值、缺失、结构不完整必须分开**，不得合并。
MASKS: Tuple[str, ...] = (
    "all",
    "positive_flow",        # true_rate > EPS_RATE_ML_PER_H
    "zero_flow",            # 真值恰为 0（真实零输注）
    "missing",              # 哨兵/未记录 —— 与真实零值不同
    "incomplete",           # 时域未满，结构不完整
    "true_change_hour",
    "no_change_hour",
    "zero_dose_patient",
    "eligible_patient",
    "matched_events",
    "na",
)

#: 液体类别。``total`` 为三类之和。
FLUID_CLASSES: Tuple[str, ...] = ("total", "crystal", "colloid", "water", "na")

#: 质量轨。主表与敏感性表必须可区分，否则两表撞键。
QUALITY_TRACKS: Tuple[str, ...] = ("main", "adjudicated_mask", "na")

#: 证据分层（§2A）。每条临床结果都必须携带。
EVIDENCE_TIERS: Tuple[str, ...] = (
    "measured",               # 实测值
    "derived",                # 由实测值推导（如 mL/kg/%TBSA）
    "proposed_exploratory",   # 本研究拟议、未经临床验证
    "guidance_only",          # 指南参考值，**不是**金标准目标
    "not_assessable",         # 数据不可用
    "na",
)

#: 结果的观测来源：实测 vs 插补。混用会让"不确定性"失去信息量。
OBSERVATION_SOURCES: Tuple[str, ...] = ("observed", "imputed", "unavailable", "na")

#: 标量单位。v1 把掩码编进单位名（``positive_flow_ml_per_h``），
#: 这正是 v2 拆开 ``unit`` 与 ``mask`` 的原因。
UNITS: Tuple[str, ...] = (
    "probability",
    "count",
    "ratio",
    "hours",
    "mL/h",
    "mL",
    "mL/kg",
    "mL/kg/%TBSA",
)

#: 聚合方式。``patient_*`` 表示**先按患者聚合再对患者平均**，这是本协议的
#: 默认口径；``target_pooled`` 把每个 (小时, 类别) 当独立单元，**仅作辅助**。
AGGREGATIONS: Tuple[str, ...] = (
    "per_class",
    "macro",
    "micro",
    "weighted",
    "samples",
    "patient_mean",
    "patient_median",
    "patient_share",
    "target_mean",
    "target_pooled",
    "event_macro",
    "event_micro",
)

#: 聚合层级。与 v1 保持一致（有测试钉住），额外结构由专用字段承载。
AGGREGATION_LEVELS: Tuple[str, ...] = ("window", "patient")

#: 量化行的单位（相对 ``probability``/``count`` 而言）。这些单位下的行
#: 必须声明一个真正的聚合方式，不得写 ``"none"``。
QUANTITATIVE_UNITS: Tuple[str, ...] = ("mL/h", "mL", "mL/kg", "mL/kg/%TBSA")


def freeze_payload() -> Dict[str, object]:
    """返回参与 hash 的全部冻结项。

    只包含**会改变指标含义**的东西，不含代码本身。键名稳定，
    因此 hash 可复算、可写进论文与回复信。
    """
    return {
        "protocol_version": PROTOCOL_VERSION,
        "tolerances": {"tol_abs_ml": TOL_ABS_ML, "tol_pct": TOL_PCT},
        "change_point": {
            "change_abs_ml": CHANGE_ABS_ML,
            "change_rel": CHANGE_REL,
            "rel_base_ml": REL_BASE_ML,
        },
        "positive_flow_epsilon_ml_per_h": EPS_RATE_ML_PER_H,
        "event_matching": {
            "primary_tolerance_hours": MATCH_TOLERANCE_PRIMARY,
            "sensitivity_tolerance_hours": MATCH_TOLERANCE_SENSITIVITY,
            "rule": "maximum_cardinality_then_min_total_offset",
        },
        "horizons": list(HORIZONS),
        "predicted_hours_by_horizon": {str(k): v for k, v in PREDICTED_HOURS_BY_HORIZON.items()},
        "statistics": {
            "n_outer_folds": N_OUTER_FOLDS,
            "n_seeds_min": N_SEEDS_MIN,
            "n_bootstrap": N_BOOTSTRAP,
            "ci_level": CI_LEVEL,
            "independence_unit": INDEPENDENCE_UNIT,
        },
        "vocabularies": {
            "eval_modes": list(EVAL_MODES),
            "rate_definitions": list(RATE_DEFINITIONS),
            "estimands": list(ESTIMANDS),
            "horizon_origins": list(HORIZON_ORIGINS),
            "masks": list(MASKS),
            "fluid_classes": list(FLUID_CLASSES),
            "quality_tracks": list(QUALITY_TRACKS),
            "evidence_tiers": list(EVIDENCE_TIERS),
            "observation_sources": list(OBSERVATION_SOURCES),
            "units": list(UNITS),
            "aggregations": list(AGGREGATIONS),
            "aggregation_levels": list(AGGREGATION_LEVELS),
        },
    }


def protocol_hash(payload: Dict[str, object] = None) -> str:
    """冻结项的 sha256 前 16 位。

    用 ``sort_keys=True`` 与 ``separators`` 固定序列化形式，保证同一份
    冻结项在任何机器上得到同一个 hash。
    """
    data = freeze_payload() if payload is None else payload
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


#: 当前冻结项的 hash。写进每一行结果与协议文档。
PROTOCOL_HASH: str = protocol_hash()


def assert_frozen(expected_hash: str) -> None:
    """校验运行环境的冻结项与 ``expected_hash`` 一致，不一致即拒绝运行。

    这是"先冻结、后评估"的执行点：阈值/词表一旦改动，历史结论就不能
    与新结果混在一张表里。
    """
    actual = protocol_hash()
    if actual != expected_hash:
        raise AssertionError(
            f"协议 hash 不匹配：期望 {expected_hash}，实际 {actual}。"
            "冻结后修改定义必须升级 PROTOCOL_VERSION 并保留变更记录。"
        )


#: 词表容器，便于按名字取用与做成员校验。
VOCABULARIES: Dict[str, Tuple[str, ...]] = {
    "eval_mode": EVAL_MODES,
    "rate_definition": RATE_DEFINITIONS,
    "estimand": ESTIMANDS,
    "horizon_origin": HORIZON_ORIGINS,
    "mask": MASKS,
    "fluid_class": FLUID_CLASSES,
    "quality_track": QUALITY_TRACKS,
    "evidence_tier": EVIDENCE_TIERS,
    "observation_source": OBSERVATION_SOURCES,
    "unit": UNITS,
    "aggregation": AGGREGATIONS,
    "level": AGGREGATION_LEVELS,
}

#: 生成区块在 docs/eval_protocol.md 中的定界标记。区块内容**全部由本模块生成**，
#: 手写叙事只能放在标记之外，因此文档与代码不会漂移。
FROZEN_BLOCK_BEGIN = "<!-- BEGIN GENERATED:PROTOCOL-CONSTANTS -->"
FROZEN_BLOCK_END = "<!-- END GENERATED:PROTOCOL-CONSTANTS -->"


def render_frozen_block() -> str:
    """把全部冻结项渲染成 markdown 表格，供文档生成器拼接。

    输出只依赖常量，因此同一份代码在任何机器上生成同一段文字。
    """
    def _tbl(title: str, rows) -> str:
        out = [f"**{title}**", "", "| 项 | 值 |", "|---|---|"]
        out += [f"| `{k}` | {v} |" for k, v in rows]
        return "\n".join(out) + "\n"

    def _vocab(name: str, values: Tuple[str, ...]) -> str:
        return f"- `{name}`： " + "、".join(f"`{v}`" for v in values)

    parts = [
        f"**协议版本**：`{PROTOCOL_VERSION}`　　**冻结哈希**：`{PROTOCOL_HASH}`",
        "",
        "> 哈希由本区块的全部冻结项经 sha256 计算（前 16 位）。任何阈值或词表"
        "变更都会改变哈希；冻结后修改定义必须递增版本并保留变更记录。",
        "",
        _tbl("容差与阈值", [
            ("TOL_ABS_ML", TOL_ABS_ML),
            ("TOL_PCT", TOL_PCT),
            ("CHANGE_ABS_ML", CHANGE_ABS_ML),
            ("CHANGE_REL", CHANGE_REL),
            ("REL_BASE_ML", REL_BASE_ML),
            ("EPS_RATE_ML_PER_H", EPS_RATE_ML_PER_H),
            ("MATCH_TOLERANCE_PRIMARY", MATCH_TOLERANCE_PRIMARY),
            ("MATCH_TOLERANCE_SENSITIVITY", MATCH_TOLERANCE_SENSITIVITY),
        ]),
        _tbl("时域", [("HORIZONS", list(HORIZONS))] +
             [(f"预测小时数@{h} 记录小时", v)
              for h, v in sorted(PREDICTED_HOURS_BY_HORIZON.items())]),
        _tbl("统计契约", [
            ("N_OUTER_FOLDS", N_OUTER_FOLDS),
            ("N_SEEDS_MIN", N_SEEDS_MIN),
            ("N_BOOTSTRAP", N_BOOTSTRAP),
            ("CI_LEVEL", CI_LEVEL),
            ("INDEPENDENCE_UNIT", INDEPENDENCE_UNIT),
        ]),
        "**词表**",
        "",
    ]
    for name, values in VOCABULARIES.items():
        parts.append(_vocab(name, values))
    parts.append("")
    parts.append(f"**解码规则**： " + "、".join(f"`{d}`" for d in DECODERS))
    parts.append("")
    return "\n".join(parts)
