"""评估协议 v2 的合成 fixture：结构与真实数据同形，数值全部虚构且**手算可检**。

安全约束
--------
- patient_id 一律用 ``SYNTH_`` 前缀，与真实编号空间不重叠；
- 本模块**不读取** ``data/`` 下任何文件；
- 每个 fixture 的期望值都能在注释里手算出来——若实现改动导致数字变化，
  测试会明确指出是哪一个语义被改掉了，而不是笼统地"数值不符"。

每个 fixture 都刻意做得极小（2–4 位患者、几到几十小时），目的是让审查者
可以拿纸笔核对，而不是相信实现。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import label_protocol as lp
from utils.trajectories import (PatientTrajectory, PredictionMatrix,
                                assemble_trajectories)

SYNTH = "SYNTH_"


def _bits(rates: np.ndarray) -> np.ndarray:
    """按协议由速率导出状态位（bit = 1 ⟺ rate > 0）。"""
    return (np.asarray(rates, dtype=float) > 0).astype(float)


def build(patient_hours: Sequence[Tuple[str, Sequence[float]]],
          *,
          pred_rates: Optional[Sequence[Sequence[float]]] = None,
          pred_prob: Optional[Sequence[Sequence[float]]] = None,
          fold: str = "test",
          ) -> List[PatientTrajectory]:
    """由 ``[(patient_id, [总速率每小时]), ...]`` 造轨迹。

    速率全部记在**晶体**通道（第 0 类），另两类为 0——除非调用方直接给出
    完整的 ``pred_rates``。真实速率取自分组后的时间序列。
    """
    pids, hours, rows = [], [], []
    for pid, series in patient_hours:
        for t, v in enumerate(series):
            pids.append(pid)
            hours.append(t)
            row = np.zeros(lp.NUM_TYPES)
            row[0] = float(v)
            rows.append(row)
    true_rates = np.stack(rows) if rows else np.zeros((0, lp.NUM_TYPES))
    true_bits = _bits(true_rates)

    preds: Dict[str, PredictionMatrix] = {}
    if pred_rates is not None:
        pr = np.asarray(pred_rates, dtype=float)
        preds["speed"] = PredictionMatrix(
            rates=pr, bits=_bits(pr),
            prob=None if pred_prob is None else np.asarray(pred_prob, dtype=float))
    return assemble_trajectories(pids, hours, true_bits, true_rates,
                                 predictions=preds, fold=fold)


# --------------------------------------------------------------------------
# 1. 患者小时数不等：P1 与 all-hours MAE 是**两个不同量纲**的终点
# --------------------------------------------------------------------------

def unequal_hours():
    """A 三个小时、每小时速率误差 100 mL/h；B 一个小时、误差 0。

    手算：
    - **终末累计 P1** = mean_i |V̂_i − V_i| = (|3×100| + |0|) / 2 = **150 mL**
    - **all-hours 患者 MAE**：先按患者取小时均值，再对患者平均
      = (100 + 0) / 2 = **50 mL/h**

    两者数值不同、量纲不同（mL vs mL/h），**不得互相替代**。
    """
    true = [(f"{SYNTH}A", [100.0, 100.0, 100.0]), (f"{SYNTH}B", [0.0])]
    pred = [[200.0, 0, 0]] * 3 + [[0.0, 0, 0]]      # A 每小时高 100；B 恰好 0
    return build(true, pred_rates=pred)


# --------------------------------------------------------------------------
# 2. 零剂量：不进相对容差的分母
# --------------------------------------------------------------------------

def zero_dose():
    """三位患者，真值累计总量分别为 0 / 0 / 1000 mL。

    手算：
    - 前两位是**零剂量**患者 → 只进绝对容差与零剂量桶，**不进相对容差分母**；
    - 零剂量准确率（绝对 ±200 mL）：第 1 位预测 0（达标），第 2 位预测 500
      （不达标）→ **0.5**；
    - 相对容差的分母 = 1（只有第 3 位）。
    """
    true = [(f"{SYNTH}Z0", [0.0]), (f"{SYNTH}Z1", [0.0]),
            (f"{SYNTH}Z2", [100.0] * 10)]
    pred = [[0.0, 0, 0], [500.0, 0, 0]] + [[100.0, 0, 0]] * 10
    return build(true, pred_rates=pred)


# --------------------------------------------------------------------------
# 3. 零流小时：all-hours 有贡献，positive-flow 没有
# --------------------------------------------------------------------------

def zero_flow_hour():
    """一小时三类全零（第 1 小时）。

    真值晶体通道 = [100, 0, 100]，预测 = [150, 0, 150]；其余两个通道全 0。
    掩码按 **(小时, 类别) 单元**计，共 3×3 = 9 个单元。

    手算：
    - **positive-flow MAE**（真值 > ε 的单元：第 0、2 小时的晶体）
      = (50 + 50) / 2 = **50 mL/h**
    - **all-hours MAE**（9 个单元全部计入）
      = (50 + 0 + 0 + 0 + 0 + 0 + 50 + 0 + 0) / 9 = **11.11 mL/h**

    零流单元把 all-hours 拉低——这正是不能把 positive-flow 当成 all-hours 的原因。
    """
    true = [(f"{SYNTH}F", [100.0, 0.0, 100.0])]
    pred = [[150.0, 0, 0], [0.0, 0, 0], [150.0, 0, 0]]
    return build(true, pred_rates=pred)


# --------------------------------------------------------------------------
# 4. 并发多液体：三个通道相互独立，不得互相抵消
# --------------------------------------------------------------------------

def concurrent_fluids():
    """同一小时晶体 100 + 胶体 50，第三类为 0。

    手算：分类和 = (100, 50, 0)，总量 = 150 mL。累加顺序不影响结果。
    """
    pids = [f"{SYNTH}C"] * 2
    hours = [0, 1]
    true = np.array([[100.0, 50.0, 0.0], [100.0, 50.0, 0.0]])
    pred = np.array([[100.0, 50.0, 0.0], [100.0, 50.0, 0.0]])
    return assemble_trajectories(
        pids, hours, _bits(true), true,
        predictions={"speed": PredictionMatrix(rates=pred, bits=_bits(pred))})


# --------------------------------------------------------------------------
# 5. 没有正例的类别：Macro-F1 的缺值规则
# --------------------------------------------------------------------------

def class_without_positives():
    """胶体与葡萄糖（第 1、2 类）在真值里**从未出现**。

    真值：4 行全部只有晶体；预测：晶体为 ``[1, 0, 1, 1]``，其余全 0。

    手算逐类 F1：
    - 晶体：precision = 3/3 = 1，recall = 3/4 = 0.75 → F1 = 2·1·0.75/1.75 = **0.8571**
    - 胶体、葡萄糖：真值无正例，``zero_division=0`` → F1 = **0**
    - **主指标**（固定三类，``zero_division=0``）= (0.8571 + 0 + 0)/3 = **0.2857**
    - **敏感性** ``present_class_macro_f1``（只在有正例的类上取均值）= **0.8571**

    两者相差 3 倍：把无正例类"排除"会显著抬高数字，因此它不是同一个指标，
    不能互换。主指标固定用前者（与稿件口径可比）。
    """
    pids = [f"{SYNTH}P"] * 4
    hours = [0, 1, 2, 3]
    true = np.array([[1.0, 0, 0]] * 4)          # 只有晶体
    pred = np.array([[1.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]])
    return assemble_trajectories(
        pids, hours, _bits(true), true,
        predictions={"speed": PredictionMatrix(rates=pred, bits=_bits(pred))})


# --------------------------------------------------------------------------
# 6. 正流 SMAPE 的手算单元
# --------------------------------------------------------------------------

#: 手算 SMAPE 单元：``(真值, 预测, 期望贡献)``。
#: SMAPE 定义为 ``200·|ŷ−y| / (|ŷ|+|y|)``（百分比）。
SMAPE_CELLS = (
    (100.0, 100.0, 0.0),            # 完全命中
    (200.0, 100.0, 200 * 100 / 300),  # 2·|100−200| / 300 → 66.67%
)


# --------------------------------------------------------------------------
# 7. 时域完整性：30 小时的患者不是 48h/72h
# --------------------------------------------------------------------------

def thirty_hour_patient():
    """30 个记录小时的患者。

    手算：24h 完整；48h 与 72h 均**不完整** → 必须输出
    ``NA + na_reason="insufficient_horizon"``，不得把 30 小时当 48/72 求和。
    """
    true = [(f"{SYNTH}H", [10.0] * 30)]
    pred = [[10.0, 0, 0]] * 30
    return build(true, pred_rates=pred)


def hollow_trajectory():
    """结构不完整：小时序号有洞（0,1,3）。

    必须在装配阶段就被拒绝——**不插值补齐**，也不静默跳过。
    """
    pids = [f"{SYNTH}GAP"] * 3
    hours = [0, 1, 3]
    true = np.array([[10.0, 0, 0]] * 3)
    return pids, hours, _bits(true), true


# --------------------------------------------------------------------------
# 8. 变点事件：最大基数匹配 vs 最近邻贪心
# --------------------------------------------------------------------------

def change_events_max_cardinality():
    """同通道真实变点在 10、11；预测变点在 9、10（±1h 容差）。

    手算：
    - **最大基数匹配**：10→9、11→10 → 2 个 TP（precision 2/2、recall 2/2）
    - **最近邻贪心**（先取距离 0 的 10→10）→ 只剩 1 个 TP（recall 1/2）

    贪心不是最优，这正是 v2 弃用它的原因。
    """
    true_cp = np.zeros((14, 1), dtype=bool)
    true_cp[10, 0] = true_cp[11, 0] = True
    pred_cp = np.zeros((14, 1), dtype=bool)
    pred_cp[9, 0] = pred_cp[10, 0] = True
    return true_cp, pred_cp


def change_events_exact():
    """真实变点在 10、11；预测只有 10。

    手算：精确匹配 recall = 1/2、precision = 1/1；11 记为未匹配。
    """
    true_cp = np.zeros((14, 1), dtype=bool)
    true_cp[10, 0] = true_cp[11, 0] = True
    pred_cp = np.zeros((14, 1), dtype=bool)
    pred_cp[10, 0] = True
    return true_cp, pred_cp


def change_rate_series_duplicate_hit():
    """**重复命中**的可复算实例（速率序列，供 v1 接口直接消费）。

    真值：t<10 为 100，t=10 为 200，t>=11 为 300 → 变点在 **{10, 11}**。
    预测：t<10 为 100，t>=10 为 200          → 变点只有 **{10}**。

    两者的变点集合正是 :func:`change_events_exact`。
    v1 的 ``change_timing_error`` 用最近邻且允许重复命中，因此 t=11 会再次
    "命中" 同一个预测变点 10，报出 ``match_rate = 2/2 = 1.0``——看起来完美。
    v2 的一对一匹配只给 1 个 TP：recall = 1/2。
    """
    n = 14
    true = np.full((n, 1), 100.0)
    true[10] = 200.0
    true[11:] = 300.0
    pred = np.full((n, 1), 100.0)
    pred[10:] = 200.0
    return true, pred


# --------------------------------------------------------------------------
# 9. 速率口径分叉：prob 低而 speed 高
# --------------------------------------------------------------------------

def rate_decoupling():
    """一小时：``speed = 800``（无条件均值很大）但 ``prob = 0.05``（阈值 0.5 下不输注）。

    手算（τ = 0.5）：
    - ``speed``              = 800
    - ``rate = prob × speed`` = 0.05 × 800 = 40
    - ``action_rate_uncond`` = 1[0.05 ≥ 0.5] × 800 = **0**

    真值该小时为 0。于是 ``speed`` 口径报出 800 的误差，而 ``action_rate``
    报出 0 —— 后者才是模型真正会执行的动作。
    """
    true = [(f"{SYNTH}R", [0.0])]
    pred = [[800.0, 0, 0]]
    prob = [[0.05, 0.0, 0.0]]
    return build(true, pred_rates=pred, pred_prob=prob)


# --------------------------------------------------------------------------
# 10. 真实零值 vs 缺失：语义不同，不得合并
# --------------------------------------------------------------------------

def zero_vs_missing():
    """两位患者：一位是**真实零输注**（速率 0），一位是**记录缺失**（哨兵 −1）。

    经 :func:`utils.label_protocol.normalize` 后二者的速率都是 0——**在标签
    矩阵里已经不可区分**。本 fixture 把这个事实固定下来，使协议文档里
    "目标侧缺失率无法计算"这一限制有可执行的证据，而不是一句声明。
    """
    raw_missing = np.array([[lp.SENTINEL_UNRECORDED] * lp.NUM_TYPES
                            + [0.0] * lp.NUM_TYPES])
    raw_zero = np.zeros((1, 2 * lp.NUM_TYPES))
    return lp.normalize(raw_missing), lp.normalize(raw_zero)


# --------------------------------------------------------------------------
# 11. N1 调速一致性：方向一致率与"到处乱变"的对照
# --------------------------------------------------------------------------

def titration_series():
    """真值单调上升 100 → 200 → 300（δ = 50 mL/h 下两步都是真实变化事件）。

    两个预测对照（Δr̂ 的基准是 ``t`` 时刻**真实**速率）：

    - ``following`` = [100, 220, 340]：Δr̂ = +120 / +120，两步方向都对 → **DC = 1.0**
    - ``constant``  = [100, 100, 100]：Δr̂ = 0 / 0，从不调整 → **DC = 0.0**

    真值刻意取单调上升：若真值先升后降，恒定预测会在下降那一步"蒙对"方向，
    看起来不那么糟——这里要的是干净的 1.0 与 0.0 对照。
    """
    true = [100.0, 200.0, 300.0]
    following = [100.0, 220.0, 340.0]
    constant = [100.0, 100.0, 100.0]
    return true, following, constant


def titration_always_changing():
    """真值 100×3 → 200×3（只有一个真实变化事件）；预测每小时 ±80 抖动。

    手算（δ = 50，逐 (时刻, 类别) 单元计，晶体通道有值、另两类恒 0）：

    - 真实变化事件只有第 2 步（100→200）；预测在该步 Δr̂ = 180 − 100 = +80，
      方向一致 → **DC = 1.0**（只看 DC 会把这个"到处乱变"的预测器评为完美）
    - 非变化单元 = 5 步 × 3 类 − 1 个事件 = 14；其中预测"变了"的是晶体通道的
      第 0、3 步（Δr̂ = +80、−100）→ **false_change_rate = 2/14 ≈ 0.143**

    这正是必须把 DC 与误报率**配对报告**的原因：DC 高而误报率不为 0，
    说明它靠乱动换来的方向命中。另注意恒零的两个通道会**稀释**误报率——
    它们是真实存在的"可以误报却没报"的机会，不是噪声。
    """
    true = [100.0, 100.0, 100.0, 200.0, 200.0, 200.0]
    jitter = [100.0, 180.0, 100.0, 180.0, 100.0, 180.0]
    return true, jitter


# --------------------------------------------------------------------------
# 12. E1 ≡ E3：同窗同锚点下终末累计绝对误差必然相同
# --------------------------------------------------------------------------

def anchor_identity():
    """首小时锚点 ``V0 = 100``，其后 3 个预测小时真值各 10、预测各 20。

    手算：
    - E1（72h 锚定）：真值 100+30 = 130，预测 100+60 = 160 → |Δ| = **30**
    - E3（纯预测窗）：真值 30，预测 60 → |Δ| = **30**

    两者逐位相等：``|(V0+Ŝ) − (V0+S)| = |Ŝ − S|``。
    因此 E3 **不能**作为 E1 误差的独立稳健性证据；E3 只在相对容差分母、
    累计量绝对水平、时域解释三处提供信息。
    """
    v0 = 100.0
    true_tail = np.array([10.0, 10.0, 10.0])
    pred_tail = np.array([20.0, 20.0, 20.0])
    return {"v0": v0, "true_tail": true_tail, "pred_tail": pred_tail}
