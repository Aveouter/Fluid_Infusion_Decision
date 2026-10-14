"""BurnResu 目标标签协议（v1）。

本模块是目标语义的**单一事实来源**：标签列的定义、不变量、规范化与校验都放这里，
`utils/data_loader.py`、训练脚本与论文 Methods 应引用同一套定义，避免口径漂移。

协议 v1 要点
-----------
1. **三类液体相互独立，是多标签任务**，不是 softmax 互斥分类。
   同一小时可以有任意子集（含全部三类）同时处于输注状态。
2. **速率是权威字段**：``bit = 1 ⟺ rate > 0``。位由速率推导，
   与生成已发表结果的 ``data2pkl.py`` 口径一致（该脚本按速度重写状态位）。
3. **没有独立的「无输注」类别**。三类位全为 0 的小时即为无输注（零流量）小时。
4. **速率单位 mL/h**；时间轴步长固定 1 小时，故某小时体积 = 速率 × 1h。
5. **哨兵值 -1 表示「未记录」**，规范化时置 0；规范化后必须满足上述不变量。

已知限制（详见 docs/label_protocol.md）
------------------------------------
- 原始状态列与速率的语义不同（前者可能表示「该小时有入量记录」而非持续泵速）。
  本协议统一以速率为准，并在文档中记录了该取舍。
- 推注（bolus）与持续输注在当前目标中未加区分。
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

#: 协议版本。标签语义发生不兼容变更时必须递增，并同步 docs/label_protocol.md。
PROTOCOL_VERSION = "1.0"

#: 液体类别顺序（与标签列 0/1/2 一一对应）。
LIQUID_TYPES: tuple = ("crystal", "colloid", "water")

#: 中文列名，对应真实数据表头。
LIQUID_TYPES_ZH: tuple = ("晶体", "胶体", "水")

#: 完整列顺序：3 个状态位 + 3 个速率。
COLUMN_ORDER: tuple = (
    "crystal", "colloid", "water",
    "crystal_rate", "colloid_rate", "water_rate",
)

NUM_TYPES = len(LIQUID_TYPES)
BIT_SLICE = slice(0, NUM_TYPES)
RATE_SLICE = slice(NUM_TYPES, 2 * NUM_TYPES)

#: 时间轴步长（小时）。协议固定为 1h。
DT_HOURS = 1.0

#: 「未记录」哨兵值。
SENTINEL_UNRECORDED = -1.0

#: 状态位合法取值。
VALID_BITS = (0.0, 1.0)


def _as_array(label) -> np.ndarray:
    """转成 (T, 6) 的 float32 数组；不复制调用方的数据用于写回。"""
    arr = np.asarray(label, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2 * NUM_TYPES:
        raise ValueError(
            f"标签形状应为 (T, {2 * NUM_TYPES})，实际为 {arr.shape}"
        )
    return arr


def normalize(label) -> np.ndarray:
    """按协议 v1 规范化标签，返回**新数组**（不修改输入）。

    处理顺序：
    1. 状态位中的哨兵 -1 置 0；
    2. 由速率推导状态位：``bit = 1 ⟺ rate > 0``；
    3. 速率保持原值（负速率会被规范化为 0，因其不具物理意义）。
    """
    arr = np.array(label, dtype=np.float32, copy=True)
    if arr.ndim != 2 or arr.shape[1] != 2 * NUM_TYPES:
        raise ValueError(
            f"标签形状应为 (T, {2 * NUM_TYPES})，实际为 {arr.shape}"
        )

    bits = arr[:, BIT_SLICE]
    rates = arr[:, RATE_SLICE]

    # 1) 哨兵 → 0
    bits[bits == SENTINEL_UNRECORDED] = 0.0
    # 3) 负速率无物理意义
    rates[rates < 0] = 0.0
    # 2) 速率是权威字段
    bits[:] = (rates > 0).astype(np.float32)

    arr[:, BIT_SLICE] = bits
    arr[:, RATE_SLICE] = rates
    return arr


def validate(label, *, allow_sentinel: bool = False) -> List[str]:
    """校验标签是否满足协议 v1，返回违规描述列表（空列表表示合规）。

    参数
    ----
    allow_sentinel: 为 True 时容忍状态位中的 -1（尚未规范化的原始数据）。
    """
    arr = _as_array(label)
    problems: List[str] = []

    bits = arr[:, BIT_SLICE]
    rates = arr[:, RATE_SLICE]

    if not allow_sentinel:
        unexpected = np.unique(bits[~np.isin(bits, VALID_BITS)])
        if unexpected.size:
            problems.append(
                f"状态位出现非法取值 {unexpected.tolist()}，合法值为 {list(VALID_BITS)}"
            )

    if np.any(rates < 0):
        n = int(np.sum(rates < 0))
        problems.append(f"速率出现 {n} 个负值，协议要求非负")

    # 核心不变量：bit = 1 ⟺ rate > 0
    expected = (rates > 0).astype(np.float32)
    mismatch = bits != expected
    if allow_sentinel:
        mismatch &= bits != SENTINEL_UNRECORDED
    if np.any(mismatch):
        n = int(np.sum(mismatch))
        problems.append(
            f"状态位与速率不一致的单元格有 {n} 个（要求 bit=1 ⟺ rate>0）"
        )

    return problems


def assert_valid(label, **kwargs) -> None:
    """不满足协议时抛出 AssertionError，附全部违规描述。"""
    problems = validate(label, **kwargs)
    if problems:
        raise AssertionError("标签不符合协议 v%s：\n- %s" % (
            PROTOCOL_VERSION, "\n- ".join(problems)))


def zero_flow_mask(label) -> np.ndarray:
    """返回 (T,) 布尔数组：该小时三类状态位全为 0（无输注）。"""
    arr = _as_array(label)
    return ~np.any(arr[:, BIT_SLICE] > 0, axis=1)


def active_mask(label) -> np.ndarray:
    """返回 (T, 3) 布尔数组：每类在该小时是否处于输注状态。"""
    arr = _as_array(label)
    return arr[:, BIT_SLICE] > 0


def coadministration_mask(label) -> np.ndarray:
    """返回 (T,) 布尔数组：该小时是否同时有两种及以上液体在输注。"""
    return np.sum(active_mask(label), axis=1) >= 2


def cumulative_volume(label, dt_hours: float = DT_HOURS) -> np.ndarray:
    """按类累计体积，返回 (3,)。

    定义：``V_c = Σ_t active[c,t] × rate[c,t] × dt``（单位 mL）。
    因协议中 active 与 rate>0 等价，等价于对速率直接求和再乘 dt。
    """
    arr = _as_array(label)
    active = active_mask(arr).astype(np.float32)
    rates = arr[:, RATE_SLICE] * active
    return np.sum(rates, axis=0) * float(dt_hours)


def hourly_volume(label, dt_hours: float = DT_HOURS) -> np.ndarray:
    """返回 (T, 3) 的逐小时体积（mL）。"""
    arr = _as_array(label)
    active = active_mask(arr).astype(np.float32)
    return arr[:, RATE_SLICE] * active * float(dt_hours)


def aggregate_infusion_records(
    records_by_hour: dict,
    num_hours: int,
    dt_hours: float = DT_HOURS,
) -> np.ndarray:
    """把逐小时的入量记录聚合为符合协议的标签矩阵。

    参数
    ----
    records_by_hour:
        ``{小时序号: [(液体序号, 该记录体积 mL), ...]}``。液体序号按
        :data:`LIQUID_TYPES` 的顺序。
    num_hours:
        时间轴长度；未出现在 ``records_by_hour`` 中的小时按无输注处理。

    聚合规则（协议 v1，回答 R4-5 的重叠医嘱问题）
    -------------------------------------------
    1. **同一小时、同一类**的多条记录**求和**为小时体积，不丢弃、不取首条。
    2. 小时速率 = 该小时该类总体积 / ``dt_hours``，单位 mL/h。
    3. **不同类**相互独立，同小时可同时存在（多标签）。
    4. **推注与持续输注不加区分**：二者都折算为该小时的等效平均速率。
       这是本协议的已知限制，因为原始记录无法可靠区分二者。
    5. 体积为 0 或缺失的记录不产生状态位（该小时该类为无输注）。

    返回 ``(num_hours, 6)`` 的 float32 数组，满足 :func:`validate`。
    """
    if num_hours < 0:
        raise ValueError("num_hours 必须非负")

    out = np.zeros((num_hours, 2 * NUM_TYPES), dtype=np.float32)
    for hour, records in records_by_hour.items():
        if not 0 <= int(hour) < num_hours:
            raise ValueError(f"小时序号 {hour} 越界（时间轴长度 {num_hours}）")
        h = int(hour)
        for liquid_idx, volume_ml in records:
            idx = int(liquid_idx)
            if not 0 <= idx < NUM_TYPES:
                raise ValueError(f"液体序号 {liquid_idx} 超出 0..{NUM_TYPES - 1}")
            if volume_ml is None:
                continue
            vol = float(volume_ml)
            if vol < 0:
                raise ValueError(f"体积不得为负：{vol}")
            out[h, RATE_SLICE][idx] += vol

    # 体积 → 速率
    out[:, RATE_SLICE] = out[:, RATE_SLICE] / float(dt_hours)
    # 速率 → 状态位
    out[:, BIT_SLICE] = (out[:, RATE_SLICE] > 0).astype(np.float32)
    return out


def summarize(label) -> dict:
    """汇总一段标签的分布，用于报告与数据字典核对。

    返回的 ``zero_flow_hours`` 即「三类均无输注」的小时数——
    论文中不得把类别占比之和的余数直接解释为零输液，必须用此口径。
    """
    arr = _as_array(label)
    active = active_mask(arr)
    total = int(arr.shape[0])
    per_type = {t: int(np.sum(active[:, i])) for i, t in enumerate(LIQUID_TYPES)}
    zero_flow = int(np.sum(zero_flow_mask(arr)))
    return {
        "hours": total,
        "per_type_hours": per_type,
        "per_type_ratio": {k: (v / total if total else 0.0) for k, v in per_type.items()},
        "zero_flow_hours": zero_flow,
        "zero_flow_ratio": (zero_flow / total if total else 0.0),
        "coadministration_hours": int(np.sum(coadministration_mask(arr))),
        "all_three_hours": int(np.sum(np.sum(active, axis=1) == NUM_TYPES)),
        "note": "多标签任务，各类占比之和可超过 100%；无输注由三类全零定义，"
                "不能由占比余数推出。",
    }


def stack_records(records: Iterable[Sequence[np.ndarray]]) -> np.ndarray:
    """把逐患者的标签序列拼成 (N, 6)，便于整体校验与汇总。"""
    parts = [_as_array(r) for r in records if len(r) > 0]
    if not parts:
        return np.zeros((0, 2 * NUM_TYPES), dtype=np.float32)
    return np.concatenate(parts, axis=0)
