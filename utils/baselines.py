"""朴素时序基线：持续性（last-value）与因果移动平均。

背景（issue #9，对应 R1-1 / R3-6 / R4-4）
-----------------------------------------
输液速率具有强自相关性。若 BurnResu 只是复制上一小时，那么报告 MAE/R²
就不能解读为"捕捉了独特的生理动态"。审稿人 1 从轨迹图的约一小时滞后
提出这一质疑，因此必须把朴素基线放进同一张表。

设计约束
--------
- **严格因果**：预测 t+1 只允许读取 ≤ t 的信息。
- **遵守标签协议**（``utils/label_protocol.py``）：输出必须满足
  ``bit = 1 ⟺ rate > 0``，因此先算速率再由速率导出状态位。
- **确定性**：无随机数，同输入必得同输出，便于复现与对照。

与真实输液的对应：速率列保留物理量纲，但对照实验应在与主模型完全相同的
掩码、聚合层级与指标计算器下进行，否则差异无法归因。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from utils import label_protocol as lp

#: 默认移动平均窗口（小时）。k=1 退化为持续性。
DEFAULT_K = 1


def _as_2d(label) -> np.ndarray:
    arr = np.asarray(label, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2 * lp.NUM_TYPES:
        raise ValueError(f"标签形状应为 (T, {2 * lp.NUM_TYPES})，实际 {arr.shape}")
    return arr


def persistence_next(label, *, index: Optional[int] = None) -> np.ndarray:
    """持续性预测：下一个小时 = 当前小时观测到的 (bit, rate)。

    ``index`` 指定决策时刻 ``t``；默认取序列最后一步。返回形状 ``(6,)``：
    前三列为状态位、后三列为速率，且已按协议规范化。
    """
    arr = _as_2d(label)
    t = len(arr) - 1 if index is None else int(index)
    if t < 0 or t >= len(arr):
        raise IndexError(f"决策时刻 {t} 超出序列范围 [0, {len(arr) - 1}]")
    return lp.normalize(arr[t:t + 1])[0]


def moving_average_next(label, k: int = DEFAULT_K, *,
                        index: Optional[int] = None) -> np.ndarray:
    """因果移动平均：用最近 ``k`` 个小时的速率均值预测下一小时。

    - ``k=1`` 与 :func:`persistence_next` 等价（有用例保证）。
    - 历史不足 ``k`` 小时时，用**已有的全部**历史小时（不补零、不回填）；
      序列开头没有历史时返回全零（无输注）。
    - 状态位由速率导出（``bit = 1 ⟺ rate > 0``），与协议一致。
    """
    arr = _as_2d(label)
    if k < 1:
        raise ValueError(f"k 必须 >= 1，实际 {k}")
    t = len(arr) - 1 if index is None else int(index)
    if t < 0 or t >= len(arr):
        raise IndexError(f"决策时刻 {t} 超出序列范围 [0, {len(arr) - 1}]")

    lo = max(0, t - k + 1)
    window = arr[lo:t + 1]
    out = np.zeros(2 * lp.NUM_TYPES, dtype=np.float32)
    if len(window) == 0:
        return out

    rates = window[:, lp.RATE_SLICE]
    out[lp.RATE_SLICE] = rates.mean(axis=0)
    out[lp.BIT_SLICE] = (out[lp.RATE_SLICE] > 0).astype(np.float32)
    return out


def predict_series(label, method: str = "persistence", k: int = DEFAULT_K) -> np.ndarray:
    """对一条序列产出逐步预测：第 t 行是**对 t+1 的预测**。

    返回形状 ``(T-1, 6)``；与之配对的目标是 ``label[1:]``。
    第一步（t=0）没有历史，持续性给出全零。
    """
    arr = _as_2d(label)
    if len(arr) < 2:
        return np.zeros((0, 2 * lp.NUM_TYPES), dtype=np.float32)

    fn = (persistence_next if method == "persistence"
          else (lambda a, index: moving_average_next(a, k=k, index=index)))
    if method not in ("persistence", "moving_average"):
        raise ValueError(f"未知的基线方法：{method}")

    return np.stack([fn(arr, index=t) for t in range(len(arr) - 1)], axis=0)


def select_k_on_validation(series_by_patient: dict, candidates=(1, 2, 3, 6),
                           metric: str = "mae_positive_flow") -> int:
    """在**验证集**上选择移动平均窗口 k。

    绝不允许用测试集选 k（见 issue #6 的仅训练/验证拟合原则）。
    """
    if not candidates:
        raise ValueError("candidates 不能为空")

    scores = {}
    for k in candidates:
        errs = []
        for label in series_by_patient.values():
            pred = predict_series(label, method="moving_average", k=k)
            true = _as_2d(label)[1:]
            if len(pred) == 0:
                continue
            mask = true[:, lp.RATE_SLICE] > 0
            if mask.sum() == 0:
                continue
            if metric == "mae_positive_flow":
                errs.append(float(np.abs(pred[:, lp.RATE_SLICE][mask]
                                         - true[:, lp.RATE_SLICE][mask]).mean()))
            elif metric == "mae_all":
                errs.append(float(np.abs(pred[:, lp.RATE_SLICE]
                                         - true[:, lp.RATE_SLICE]).mean()))
            else:
                raise ValueError(f"未知的 metric：{metric}")
        scores[k] = float(np.mean(errs)) if errs else float("inf")
    return min(scores, key=scores.get)
