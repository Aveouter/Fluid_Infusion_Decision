"""确定性解码规则（issue #81，Gate A）。

解码**不是指标定义**的一部分，但属于冻结协议：同一个模型输出，用不同的
解码规则会得到不同的动作，而指标必须能说清自己评的是哪一个动作。

本模块只做确定性的、无参数的映射，便于复算与写进论文：

- :func:`decode_bits`          ``1[prob >= tau_c]``
- :func:`decode_action_rate`   ``1[prob >= tau_c] * speed``

为什么阈值必须显式传入
----------------------
``predict_test_set.resolve_thresholds`` 在阈值文件缺失时会**静默回退**
``[0.5, 0.5, 0.5]``。那会让一份没有验证集来源的阈值产出看起来正常的指标。
本模块因此把 ``thresholds`` 设为**必填**：拿不到就报错或输出 NA，不猜。

阈值本身的来源（哪次验证、哪个 split）由调用方记录在结果行的
``threshold`` 与 ``note`` 里。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from utils import eval_protocol as EP
from utils import label_protocol as lp

#: 冻结的解码规则名（见 eval_protocol.DECODERS）。
DECODER_OF_RATE_DEFINITION = {
    "action_rate_uncond": "threshold_gate_uncond",
    "action_rate_cond": "threshold_gate_cond",
}


def _check_thresholds(thresholds, n_classes: int) -> np.ndarray:
    if thresholds is None:
        raise ValueError(
            "thresholds 必填：阈值必须来自验证集。缺失时不得回退 0.5——"
            "那会产出一份没有来源的指标。")
    t = np.asarray(thresholds, dtype=float).ravel()
    if t.size == 0:
        raise ValueError("thresholds 为空：阈值必须来自验证集，不得回退 0.5")
    if t.size != n_classes:
        raise ValueError(f"thresholds 长度 {t.size} 与类别数 {n_classes} 不一致")
    if np.any(~np.isfinite(t)) or np.any(t < 0.0) or np.any(t > 1.0):
        raise ValueError(f"阈值必须落在 [0, 1]，实际 {t.tolist()}")
    return t


def _check_pair(prob, speed) -> (np.ndarray, np.ndarray):
    p = np.asarray(prob, dtype=float)
    s = np.asarray(speed, dtype=float)
    if p.shape != s.shape:
        raise ValueError(f"prob 与 speed 形状不一致：{p.shape} vs {s.shape}")
    if p.ndim != 2 or p.shape[1] != lp.NUM_TYPES:
        raise ValueError(f"形状应为 (N, {lp.NUM_TYPES})，实际 {p.shape}")
    return p, s


def decode_bits(prob: np.ndarray, thresholds, *,
                out: Optional[np.ndarray] = None) -> np.ndarray:
    """硬标签 ``1[prob_c >= tau_c]``，返回 float 数组（与标签位同类型）。"""
    p, _ = _check_pair(prob, np.zeros_like(np.asarray(prob, dtype=float)))
    t = _check_thresholds(thresholds, p.shape[1])
    return (p >= t[None, :]).astype(float)


def decode_action_rate(prob: np.ndarray, speed: np.ndarray, thresholds
                       ) -> np.ndarray:
    """动作速率 ``1[prob_c >= tau_c] * speed_c``（mL/h）。

    这是模型在**固定阈值**下真正会执行的输注动作：概率低于阈值即不输注，
    无论条件强度有多大。它不同于：

    - ``speed``：无条件均值，**不表达"是否输注"**；
    - ``rate = prob * speed``：对多个可能动作按概率加权的期望，不是任何一个
      具体动作（而且 ``prob`` 来自分类头、``speed`` 来自 ZILN，门控了两次）。
    """
    p, s = _check_pair(prob, speed)
    t = _check_thresholds(thresholds, p.shape[1])
    return (p >= t[None, :]).astype(float) * s


def decode_by_name(rate_definition: str, *, prob, speed=None, pos_mean=None,
                   thresholds) -> np.ndarray:
    """按冻结口径名解码，供适配器统一调用。

    ``action_rate_cond`` 需要 ``pos_mean``（条件正流均值）。它**不能**从
    ``speed`` 反推，缺失即报错——由调用方转成 NA + 原因，而不是退而求其次。
    """
    if rate_definition == "action_rate_uncond":
        if speed is None:
            raise ValueError("action_rate_uncond 需要 speed")
        return decode_action_rate(prob, speed, thresholds)
    if rate_definition == "action_rate_cond":
        if pos_mean is None:
            raise ValueError(
                "action_rate_cond 需要 pos_mean（条件正流均值 = exp(mu + sigma^2/2) - 1）；"
                "它不能从 speed 反推，请让模型侧落盘 pos_mean 或改报 NA")
        return decode_action_rate(prob, pos_mean, thresholds)
    raise ValueError(f"{rate_definition!r} 不是解码产出的口径；"
                     f"可选 {sorted(DECODER_OF_RATE_DEFINITION)}")


def rate_definition_of_decoder(decoder: str) -> str:
    """反向查表：解码规则 → 它产出的速率口径名。"""
    for rd, d in DECODER_OF_RATE_DEFINITION.items():
        if d == decoder:
            return rd
    raise ValueError(f"未知的解码规则：{decoder}；可选 {sorted(EP.DECODERS)}")
