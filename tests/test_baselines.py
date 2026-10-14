"""issue #9 / RED：朴素时序基线的不变量。

持续性基线与移动平均必须严格因果、遵守标签协议、且 k 只能在验证集上选。
"""

from __future__ import annotations

import numpy as np
import pytest

from utils import baselines as B
from utils import label_protocol as lp


def row(bits, rates):
    return list(bits) + list(rates)


def make(rows):
    return np.array(rows, dtype=np.float32)


TWO = make([
    row([1, 0, 0], [100, 0, 0]),
    row([1, 1, 0], [80, 40, 0]),
    row([0, 0, 0], [0, 0, 0]),
    row([0, 0, 1], [0, 0, 200]),
])


# ------------------------------------------------------------ 严格因果

def test_persistence_only_past():
    """t+1 的预测必须严格等于 t 时刻观测到的值，且不读未来。"""
    got = B.persistence_next(TWO, index=0)
    assert got[lp.RATE_SLICE].tolist() == [100.0, 0.0, 0.0]

    got1 = B.persistence_next(TWO, index=1)
    assert got1[lp.RATE_SLICE].tolist() == [80.0, 40.0, 0.0]


def test_persistence_does_not_read_future():
    """改动 index 之后的数据，预测必须逐位不变。"""
    tampered = TWO.copy()
    tampered[2:] = 999.0
    assert np.array_equal(B.persistence_next(TWO, index=1),
                          B.persistence_next(tampered, index=1))


def test_index_bounds_are_checked():
    with pytest.raises(IndexError):
        B.persistence_next(TWO, index=len(TWO))
    with pytest.raises(IndexError):
        B.moving_average_next(TWO, k=2, index=-1)


# ------------------------------------------------------ 移动平均与 k=1

def test_k1_equals_persistence():
    """k=1 的移动平均必须与持续性完全一致。"""
    for t in range(len(TWO)):
        assert np.allclose(B.moving_average_next(TWO, k=1, index=t),
                           B.persistence_next(TWO, index=t))


def test_moving_average_uses_last_k_hours():
    rates = np.array([[10.0], [20.0], [30.0], [40.0]], dtype=np.float32)
    label = np.hstack([(rates > 0).astype(np.float32),
                       np.zeros((4, 2), dtype=np.float32), rates,
                       np.zeros((4, 2), dtype=np.float32)])
    got = B.moving_average_next(label, k=2, index=3)
    assert got[lp.RATE_SLICE][0] == pytest.approx(35.0)      # (30+40)/2

    got3 = B.moving_average_next(label, k=3, index=3)
    assert got3[lp.RATE_SLICE][0] == pytest.approx(30.0)     # (20+30+40)/3


def test_insufficient_history_uses_available_only():
    """历史不足 k 小时时只用已有小时，不补零、不回填。"""
    got = B.moving_average_next(TWO, k=6, index=1)           # 只有 2 小时
    assert got[lp.RATE_SLICE].tolist() == pytest.approx([90.0, 20.0, 0.0])


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="k 必须 >= 1"):
        B.moving_average_next(TWO, k=0)


# --------------------------------------------- 零流、并发与协议一致性

def test_zero_flow_is_preserved():
    """上一小时无输注时，持续性预测也必须是零流，不得回填。"""
    got = B.persistence_next(TWO, index=2)
    assert got[lp.RATE_SLICE].tolist() == [0.0, 0.0, 0.0]
    assert got[lp.BIT_SLICE].tolist() == [0.0, 0.0, 0.0]


def test_coadministration_is_preserved():
    got = B.persistence_next(TWO, index=1)
    assert got[lp.BIT_SLICE].tolist() == [1.0, 1.0, 0.0]


def test_outputs_satisfy_label_protocol():
    """基线输出必须满足 bit = 1 ⟺ rate > 0，否则无法与主模型同口径比较。"""
    for method, kw in (("persistence", {}), ("moving_average", {"k": 3})):
        pred = B.predict_series(TWO, method=method, **kw)
        assert pred.shape == (len(TWO) - 1, 6)
        assert not lp.validate(pred), f"{method} 输出违反协议"


def test_predict_series_is_causal_and_pairs_with_label_shift():
    """predict_series 的第 t 行必须只由 label[:t+1] 决定。"""
    full = B.predict_series(TWO, method="persistence")
    # predict_series(X) 对长度 L 的序列产出 L-1 行，最后一行的决策时刻是 L-2。
    # 要核验「决策时刻 t 的预测」，截断长度需为 t+2。
    for t in range(len(TWO) - 1):
        truncated = B.predict_series(TWO[:t + 2], method="persistence")
        assert len(truncated) >= 1
        assert np.allclose(truncated[-1], full[t]), f"t={t} 的预测依赖了未来数据"


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="未知的基线方法"):
        B.predict_series(TWO, method="oracle")


def test_shape_is_validated():
    with pytest.raises(ValueError, match="标签形状"):
        B.persistence_next(np.zeros((4, 5)))


# ------------------------------------------------------- k 的选取

def test_select_k_prefers_persistent_series():
    """若序列恒定，k=1 与更大的 k 表现相同，应能选出最小 k。"""
    flat = make([row([1, 0, 0], [100, 0, 0])] * 10)
    k = B.select_k_on_validation({"p1": flat, "p2": flat}, candidates=(1, 3))
    assert k in (1, 3)


def test_select_k_on_validation_reads_only_given_series():
    """选取函数只使用传入（即验证集）的序列，接口上无法接触测试集。"""
    val = {"p1": make([row([1, 0, 0], [100, 0, 0])] * 8)}
    k = B.select_k_on_validation(val, candidates=(1, 2))
    assert k in (1, 2)
    with pytest.raises(ValueError, match="candidates 不能为空"):
        B.select_k_on_validation(val, candidates=())


def test_predict_series_handles_short_input():
    one = make([row([1, 0, 0], [50, 0, 0])])
    assert B.predict_series(one).shape == (0, 6)
    assert B.predict_series(np.zeros((0, 6), dtype=np.float32)).shape == (0, 6)
