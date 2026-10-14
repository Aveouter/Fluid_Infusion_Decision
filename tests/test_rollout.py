"""issue #11 / RED：教师强制与开环的分离。

对应审稿人 1 的第 2 条意见：图 5/图 6 究竟用的是真实临床历史，还是模型自己的
先前预测？两者必须能分别产出并逐图标明。
"""

from __future__ import annotations

import numpy as np
import pytest

from utils import label_protocol as lp
from utils import rollout as RO


def row(bits, rates):
    return list(bits) + list(rates)


def make_labels(rows):
    return np.array(rows, dtype=np.float32)


LABELS = make_labels([
    row([1, 0, 0], [100, 0, 0]),
    row([1, 1, 0], [80, 40, 0]),
    row([0, 0, 0], [0, 0, 0]),
    row([0, 0, 1], [0, 0, 200]),
    row([1, 0, 0], [60, 0, 0]),
])
FEATS = np.arange(5 * 9, dtype=np.float32).reshape(5, 9)


def recording_predictor(seen):
    """记录每次输入；返回一个简单可预测的输出。"""
    def f(inp):
        seen.append(np.asarray(inp).copy())
        return inp[-6:]                       # 回显输入里的标签部分
    return f


# ------------------------------------------------- 教师强制只用真实历史

def test_teacher_forced_ground_truth_history():
    """教师强制下，每次输入里的标签与累计量都必须来自真实序列。"""
    seen = []
    r = RO.rollout(recording_predictor(seen), FEATS, LABELS, mode="teacher_forced")
    true_cum = RO.cum_from_labels(LABELS)

    for t, inp in enumerate(seen):
        assert np.allclose(inp[9:13], true_cum[t]), f"第 {t} 步的 cum 非真实值"
        assert np.allclose(inp[13:], LABELS[t]), f"第 {t} 步的历史标签非真实值"
    assert len(seen) == len(LABELS) - 1


def test_teacher_forced_target_is_next_step():
    seen = []
    r = RO.rollout(recording_predictor(seen), FEATS, LABELS, mode="teacher_forced")
    assert np.allclose(r["target"], LABELS[1:])


# ------------------------------------------------- 开环禁止读取未来标签

def test_open_loop_no_future_labels():
    """开环第 t 步的输入只能含 t 之前的预测，不得出现 t+1 的真实标签。"""
    seen = []
    RO.rollout(recording_predictor(seen), FEATS, LABELS, mode="open_loop")
    for t, inp in enumerate(seen):
        for u in range(t + 1, len(LABELS)):
            assert not np.allclose(inp[13:], LABELS[u]), \
                f"第 {t} 步输入里出现了第 {u} 步的真实标签"


def test_open_loop_does_not_use_true_cum():
    """开环必须按预测重算 cum —— 只回填标签而沿用真实 cum 是伪开环。

    用一个"恒返回固定标签"的预测器：此后累计量应只按该固定值增长，
    与真实 cum 不符。
    """
    fixed = np.array(row([1, 0, 0], [50, 0, 0]), dtype=np.float32)

    def predict(_):
        return fixed

    seen = []
    RO.rollout(lambda x: (seen.append(x.copy()), fixed)[1], FEATS, LABELS,
               mode="open_loop")
    true_cum = RO.cum_from_labels(LABELS)

    # 第 2 步起，输入里的 cum 应等于 initial_cum + 预测体积，而非真实 cum
    assert not np.allclose(seen[1][9:13], true_cum[1]), "开环偷用了真实 cum"
    expected = true_cum[0] + RO.inst_from_label(fixed)
    assert np.allclose(seen[1][9:13], expected)


def test_error_accumulation_is_not_reset():
    """开环中偏差会累积；每一步都重置就不是真开环。"""
    def predict_biased(inp):
        lab = inp[-6:].copy()
        lab[3:] = lab[3:] + 30.0              # 每步高估 30 mL/h
        return lp.normalize(lab[None, :])[0]

    r = RO.rollout(predict_biased, FEATS, LABELS, mode="open_loop")
    cum = r["cum_used"]
    assert len(cum) >= 3
    # 累计量应逐步偏离，而不是每步归零
    growth = np.diff(cum[:, 3])
    assert np.all(growth > 0), f"累计量未单调增长：{growth}"


def test_open_loop_prediction_feeds_back():
    """回填的是**预测**：第二步的输入标签应等于第一步的输出。"""
    seen = []

    def predict(inp):
        seen.append(np.asarray(inp).copy())
        lab = inp[-6:].copy()
        lab[3:] = 7.0
        return lp.normalize(lab[None, :])[0]

    RO.rollout(predict, FEATS, LABELS, mode="open_loop")
    assert len(seen) >= 2
    assert np.allclose(seen[1][-6:][3:], 7.0), "预测未被回填成历史"


# --------------------------------------------------------- 累计与时钟

def test_volume_accumulation_and_clock():
    """逐步累计体积 = Σ(bit × rate × Δt)，与 data_loader 的口径一致。"""
    cum = RO.cum_from_labels(LABELS, step_hours=1.0)
    assert cum[0, 3] == pytest.approx(100.0)
    assert cum[1, 3] == pytest.approx(100.0 + 120.0)
    assert cum[2, 3] == pytest.approx(cum[1, 3])          # 零流不增加
    assert cum[3, 3] == pytest.approx(cum[1, 3] + 200.0)
    assert np.all(np.diff(cum[:, 3]) >= 0), "累计量非单调"


def test_step_hours_scales_volume():
    a = RO.cum_from_labels(LABELS, step_hours=1.0)
    b = RO.cum_from_labels(LABELS, step_hours=2.0)
    assert b[-1, 3] == pytest.approx(2 * a[-1, 3])


def test_horizon_and_start_do_not_overflow():
    for start in range(len(LABELS)):
        r = RO.rollout(lambda x: x[-6:].copy(), FEATS, LABELS,
                       mode="teacher_forced", start=start)
        assert len(r["pred"]) == len(LABELS) - 1 - start
    r2 = RO.rollout(lambda x: x[-6:].copy(), FEATS, LABELS, horizon=2)
    assert len(r2["pred"]) == 2


def test_out_of_range_start_is_rejected():
    with pytest.raises(ValueError, match="start"):
        RO.rollout(lambda x: x[-6:].copy(), FEATS, LABELS, start=99)


# --------------------------------------------------------- 其它约束

def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        RO.rollout(lambda x: x[-6:].copy(), FEATS, LABELS, mode="semi")


def test_feature_length_must_match():
    with pytest.raises(ValueError, match="长度不一致"):
        RO.rollout(lambda x: x[-6:].copy(), FEATS[:3], LABELS)


def test_prediction_shape_is_validated():
    with pytest.raises(ValueError, match="预测长度"):
        RO.rollout(lambda x: np.zeros(3), FEATS, LABELS)


def test_outputs_obey_label_protocol():
    r = RO.rollout(lambda x: x[-6:].copy(), FEATS, LABELS, mode="open_loop")
    assert not lp.validate(r["pred"])


def test_compare_modes_reports_both():
    out = RO.compare_modes(lambda x: x[-6:].copy(), FEATS, LABELS)
    assert set(out) == {"teacher_forced", "open_loop"}
    for v in out.values():
        assert v["n_steps"] == len(LABELS) - 1


def test_cum_final_reflects_true_series_in_teacher_forced():
    """教师强制下，最后一次输入用的 cum 必须是真实序列在**该步**的值。

    注意 cum_used 记录的是每次**输入**时的累计量，故其末项对应第 T-2 步，
    而非序列终点（终点是预测目标，本就不该被输入看到）。
    """
    out = RO.compare_modes(lambda x: x[-6:].copy(), FEATS, LABELS)
    tf = out["teacher_forced"]
    true_cum = RO.cum_from_labels(LABELS)
    assert tf["cum_final_used"] == pytest.approx(float(true_cum[-2, 3]))
    assert tf["cum_final_true"] == pytest.approx(float(true_cum[-1, 3]))
    assert tf["cum_final_used"] < tf["cum_final_true"]
