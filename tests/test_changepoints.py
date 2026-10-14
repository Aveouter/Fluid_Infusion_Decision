"""issue #10 / RED：变更点定义、时点误差与分层评估。

对应审稿意见：总体 MAE 易被大量稳定小时稀释；需单独报告临床医生改变
输注速率时的性能，并核实约一小时滞后的说法。
"""

from __future__ import annotations

import numpy as np
import pytest

from utils import changepoints as CP


def rates(seq):
    return np.array(seq, dtype=float).reshape(-1, 1)


# --------------------------------------------------------- 变更点定义

def test_change_point_definition_absolute():
    """绝对变化达到阈值即构成变更。"""
    r = rates([100, 100, 160, 160])          # +60 >= 50
    cp = CP.detect_changes(r, abs_thresh=50, rel_thresh=1e9)   # 关掉相对
    assert cp[:, 0].tolist() == [False, False, True, False]


def test_change_point_definition_relative():
    """高速率下的小幅调整由相对阈值捕获。"""
    r = rates([100, 100, 130, 130])          # +30 < 50 绝对，但 30/130 = 0.23
    assert not CP.detect_changes(r, abs_thresh=50, rel_thresh=0.5)[2, 0]
    assert CP.detect_changes(r, abs_thresh=50, rel_thresh=0.2)[2, 0]


def test_relative_threshold_needs_a_base():
    """基数过小时不套用相对阈值，避免把噪声当变更。"""
    r = rates([0, 0, 10, 10])                # 10 mL/h，基数 < REL_BASE
    cp = CP.detect_changes(r, abs_thresh=50, rel_thresh=0.01, rel_base=50)
    assert not cp[2, 0]


def test_first_step_is_never_a_change():
    r = rates([500, 500])
    assert not CP.detect_changes(r)[0, 0]


def test_shape_is_validated():
    with pytest.raises(ValueError, match="rates 应为"):
        CP.detect_changes(np.zeros(5))


@pytest.mark.parametrize("seq,expect", [
    ([0, 0, 100], "off_to_on"),
    ([100, 100, 0], "on_to_off"),
    ([100, 100, 300], "rate_up"),
    ([300, 300, 100], "rate_down"),
])
def test_transition_types_are_distinguished(seq, expect):
    """开启/停止与升/降必须分开统计，临床含义不同。"""
    r = rates(seq)
    cp = CP.detect_changes(r)
    types = CP.change_type_matrix(r, cp)
    assert types[2, 0] == expect


def test_zero_to_positive_and_back():
    r = rates([0, 0, 200, 200, 0, 0])
    cp = CP.detect_changes(r)
    types = CP.change_type_matrix(r, cp)
    assert types[2, 0] == "off_to_on"
    assert types[4, 0] == "on_to_off"


def test_multilabel_channels_are_independent():
    r = np.array([[0, 0], [100, 0], [100, 200]], dtype=float)
    cp = CP.detect_changes(r)
    assert cp[1].tolist() == [True, False]
    assert cp[2].tolist() == [False, True]


# --------------------------------------------------------- 时点误差

def test_timing_error_scores_lag_and_lead():
    """预测滞后 1 小时记 +1，提前 1 小时记 -1。"""
    true = rates([0, 0, 200, 200, 200])
    lag = rates([0, 0, 0, 200, 200])          # 变更点晚一步
    lead = rates([0, 200, 200, 200, 200])     # 变更点早一步

    e_lag = CP.change_timing_error(lag, true, tolerance=2)
    assert e_lag["mean_lag_hours"] == pytest.approx(1.0)

    e_lead = CP.change_timing_error(lead, true, tolerance=2)
    assert e_lead["mean_lag_hours"] == pytest.approx(-1.0)


def test_unmatched_change_points_are_not_counted_as_on_time():
    """超出容差的未命中必须记为 unmatched，不能按 0 混进均值。"""
    true = rates([0, 0, 200, 200, 200, 200, 200])
    pred = rates([0, 0, 0, 0, 0, 0, 0])       # 完全不预测变更
    e = CP.change_timing_error(pred, true, tolerance=1)
    assert e["n_unmatched"] == e["n_true_change_points"]
    assert e["n_matched"] == 0
    assert np.isnan(e["mean_lag_hours"]), "未命中却给出了时延均值"


def test_perfect_prediction_has_zero_lag():
    true = rates([0, 0, 200, 200, 0])
    e = CP.change_timing_error(true, true, tolerance=1)
    assert e["mean_lag_hours"] == pytest.approx(0.0)
    assert e["match_rate"] == pytest.approx(1.0)


# --------------------------------------------------------- 分层评估

def test_stable_vs_changed_stratification():
    """变更小时与稳定小时必须分开报告。"""
    true = rates([100, 100, 100, 400, 400])
    pred = rates([100, 100, 100, 100, 100])       # 变更后没跟上
    cp = CP.detect_changes(true)
    m = CP.stratified_metrics(pred, true, cp)

    assert m["changed"]["n_cells"] == 1
    assert m["changed"]["mae"] == pytest.approx(300.0)
    assert m["stable"]["mae"] == pytest.approx(0.0)
    assert m["changed"]["mae"] > m["stable"]["mae"]


@pytest.mark.parametrize("radius", [1, 2])
def test_neighborhood_mask_widens_coverage(radius):
    cp = np.zeros((6, 1), dtype=bool)
    cp[3, 0] = True
    near = CP.neighborhood_mask(cp, radius)
    expected = set(range(max(0, 3 - radius), min(6, 3 + radius + 1)))
    assert set(np.flatnonzero(near[:, 0])) == expected


def test_neighborhood_radius_is_validated():
    with pytest.raises(ValueError, match="radius"):
        CP.neighborhood_mask(np.zeros((3, 1), dtype=bool), radius=-1)


def test_zero_flow_cells_are_excluded():
    """零流单元不计入回归误差，与 #9 的对照口径一致。

    这里零流单元（t=0）恰好与变更点同处一个邻域，无论分到哪一组，
    都不应出现在统计里；预测在该处给出 999 也不能影响任何一组的 MAE。
    """
    true = rates([0, 100, 100])
    pred = rates([999, 100, 100])
    cp = CP.detect_changes(true)
    assert cp[1, 0] and true[0, 0] == 0.0

    m = CP.stratified_metrics(pred, true, cp)
    # 正流单元是 t=1,2；t=0 为零流，不得出现在任何一组里。
    # 注意 changed ⊂ changed_neighborhood，两组本就重叠，不可相加求解总数。
    assert m["stable"]["n_cells"] == 0
    assert m["changed"]["n_cells"] == 1
    assert m["changed_neighborhood"]["n_cells"] == 2, \
        "邻域组应只含正流的 t=1,2，零流的 t=0 被算进来了"
    for name, g in m.items():
        if g["n_cells"]:
            assert g["mae"] == pytest.approx(0.0), f"{name} 被零流单元的 999 污染"


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="形状不一致"):
        CP.stratified_metrics(np.zeros((3, 1)), np.zeros((3, 2)), np.zeros((3, 1), bool))


def test_per_type_metrics_cover_all_types():
    true = rates([0, 0, 200, 200, 0, 0])
    pred = rates([0, 0, 150, 150, 0, 0])
    cp = CP.detect_changes(true)
    out = CP.per_type_metrics(pred, true, cp)
    assert set(out) == set(CP.CHANGE_TYPES)
    assert out["off_to_on"]["n_cells"] == 1
    assert out["off_to_on"]["mae"] == pytest.approx(50.0)


def test_patient_level_aggregation_weights_patients_equally():
    """患者窗口数不等时，逐患者平均与逐单元平均会不同。"""
    true = np.array([[100.0], [100.0], [100.0], [400.0], [400.0]])
    pred = np.array([[200.0], [200.0], [200.0], [400.0], [400.0]])
    cp = CP.detect_changes(true)
    pids = ["A", "A", "A", "B", "B"]

    # radius=0：只看变更点本身，避免邻域把稳定单元也划走
    cell = CP.stratified_metrics(pred, true, cp, radius=0)["stable"]["mae"]
    pat = CP.patient_level_stratification(pred, true, cp, pids, radius=0)["stable"]
    # 稳定单元：t=0,1,2（误差各 100）与 t=4（误差 0）-> 逐单元 (300+0)/4 = 75
    assert cell == pytest.approx(75.0)
    # 逐患者：A 的三个窗口全为 100；B 只有 t=4，误差 0 -> (100+0)/2 = 50
    assert pat == pytest.approx(50.0)
    assert cell != pytest.approx(pat), "两种聚合层级未区分开"


def test_patient_level_validates_lengths():
    with pytest.raises(ValueError, match="长度不一致"):
        CP.patient_level_stratification(np.zeros((3, 1)), np.zeros((3, 1)),
                                        np.zeros((3, 1), bool), ["A", "B"])
