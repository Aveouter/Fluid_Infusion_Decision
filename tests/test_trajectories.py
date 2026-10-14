"""issue #81 / RED：窗口 → 患者轨迹的装配、完整性与审计。

对应审稿意见：独立单位必须是**患者**而不是小时窗口；30 小时的部分轨迹
不得被当作 48h/72h 报告；有洞的窗口表必须被拒绝而不是插值补齐。
"""

from __future__ import annotations

import numpy as np
import pytest

from tests import eval_fixtures as F
from utils import trajectories as TR


# --------------------------------------------------------------- 装配与校验

def test_patients_are_grouped_and_hours_sorted():
    trajs = F.unequal_hours()
    assert [t.patient_id for t in trajs] == [f"{F.SYNTH}A", f"{F.SYNTH}B"]
    assert trajs[0].n_hours == 3 and trajs[1].n_hours == 1
    for t in trajs:
        assert np.all(np.diff(t.hours) > 0), "小时必须严格升序"


def test_row_index_maps_back_to_the_window_table():
    trajs = F.unequal_hours()
    assert trajs[0].row_index.tolist() == [0, 1, 2]
    assert trajs[1].row_index.tolist() == [3]


def test_hollow_window_table_is_rejected_not_interpolated():
    """有洞的轨迹必须在装配阶段报错——绝不插值补齐。"""
    pids, hours, bits, rates = F.hollow_trajectory()
    with pytest.raises(ValueError, match="空洞"):
        TR.assemble_trajectories(pids, hours, bits, rates)


def test_contiguity_can_be_relaxed_explicitly():
    pids, hours, bits, rates = F.hollow_trajectory()
    trajs = TR.assemble_trajectories(pids, hours, bits, rates,
                                     require_contiguous=False)
    assert trajs[0].hours.tolist() == [0, 1, 3]
    assert trajs[0].completeness(4)["complete"] is False


def test_bit_rate_invariant_is_enforced():
    """bit = 1 ⟺ rate > 0 是标签协议的核心不变量，评估入口必须挡住违规输入。"""
    with pytest.raises(ValueError, match="bit=1"):
        TR.assemble_trajectories(["P"], [0], np.array([[1.0, 0, 0]]),
                                 np.zeros((1, 3)))


def test_duplicate_hours_are_rejected():
    with pytest.raises(ValueError, match="重复"):
        TR.assemble_trajectories(["P", "P"], [3, 3], np.zeros((2, 3)),
                                 np.zeros((2, 3)))


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="长度不一致"):
        TR.assemble_trajectories(["P", "P"], [0], np.zeros((2, 3)), np.zeros((2, 3)))


# ------------------------------------------------------------------ 完整性

def test_twenty_four_hours_do_not_masquerade_as_seventy_two():
    """30 小时的患者：24h 完整，48/72 不完整。"""
    trajs = F.thirty_hour_patient()
    t = trajs[0]
    assert t.completeness(24)["complete"] is True
    for horizon in (48, 72):
        c = t.completeness(horizon)
        assert c["complete"] is False
        assert c["hours_present"] == 30
        assert c["missing_hours"] == list(range(30, horizon))


def test_seventy_two_recorded_hours_are_seventy_one_predicted():
    """协议算术：72 个记录小时只对应 71 个模型预测小时（首小时是观测锚点）。"""
    from utils import eval_protocol as EP
    assert EP.PREDICTED_HOURS_BY_HORIZON == {24: 23, 48: 47, 72: 71}


def test_completeness_rule_is_validated():
    trajs = F.thirty_hour_patient()
    with pytest.raises(ValueError, match="rule"):
        trajs[0].completeness(24, rule="whatever")


def test_eligibility_rule_requires_the_quality_mask():
    """``all_hours_eligible`` 在没有掩码时不得默默判为合格。"""
    trajs = F.thirty_hour_patient()
    t = trajs[0]
    assert t.completeness(24, rule="all_hours_eligible")["complete"] is False
    mask = np.ones(t.n_hours, dtype=bool)
    assert t.completeness(24, rule="all_hours_eligible",
                          quality_mask=mask)["complete"] is True
    mask[5] = False
    assert t.completeness(24, rule="all_hours_eligible",
                          quality_mask=mask)["complete"] is False


# --------------------------------------------------------------------- 审计

def test_audit_counts_patients_and_hours_separately():
    """窗口/小时总数不是独立样本量——两者必须分别给出。"""
    audit = TR.audit_trajectories(F.unequal_hours(), horizons=(24,))
    assert audit["n_patients"] == 2
    assert audit["n_patient_hours"] == 4
    assert audit["n_targets"] == 4 * 3          # (小时, 类别) 单元
    assert audit["horizon_complete"][24] == 0   # 没有一位患者满 24h


def test_audit_reports_zero_dose_and_zero_flow():
    audit = TR.audit_trajectories(F.zero_dose(), horizons=(24,))
    assert audit["n_zero_dose_patients"] == 2   # Z0 与 Z1 真值累计为 0
    zero_flow = TR.audit_trajectories(F.zero_flow_hour(), horizons=(24,))
    assert zero_flow["n_zero_flow_hours"] == 1  # 中间那一小时三类全零


# ------------------------------------------------------- 跨 fold 泄漏与族群

def test_disjoint_fold_check_catches_leakage():
    a = F.unequal_hours()
    b = F.unequal_hours()
    for t in b:
        t.fold = "train"
    with pytest.raises(AssertionError, match="泄漏"):
        TR.assert_disjoint_fold(a + b)


def test_population_delta_makes_the_change_visible():
    """双轨敏感性分析必须先暴露族群差异，而不是默认两轨是同一批人。"""
    before = TR.audit_trajectories(F.thirty_hour_patient(), horizons=(24, 48))
    after = {"n_patients": 0, "n_patient_hours": 0, "horizon_complete": {24: 0, 48: 0}}
    d = TR.population_delta(before, after)
    assert d["delta_patients"] == 1
    assert d["delta_patient_hours"] == 30
    assert d["delta_horizon_complete"][24] == 1


# --------------------------------------------------------------- 基线与预测

def test_attach_baseline_reads_weight_and_tbsa_indices():
    trajs = F.unequal_hours()
    TR.attach_baseline(trajs, {f"{F.SYNTH}A": [1, 40, 170, 70.0, 24.0, 45.0, 10, 5, 30, 1],
                               f"{F.SYNTH}B": [0, 30, 160, np.nan, 22.0, 35.0, 0, 0, 35, 0]})
    assert trajs[0].weight_kg == pytest.approx(70.0)
    assert trajs[0].tbsa_percent == pytest.approx(45.0)
    # 缺失留 None，不用 0 顶替
    assert trajs[1].weight_kg is None
    assert trajs[1].tbsa_percent == pytest.approx(35.0)


def test_prediction_matrix_validates_shape():
    with pytest.raises(ValueError, match="形状"):
        TR.PredictionMatrix(rates=np.zeros((4, 2)))
    with pytest.raises(ValueError, match="不一致"):
        TR.PredictionMatrix(rates=np.zeros((4, 3)), bits=np.zeros((4, 2)))


def test_predictions_are_aligned_to_patient_rows():
    """预测必须按患者切片，不能把整表原样挂上去。"""
    trajs = F.unequal_hours()
    a = trajs[0]
    assert a.rates_for("speed").shape == (3, 3)
    assert a.rates_for("speed")[0, 0] == pytest.approx(200.0)
    b = trajs[1]
    assert b.rates_for("speed").shape == (1, 3)
    assert b.rates_for("speed")[0, 0] == pytest.approx(0.0)
    assert a.rates_for("nope") is None
