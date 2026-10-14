"""issue #81 / RED：§2A 医学过程指标 C1 / C2 / N1 / PS-TC / N2。

对应审稿意见：预测性能之外还要有医学解释；但**不得**把"复现医生操作"说成
"改善结局"，也不得在数据不支持时用别的变量顶替。
"""

from __future__ import annotations

import numpy as np
import pytest

from tests import eval_fixtures as F
from utils import clinical_endpoints as CE
from utils.trajectories import PredictionMatrix, assemble_trajectories


def _with_baseline(trajs, weights, tbsas):
    base = {}
    for traj, w, t in zip(trajs, weights, tbsas):
        base[traj.patient_id] = [1, 40, 170, w, 24.0, t, 0, 0, t, 0]
    from utils.trajectories import attach_baseline
    attach_baseline(trajs, base)
    return trajs


def _single(true_series, pred_series, pid=None):
    n = len(true_series)
    true = np.zeros((n, 3))
    true[:, 0] = true_series
    pred = np.zeros((n, 3))
    pred[:, 0] = pred_series
    return assemble_trajectories(
        [pid or f"{F.SYNTH}N"] * n, list(range(n)),
        (true > 0).astype(float), true,
        predictions={"speed": PredictionMatrix(rates=pred)})


# --------------------------------------------------------------------- C1

def test_c1_matches_hand_computation():
    """C1 = V_cryst(0–24h) / (weight_kg × TBSA%)，单位 mL/kg/%TBSA。

    手算（晶体通道）：
    - A：3 小时 × 100 mL/h = 300 mL；70 kg × 45% = 3150 → 0.095238
    - B：0 mL；50 kg × 30% = 1500 → 0
    - 患者等权平均 = (0.095238 + 0) / 2 = **0.047619**
    """
    trajs = _with_baseline(F.unequal_hours(), [70.0, 50.0], [45.0, 30.0])
    out = CE.early_resuscitation_exposure(trajs, "speed", horizon=24,
                                        require_complete=False)

    assert out["available"] is True
    assert out["unit"] == "mL/kg/%TBSA"
    assert out["fluid_class"] == "crystal"
    assert out["true_mean"] == pytest.approx((300 / 3150 + 0) / 2, rel=1e-6)
    assert out["n_patients_eligible"] == 2
    assert out["missing_fraction"] == pytest.approx(0.0)
    assert out["evidence_tier"] == CE.TIER_MUST


def test_c1_guidance_value_is_reference_only_never_a_target():
    """指南的 2 mL/kg/%TBSA 是起始估算建议，**不是**金标准目标。

    它只能以 ``guidance_only`` 出现，且不得被当作达标率的分母。
    """
    trajs = _with_baseline(F.unequal_hours(), [70.0, 50.0], [45.0, 30.0])
    out = CE.early_resuscitation_exposure(trajs, "speed", horizon=24,
                                        require_complete=False)

    g = out["guidance"]
    assert g["value_ml_per_kg_tbsa"] == pytest.approx(2.0)
    assert g["evidence_tier"] == CE.TIER_GUIDANCE
    assert g["role"] == "reference_only"
    assert "不是" in g["note"] and "分母" in g["note"]
    # 计数字段只是分布位置，措辞上不得叫达标率
    assert "n_above_guidance_true" in out
    assert "不构成达标率" in out["note"]


def test_c1_reports_unavailable_patients_instead_of_guessing():
    """体重或 TBSA 缺失/超范围时不猜、不用 0 顶替，报出比例与原因。"""
    trajs = F.unequal_hours()
    from utils.trajectories import attach_baseline
    attach_baseline(trajs, {
        f"{F.SYNTH}A": [1, 40, 170, np.nan, 24.0, 45.0, 0, 0, 45, 0],   # 体重缺失
        f"{F.SYNTH}B": [0, 30, 160, 500.0, 22.0, 30.0, 0, 0, 30, 0],    # 体重超范围
    })
    out = CE.early_resuscitation_exposure(trajs, "speed", horizon=24,
                                        require_complete=False)

    assert out["available"] is False
    assert out["n_patients_unavailable"] == 2
    assert out["missing_fraction"] == pytest.approx(1.0)
    assert set(out["unavailable_reasons"]) == {
        "weight_or_tbsa_missing", "weight_or_tbsa_out_of_range"}
    assert out["na_reason"] == "no_patient_with_usable_weight_and_tbsa"


def test_c1_interpretation_scope_does_not_claim_efficacy():
    trajs = _with_baseline(F.unequal_hours(), [70.0, 50.0], [45.0, 30.0])
    out = CE.early_resuscitation_exposure(trajs, "speed", horizon=24,
                                        require_complete=False)
    scope = out["interpretation_scope"]
    assert "不是最优处方" in scope and "疗效" in scope
    assert out["observation_vs_imputation"] == "observed"
    assert "ABA" in out["clinical_definition_source"]


# --------------------------------------------------------------------- C2

def test_c2_lists_every_fluid_separately():
    trajs = F.concurrent_fluids()
    out = CE.per_fluid_dose_error(trajs, "speed", horizon=24,
                                  origin="first_predicted_hour",
                                  require_complete=False)
    assert set(out["per_fluid"]) == {"total", "crystal", "colloid", "water"}
    assert out["evidence_tier"] == CE.TIER_MUST


def test_c2_flags_composition_failure_that_total_hides():
    """总量完全正确、液体组成完全错——C2 必须把它暴露出来。"""
    true = np.array([[500.0, 500.0, 0.0]])
    pred = np.array([[0.0, 1000.0, 0.0]])
    trajs = assemble_trajectories(
        [f"{F.SYNTH}K"], [0], (true > 0).astype(float), true,
        predictions={"speed": PredictionMatrix(rates=pred)})
    out = CE.per_fluid_dose_error(trajs, "speed", horizon=24,
                                  origin="first_predicted_hour",
                                  require_complete=False)
    assert out["per_fluid"]["total"]["mae_ml"] == pytest.approx(0.0)
    assert out["component_sum"]["component_sum_mae_ml"] == pytest.approx(1000.0)
    assert out["component_sum"]["cancellation_gap_ml"] == pytest.approx(1000.0)


# --------------------------------------------------------------------- N1

def test_n1_direction_concordance_separates_following_from_constant():
    """跟随型 DC = 1.0；恒定型 DC = 0.0。"""
    true, following, constant = F.titration_series()

    good = CE.titration_concordance(_single(true, following), "speed")
    assert good["overall"]["direction_concordance"] == pytest.approx(1.0)
    assert good["overall"]["n_events"] == 2

    flat = CE.titration_concordance(_single(true, constant), "speed")
    assert flat["overall"]["direction_concordance"] == pytest.approx(0.0)
    assert flat["overall"]["n_events"] == 2


def test_n1_requires_false_change_rate_to_be_read_alongside():
    """只看 DC 会偏袒"到处乱变"的预测器，误报率必须配对给出。"""
    true, jitter = F.titration_always_changing()
    out = CE.titration_concordance(_single(true, jitter), "speed")

    assert out["overall"]["direction_concordance"] == pytest.approx(1.0)
    assert out["false_change_rate"] > 0.0
    assert out["n_false_changes"] == 2
    assert out["n_non_change_cells"] == 14
    assert "change_hour_mae_ml_per_h" in out["overall"]


def test_n1_is_labelled_proposed_and_not_clinically_validated():
    true, following, _ = F.titration_series()
    out = CE.titration_concordance(_single(true, following), "speed")
    assert out["evidence_tier"] == CE.TIER_PROPOSED
    assert out["clinical_applicability"] == "not_clinically_validated"
    assert out["expert_review_status"] == "not_reviewed"
    assert "不是" in out["interpretation_scope"]


def test_n1_reports_na_when_there_are_no_change_events():
    """没有真实变化事件时 DC 无定义——报 NA，不报 0 或 1。"""
    out = CE.titration_concordance(_single([100.0] * 4, [100.0] * 4), "speed")
    assert out["available"] is False
    assert out["na_reason"] == "no_true_change_events"


def test_n1_refuses_physiology_strata_names():
    """小时段不是生理状态：把生理分层的名字用在这里必须报错。"""
    true, following, _ = F.titration_series()
    with pytest.raises(ValueError, match="PS-TC"):
        CE.titration_concordance(_single(true, following), "speed",
                                 stratum="low_urine_output")


def test_n1_stratifies_by_hour_band_without_claiming_physiology():
    """可用的分层只有小时段与观测速率水平，且必须标明不是生理状态。

    步长取 60 mL/h（> δ = 50），否则不构成变化事件、分层会是空的。
    """
    n = 30
    true = [100.0 + 60.0 * (i % 3) for i in range(n)]
    pred = [100.0 + 60.0 * ((i + 1) % 3) for i in range(n)]
    out = CE.titration_concordance(_single(true, pred), "speed",
                                   stratum="hour_band")
    assert out["available"] is True
    assert set(out["per_stratum"]) == {"0-23h", "24-47h"}
    assert out["stratum"] == "hour_band"
    # 分层名只描述时间，不含任何生理含义
    assert all("h" in k for k in out["per_stratum"])


# ------------------------------------------------------------------ PS-TC

def test_ps_tc_is_na_when_no_physiology_is_available():
    """生理状态不可得时 PS-TC 必须不可估计，**不得**退化成小时段分层。"""
    true, following, _ = F.titration_series()
    trajs = _single(true, following)
    out = CE.ps_tc(trajs, "speed")

    assert out["available"] is False
    assert out["na_reason"] == "no_valid_physiology_state_available"
    assert out["evidence_tier"] == CE.TIER_NOT_ASSESSABLE
    assert out["missing_fraction"] == pytest.approx(1.0)
    assert "尿量" in out["note"] and "冒充" in out["note"]


def test_ps_tc_runs_when_real_strata_are_supplied():
    """真的给了生理分层时它才能算——接口存在，但数据前提必须由调用方满足。"""
    true, following, _ = F.titration_series()
    trajs = _single(true, following)
    strata = {trajs[0].patient_id: ["low_uop"] * 3}
    out = CE.ps_tc(trajs, "speed", physiology_strata=strata)
    assert out["available"] is True
    assert "low_uop" in out["per_stratum"]


# --------------------------------------------------------------------- N2

def test_n2_cete_is_reported_normalised_and_absolute():
    trajs = _with_baseline(F.unequal_hours(), [70.0, 50.0], [45.0, 30.0])
    out = CE.cumulative_exposure_trajectory_error(trajs, "speed", horizons=(24,),
                                                  require_complete=False)
    e = out["horizons"][24]
    assert e["cete_ml"] is not None
    assert e["cete_ml_per_kg"] is not None
    assert out["unit_normalized"] == "mL/kg" and out["unit_absolute"] == "mL"
    assert out["evidence_tier"] == CE.TIER_PROPOSED


def test_n2_cete_is_na_for_weight_when_weight_is_missing():
    """体重缺失时归一化版本不可算——跳过并报人数，不用 0 顶替。"""
    trajs = F.unequal_hours()          # 未 attach 基线 → 无体重
    out = CE.cumulative_exposure_trajectory_error(trajs, "speed", horizons=(24,),
                                                  require_complete=False)
    e = out["horizons"][24]
    assert e["cete_ml"] is not None                       # 绝对版本仍可算
    assert e["cete_ml_per_kg"] is None
    assert e["na_reason"] == "weight_unavailable_for_normalisation"
    assert e["n_skipped_missing_weight"] == 2


def test_n2_cete_exposes_trajectory_failure_hidden_by_final_total():
    """早期过量 + 晚期不足：终末总量可能相近，但轨迹误差很大。"""
    true = [100.0, 100.0, 100.0, 100.0]
    even = [100.0, 100.0, 100.0, 100.0]         # 轨迹完全正确
    swapped = [200.0, 200.0, 0.0, 0.0]          # 同样 400 mL，但轨迹错位

    t_even = CE.cumulative_exposure_trajectory_error(
        _single(true, even), "speed", horizons=(24,),
        require_complete=False)["horizons"][24]["cete_ml"]
    t_swap = CE.cumulative_exposure_trajectory_error(
        _single(true, swapped), "speed", horizons=(24,),
        require_complete=False)["horizons"][24]["cete_ml"]

    assert t_even == pytest.approx(0.0)
    assert t_swap > 0.0, "终末总量相同但轨迹不同时必须被 CETE 捕捉"
