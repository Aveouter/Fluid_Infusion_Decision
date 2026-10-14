"""issue #81 / RED：主要终点 P1 / P2 / P3 的口径。

对应审稿意见：累计剂量的分母与时域必须写明；容差的两条判据要分别报告、
零剂量不能靠除零蒙混；Macro-F1 的缺值规则必须固定，且与稿件口径可比。
"""

from __future__ import annotations

import numpy as np
import pytest

from tests import eval_fixtures as F
from utils import endpoints as EP


# ---------------------------------------------------------------- P1 累计

def test_p1_is_patient_first_and_equals_150_ml():
    """P1 = 先患者内累积 → 绝对误差 → 患者等权平均。

    A：3 小时，每小时误差 100 mL/h → 累计误差 300 mL
    B：1 小时，误差 0            → 累计误差 0 mL
    P1 = (|300| + |0|) / 2 = **150 mL**
    """
    out = EP.cumulative_summary(F.unequal_hours(), "speed", horizon=24,
                                origin="first_predicted_hour",
                                require_complete=False)  # 详见 test_horizon_gate_*
    assert out["available"] is True
    assert out["n_patients_eligible"] == 2
    assert out["per_class"]["crystal"]["mae_ml"] == pytest.approx(150.0)
    assert out["per_class"]["total"]["mae_ml"] == pytest.approx(150.0)
    assert out["estimand"] == "pure_prediction_window_volume"


def test_all_hours_mae_is_a_different_quantity_from_p1():
    """同一份数据的 all-hours 患者 MAE = 50 mL/h——与 150 mL 不是同一个量。

    量纲不同（mL/h vs mL），**不得互相替代**。这条用例与上一条刻意分开，
    因为把两者混为一谈正是待修的缺陷之一。
    """
    out = EP.rate_error_summary(F.unequal_hours(), "speed", mask="all",
                                fluid_class="crystal")
    assert out["mae_ml_per_h"] == pytest.approx(50.0)
    assert out["unit"] == "mL/h"
    assert out["n_patients"] == 2


def test_patient_hours_are_not_independent_targets():
    """窗口/小时总数不是样本量：n_patients 与 n_targets 必须分开给出。"""
    out = EP.cumulative_summary(F.unequal_hours(), "speed", horizon=24,
                                origin="first_predicted_hour",
                                require_complete=False)  # 详见 test_horizon_gate_*
    assert out["n_patients_eligible"] == 2
    assert out["n_predicted_hours_total"] == 4          # 3 + 1
    assert out["per_class"]["total"]["n_targets"] == 4


def test_insufficient_horizon_is_excluded_not_truncated():
    """30 小时的患者不能当 48h/72h 用——**默认闸门**必须把它挡在外面。

    这是本协议最容易犯的错：把"有多少小时算多少小时"的和当成 48h/72h 总量。
    默认 ``require_complete=True``，排除原因写进 ``exclusion_reasons``。
    """
    trajs = F.thirty_hour_patient()

    # 24h 覆盖完整（该 fixture 的小时序号是 0..29）→ 纳入
    ok = EP.cumulative_summary(trajs, "speed", horizon=24)
    assert ok["n_patients_eligible"] == 1
    assert ok["n_patients_excluded"] == 0

    # 48h / 72h 都不完整 → 排除，且**不得**退化成"用现有的 30 小时求和"
    for h in (48, 72):
        out = EP.cumulative_summary(trajs, "speed", horizon=h)
        assert out["available"] is False
        assert out["n_patients_eligible"] == 0
        assert out["n_patients_excluded"] == 1
        assert "insufficient_horizon" in out["exclusion_reasons"]
        assert out["per_class"] == {}, "不得产出任何以部分轨迹冒充时域的数值"


def test_completeness_gate_can_be_disabled_explicitly():
    """算术用例可以显式关掉闸门——但必须写出来，不能是默认行为。"""
    trajs = F.thirty_hour_patient()
    out = EP.cumulative_summary(trajs, "speed", horizon=72,
                                require_complete=False)
    assert out["n_patients_eligible"] == 1
    assert out["n_predicted_hours_total"] == 30, "用的是 30 个小时，不是 72 个"


def test_tolerance_and_component_gates_agree_with_cumulative():
    """三个与累计量有关的终点必须用**同一套**资格规则，否则分母会各说各话。"""
    trajs = F.thirty_hour_patient()
    for h in (24, 48, 72):
        cum = EP.cumulative_summary(trajs, "speed", horizon=h)
        tol = EP.tolerance_summary(trajs, "speed", horizon=h)
        comp = EP.component_sum_error(trajs, "speed", horizon=h)
        expected = cum["n_patients_eligible"] > 0
        assert tol["available"] is expected, f"时域 {h} 的容差资格与累计不一致"
        assert comp["available"] is expected, f"时域 {h} 的分量资格与累计不一致"


def test_per_fluid_breakdown_exposes_cancellation():
    """总量误差会被通道间抵消——分液体必须并列报告。"""
    out = EP.cumulative_summary(F.concurrent_fluids(), "speed", horizon=24,
                                origin="first_predicted_hour",
                                require_complete=False)
    assert set(out["per_class"]) == {"total", "crystal", "colloid", "water"}
    # 完美预测：逐类误差都为 0
    for fluid in out["per_class"]:
        assert out["per_class"][fluid]["mae_ml"] == pytest.approx(0.0)


def test_component_sum_catches_cross_channel_cancellation():
    """C2：分量误差和 > 总量误差 ⟺ 存在通道间正负抵消。

    构造（单小时）：真值 晶体 500 + 胶体 500 = 总量 1000；
    预测 晶体 0 + 胶体 1000 = 总量 1000。
    → 总量误差 **0**（看起来完美），分量误差和 = |0−500| + |1000−500| = **1000**。
    """
    pids, hours = [f"{F.SYNTH}K"], [0]
    true = np.array([[500.0, 500.0, 0.0]])
    pred = np.array([[0.0, 1000.0, 0.0]])
    from utils.trajectories import PredictionMatrix, assemble_trajectories
    trajs = assemble_trajectories(
        pids, hours, (true > 0).astype(float), true,
        predictions={"speed": PredictionMatrix(rates=pred)})

    out = EP.component_sum_error(trajs, "speed", horizon=24,
                                 origin="first_predicted_hour",
                                 require_complete=False)
    assert out["total_abs_mae_ml"] == pytest.approx(0.0)      # 总量"完美"
    assert out["component_sum_mae_ml"] == pytest.approx(1000.0)
    assert out["cancellation_gap_ml"] == pytest.approx(1000.0)


# ---------------------------------------------------------------- P2 容差

def test_p2_reports_absolute_and_relative_separately():
    """绝对与相对是**两个独立主终点**，各自带分母。"""
    out = EP.tolerance_summary(F.zero_dose(), "speed", horizon=24,
                               require_complete=False)
    # 绝对：3 位合格患者里 2 位达标（Z0: 0；Z2: 0；Z1: 500 > 200）
    assert out["abs"]["value"] == pytest.approx(2 / 3)
    assert out["abs"]["n_patients"] == 3
    # 相对：只有 1 位非零剂量患者，且达标
    assert out["rel"]["value"] == pytest.approx(1.0)
    assert out["rel"]["n_patients"] == 1
    assert out["rel"]["n_excluded_zero_dose"] == 2


def test_zero_dose_patients_never_enter_the_relative_denominator():
    """除零不是达标：零剂量患者必须被排除并计数，而不是算作通过。"""
    out = EP.tolerance_summary(F.zero_dose(), "speed", horizon=24,
                               require_complete=False)
    assert out["zero_dose"]["n_patients"] == 2
    assert out["zero_dose"]["accuracy_abs"] == pytest.approx(0.5)  # 预测 0 与 500
    assert out["zero_dose"]["pred_total_ml"] == [0.0, 500.0]
    assert out["rel"]["n_patients"] + out["rel"]["n_excluded_zero_dose"] == \
        out["abs"]["n_patients"], "相对容差的分母没有把零剂量患者排除干净"


def test_p2_tolerances_are_configurable_but_default_to_the_manuscript():
    out = EP.tolerance_summary(F.zero_dose(), "speed", horizon=24,
                               require_complete=False)
    assert out["tol_abs_ml"] == pytest.approx(EP.TOL_ABS_ML)
    assert out["tol_pct"] == pytest.approx(EP.TOL_PCT)
    assert EP.TOL_ABS_ML == 200.0 and EP.TOL_PCT == 0.10


def test_p2_without_eligible_patients_is_na_not_zero():
    """没有任何合格患者时必须报 NA，不能返回 0（那不是"全部不达标"）。"""
    pids, hours = [f"{F.SYNTH}Q"], [40]      # 从第 40 小时才有数据
    from utils.trajectories import assemble_trajectories
    trajs = assemble_trajectories(pids, hours, np.array([[0.0, 0, 0]]),
                                  np.array([[0.0, 0, 0]]))
    out = EP.tolerance_summary(trajs, "speed", horizon=24)
    assert out["available"] is False
    assert out["n_patients_eligible"] == 0


# ------------------------------------------------------------------ P3 F1

def test_p3_macro_f1_uses_fixed_classes_with_zero_division_zero():
    """主指标固定三类、``zero_division=0``；敏感性只在有正例的类上取均值。

    本 fixture 手算：晶体 F1 = 0.8571，两类无正例记 0
    → 主指标 (0.8571 + 0 + 0)/3 = **0.2857**
    → 敏感性 0.8571。两者相差 3 倍，**不是同一个指标**。
    """
    t = F.class_without_positives()[0]
    out = EP.multilabel_prf(t.true_bits.astype(int), t.pred["speed"].bits.astype(int))

    assert out["per_class"]["crystal"]["f1"] == pytest.approx(6 / 7, abs=1e-6)
    assert out["per_class"]["colloid"]["support"] == 0
    assert out["macro_f1"] == pytest.approx(2 / 7, abs=1e-6)
    assert out["present_class_macro_f1"] == pytest.approx(6 / 7, abs=1e-6)
    assert out["n_classes_without_positives"] == 2
    assert out["macro_rule"] == "fixed_classes_zero_division_0"
    assert out["macro_f1"] != pytest.approx(out["present_class_macro_f1"])


def test_p3_support_is_reported_per_class():
    """每类支持样本数必须列出——没有它，macro-F1 无法解读。"""
    t = F.class_without_positives()[0]
    out = EP.multilabel_prf(t.true_bits.astype(int), t.pred["speed"].bits.astype(int))
    assert out["per_class"]["crystal"]["support"] == 4
    assert out["per_class"]["colloid"]["support"] == 0
    assert out["n_targets"] == 4 * 3


def test_auc_na_when_a_class_has_a_single_label():
    """某类只有单一标签时 AUC 不可计算 → None + 原因，不补造 0.5。"""
    t = F.class_without_positives()[0]
    prob = np.full((4, 3), 0.3)
    out = EP.multilabel_ranking(t.true_bits.astype(int), prob)
    assert out["auc_per_class"][0] is None            # 晶体全为 1
    assert out["auc_macro_n_classes"] == 0
    assert out["auc_na_reason"] == "single_class_labels_in_split"
    assert np.isnan(out["auc_macro"]), "不可计算的类不得被算进 macro AUC"
