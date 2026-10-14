"""issue #12 / RED：预设终点的实现与口径声明。

论文 Methods 列出但 Results 未报告的终点：累积剂量 MAE、±200 mL/±10% 容差
准确率、正流 SMAPE、时间稳定性。其中容差终点此前**从未实现**。
"""

from __future__ import annotations

import numpy as np
import pytest

from utils import endpoints as EP


def labels(bits, rates):
    return (np.array(bits, dtype=float), np.array(rates, dtype=float))


# --------------------------------------------------------- 累积剂量

def test_cumulative_dose_mae_units_are_ml():
    """累积剂量 MAE 的单位是 mL，与逐小时 mL/h 不是同一个量。"""
    true = np.array([[100.0], [100.0], [100.0]])
    pred = np.array([[110.0], [110.0], [110.0]])
    out = EP.cumulative_dose_mae(true, pred, step_hours=1.0, level="final")
    assert out["unit"] == "cumulative_ml"
    assert out["mae_ml"] == pytest.approx(30.0)      # 累计差 3×10

    step = float(np.abs(pred - true).mean())
    assert out["mae_ml"] != pytest.approx(step), "累积口径未与逐小时口径区分"


def test_step_hours_scales_cumulative_error():
    true = np.array([[100.0], [100.0]])
    pred = np.array([[100.0], [100.0]])
    a = EP.cumulative_dose_mae(true, pred, step_hours=1.0)
    b = EP.cumulative_dose_mae(true, pred, step_hours=2.0)
    assert a["mae_ml"] == pytest.approx(0.0)
    assert b["mae_ml"] == pytest.approx(0.0)

    pred2 = np.array([[0.0], [0.0]])
    a2 = EP.cumulative_dose_mae(true, pred2, step_hours=1.0)
    b2 = EP.cumulative_dose_mae(true, pred2, step_hours=2.0)
    assert b2["mae_ml"] == pytest.approx(2 * a2["mae_ml"])


def test_cumulative_level_all_vs_final():
    true = np.array([[100.0], [0.0], [0.0]])
    pred = np.array([[0.0], [100.0], [100.0]])       # 终末相同，过程不同
    final = EP.cumulative_dose_mae(true, pred, level="final")
    allv = EP.cumulative_dose_mae(true, pred, level="all")
    assert final["mae_ml"] == pytest.approx(100.0)   # 终末：200 vs 100
    # 逐点口径会看到"先低后高"的抵消过程，数值与终末口径不同
    assert allv["mae_ml"] != pytest.approx(final["mae_ml"]),         "两种 level 给出了相同结果，口径未区分"
    assert allv["n"] == 3 and final["n"] == 1


def test_cumulative_level_is_validated():
    with pytest.raises(ValueError, match="level"):
        EP.cumulative_dose_mae(np.zeros((3, 1)), np.zeros((3, 1)), level="mean")


# --------------------------------------------------------- 容差终点

def test_tolerance_requires_both_criteria():
    """±200 mL **且** ±10% 都要满足，不是二选一。"""
    true = np.array([[1000.0]])          # 终末 1000 mL
    pred = np.array([[1250.0]])          # 差 250 mL：超绝对，也超 10%
    out = EP.tolerance_accuracy(true, pred)
    assert out["n_abs_ok"] == 0 and out["n_pct_ok"] == 0
    assert out["accuracy"] == pytest.approx(0.0)

    # 差 150 mL：绝对 OK（<=200），但相对 15% 超限 -> 不通过
    mid = EP.tolerance_accuracy(true, np.array([[1150.0]]))
    assert mid["n_abs_ok"] == 1 and mid["n_pct_ok"] == 0
    assert mid["accuracy"] == pytest.approx(0.0)


def test_tolerance_passes_when_within_both():
    true = np.array([[1000.0]])
    out = EP.tolerance_accuracy(true, np.array([[1080.0]]))
    assert out["accuracy"] == pytest.approx(1.0)


def test_tolerance_handles_zero_denominator():
    """真实总量为 0 时不得除零，只判绝对容差。"""
    true = np.array([[0.0]])
    out = EP.tolerance_accuracy(true, np.array([[100.0]]))
    assert out["n_zero_denominator"] == 1
    assert out["accuracy"] == pytest.approx(1.0)     # 100 <= 200

    over = EP.tolerance_accuracy(true, np.array([[500.0]]))
    assert over["accuracy"] == pytest.approx(0.0)    # 500 > 200


def test_tolerance_defaults_match_the_manuscript():
    """默认阈值必须是论文方法写的 ±200 mL 与 ±10%。"""
    assert EP.TOL_ABS_ML == 200.0
    assert EP.TOL_PCT == 0.10


def test_tolerance_reports_its_denominator():
    out = EP.tolerance_accuracy(np.ones((4, 3)), np.ones((4, 3)))
    assert out["n"] == 3                              # 逐 (样本, 类别)
    assert "denominator" in out


# --------------------------------------------------------- SMAPE

def test_smape_computed_on_positive_flow_only():
    true = np.array([[100.0], [0.0], [200.0]])
    pred = np.array([[100.0], [500.0], [100.0]])      # 中间是零流
    out = EP.smape_positive_flow(true, pred)
    assert out["n"] == 2, "零流单元不应进入 SMAPE"
    # 2|0-100|/(100+0)... 实际上两个正流单元各贡献 0 与 2*100/300
    assert out["smape"] == pytest.approx((0.0 + 2 * 100 / 300) / 2)


def test_smape_skips_zero_denominator_instead_of_scoring_zero():
    true = np.array([[0.0]])
    pred = np.array([[0.0]])
    out = EP.smape_positive_flow(true, pred, mask=np.array([[True]]))
    assert out["n"] == 0 and out["n_skipped"] == 1
    assert np.isnan(out["smape"])


def test_smape_shape_is_validated():
    with pytest.raises(ValueError, match="形状不一致"):
        EP.smape_positive_flow(np.zeros((3, 1)), np.zeros((3, 2)))


# ------------------------------------------------------- 时间稳定性

def test_oscillation_ratio_flags_jittery_predictions():
    true = np.array([[100.0]] * 5)
    smooth = np.array([[100.0]] * 5)
    jittery = np.array([[100.0], [300.0], [100.0], [300.0], [100.0]])

    # 真实序列本身无变化 -> 比值无定义（不是 0，也不是 1）
    assert np.isnan(EP.temporal_stability(smooth, true)["oscillation_ratio"])

    # 真实有变化时，抖动预测的比值应显著大于 1
    moving_true = np.array([[100.0], [200.0], [300.0], [400.0], [500.0]])
    r = EP.temporal_stability(jittery, moving_true)["oscillation_ratio"]
    assert r > 1.0, f"预测比真实更抖时该比值应大于 1，实际 {r}"


def test_total_variation_is_zero_for_constant_series():
    const = np.array([[50.0]] * 4)
    out = EP.temporal_stability(const, const)
    assert out["total_variation_true_ml"] == pytest.approx(0.0)
    assert np.isnan(out["oscillation_ratio"])


# ------------------------------------------------------- 统一评估

def _mk(n=4, T=3):
    tb = np.zeros((n, T, 3)); tb[..., 0] = 1
    tr = np.full((n, T, 3), 100.0)
    pb = tb.copy()
    pr = tr * 1.1
    return np.concatenate([tb, tr], -1), np.concatenate([pb, pr], -1)


def test_evaluate_all_reports_every_endpoint_with_metadata():
    t, p = _mk()
    out = EP.evaluate_all(t, p, rate_definition="rate")
    for key in ("macro_f1", "micro_f1", "weighted_f1", "bit_accuracy",
                "mae_positive_flow", "mae_step", "cumulative_dose_mae",
                "tolerance_accuracy", "smape_positive_flow", "oscillation_ratio"):
        assert key in out, f"缺少终点 {key}"
        entry = out[key]
        for field in ("value", "aggregation", "unit", "level", "rate_definition"):
            assert field in entry, f"{key} 缺少字段 {field}"
    assert out["macro_f1"]["aggregation"] == "macro"
    assert out["micro_f1"]["aggregation"] == "micro"
    assert out["mae_positive_flow"]["unit"] == "positive_flow_ml_per_h"
    assert out["cumulative_dose_mae"]["unit"] == "cumulative_ml"


def test_rate_definition_must_be_explicit():
    t, p = _mk()
    with pytest.raises(ValueError, match="rate_definition"):
        EP.evaluate_all(t, p, rate_definition="whatever")


@pytest.mark.parametrize("defn", EP.RATE_DEFINITIONS)
def test_all_rate_definitions_are_accepted(defn):
    t, p = _mk()
    out = EP.evaluate_all(t, p, rate_definition=defn)
    assert out["mae_step"]["rate_definition"] == defn


def test_evaluate_all_validates_label_shape():
    with pytest.raises(ValueError, match="标签形状"):
        EP.evaluate_all(np.zeros((2, 3, 5)), np.zeros((2, 3, 5)))


def test_macro_and_micro_differ_on_imbalanced_data():
    """统一评估必须同时给出多种聚合，避免单一口径被当成结论。"""
    n, T = 60, 2
    tb = np.zeros((n, T, 3)); tb[..., 0] = 1; tb[:5, :, 1] = 1
    pb = tb.copy(); pb[10:30, :, 0] = 0; pb[40:45, :, 1] = 1
    tr = np.full((n, T, 3), 100.0); pr = tr
    t = np.concatenate([tb, tr], -1); p = np.concatenate([pb, pr], -1)

    out = EP.evaluate_all(t, p)
    assert out["macro_f1"]["value"] != pytest.approx(out["micro_f1"]["value"], abs=1e-6)
