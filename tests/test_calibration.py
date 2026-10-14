"""issue #23 / RED：分类校准与预测不确定性（R4-9）。"""
from __future__ import annotations

import numpy as np
import pytest

from utils import calibration as CA


def test_brier_is_zero_for_perfect_probabilities():
    y = np.array([[1, 0], [0, 1]], dtype=float)
    b = CA.brier_per_class(y, y)
    assert b["mean"] == pytest.approx(0.0)


def test_brier_is_maximal_for_confidently_wrong():
    y = np.array([[1, 0]], dtype=float)
    p = np.array([[0.0, 1.0]], dtype=float)
    b = CA.brier_per_class(y, p)
    assert b["class_0"] == pytest.approx(1.0)
    assert b["class_1"] == pytest.approx(1.0)


def test_brier_validates_shapes():
    with pytest.raises(ValueError, match="形状不一致"):
        CA.brier_per_class(np.zeros((3, 2)), np.zeros((3, 3)))


def test_ece_is_zero_when_perfectly_calibrated():
    """完美校准时 ECE 应为 0。"""
    p = np.array([0.2, 0.4, 0.6, 0.8] * 25)
    y = (np.random.default_rng(0).random(len(p)) < p).astype(float)
    # 大样本下频率趋近概率；这里用小容差
    e = CA.expected_calibration_error(y, p, n_bins=4)
    assert e["ece"] < 0.15


def test_ece_detects_systematic_overconfidence():
    """恒预测 0.9 而实际只有 10% 阳性 —— ECE 必须显著。"""
    p = np.full(1000, 0.9)
    y = np.zeros(1000)
    y[:100] = 1.0
    e = CA.expected_calibration_error(y, p, n_bins=10)
    assert e["ece"] == pytest.approx(0.8, abs=0.05)


def test_ece_reports_empty_bins():
    """空箱不参与加权，但必须报出，避免密集分箱造成 ECE 偏小的错觉。"""
    p = np.full(100, 0.05)
    y = np.zeros(100)
    e = CA.expected_calibration_error(y, p, n_bins=10)
    assert e["n_empty_bins"] == 9
    assert e["n"] == 100


def test_ece_validates_n_bins():
    with pytest.raises(ValueError, match="n_bins"):
        CA.expected_calibration_error(np.zeros(3), np.zeros(3), n_bins=0)


def test_reliability_curve_shapes_and_counts():
    rng = np.random.default_rng(1)
    p = rng.random(200)
    y = (rng.random(200) < p).astype(float)
    r = CA.reliability_curve(y, p, n_bins=5)
    assert len(r["mean_confidence"]) == len(r["fraction_positive"]) == 5
    assert sum(r["count"]) == 200
    assert len(r["bin_edges"]) == 6


def test_overconfidence_gap_sign_convention():
    y = np.array([[0, 0], [1, 0]], dtype=float)
    p = np.array([[0.9, 0.9], [0.9, 0.9]], dtype=float)
    g = CA.overconfidence_gap(y, p)
    assert g["class_0"] > 0, "过度自信时差值应为正"
    assert "过度自信" in g["note"]

    # 反向对照：低估时差值为负
    low = CA.overconfidence_gap(y, np.zeros_like(p))
    assert low["class_0"] < 0


def test_interval_coverage_requires_nominal_level_context():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo = np.array([0.0, 0.0, 0.0, 0.0])
    hi = np.array([2.0, 2.0, 2.0, 2.0])
    c = CA.prediction_interval_coverage(y, lo, hi)
    assert c["coverage"] == pytest.approx(0.5)
    assert c["mean_width"] == pytest.approx(2.0)


def test_interval_validates_bounds():
    with pytest.raises(ValueError, match="upper < lower"):
        CA.prediction_interval_coverage([1.0], [2.0], [1.0])


def test_sampling_and_predictive_intervals_are_distinguished():
    """两类区间的含义不同，混用会让"不确定性"这句话失去信息量。"""
    s = CA.describe_interval_kind("sampling")
    p = CA.describe_interval_kind("predictive")
    assert s != p
    assert "比较方法优劣" in s
    assert "临床提示" in p
    with pytest.raises(ValueError, match="kind 必须是"):
        CA.describe_interval_kind("whatever")


def test_temperature_scaling_is_declared_but_unused():
    """特征化：``--temp_scale`` 声明了却从未被使用。

    论文称"验证时应用温度缩放"，而全库检索只命中参数声明处。
    与 #17 的死配置同类。一旦接入，本用例应当失败并被改写。
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    hits = []
    for path in list(root.glob("*.py")) + list((root / "utils").glob("*.py")) \
            + list((root / "models").glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "temp_scale" in text:
            hits.append(path.name)
    assert set(hits) <= {"main.py", "tests"}, f"temp_scale 出现在 {sorted(set(hits))}"
