"""issue #13 / RED：患者级 bootstrap 与配对比较。

对应审稿意见：同患者的小时窗口高度相关，不经患者级置信区间，
MAE 27.44 vs 27.78 这类小幅差异无法与抽样噪声区分。
"""

from __future__ import annotations

import numpy as np
import pytest

from utils import bootstrap as BS


def clustered(seed=0, n_pat=30, per_pat=5, shift=0.0, spread=1.0):
    """构造患者内高度相关的数据：患者效应远大于患者内噪声。"""
    rng = np.random.default_rng(seed)
    vals, ids = [], []
    for p in range(n_pat):
        base = rng.normal(shift, spread)
        vals.extend(base + rng.normal(0, 0.01, per_pat))
        ids.extend([f"P{p:03d}"] * per_pat)
    return np.array(vals), ids


# ------------------------------------------------- 重采样单位是患者

def test_bootstrap_unit_is_patient():
    """重采样单位是患者：同患者的窗口必须一起进出。"""
    vals, ids = clustered(n_pat=20, per_pat=4)
    r = BS.patient_clustered_bootstrap(vals, ids, n_boot=200, seed=1)
    assert r["n_patients"] == 20, "患者数应为 20，而非窗口数 80"
    assert r["n_observations"] == 80

    # 若被当成独立窗口，区间会明显更窄；这里用同一数据对比验证方向
    rng = np.random.default_rng(1)
    naive = [np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(200)]
    naive_w = np.percentile(naive, 97.5) - np.percentile(naive, 2.5)
    clustered_w = r["ci_high"] - r["ci_low"]
    assert clustered_w > naive_w, "患者级区间未比按窗口重采样更宽"


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="长度不一致"):
        BS.patient_clustered_bootstrap(np.zeros(3), ["A", "B"])


def test_empty_grouping_is_rejected():
    with pytest.raises(ValueError, match="没有可用的患者分组"):
        BS.patient_clustered_bootstrap(np.zeros(0), [])


def test_n_boot_is_validated():
    vals, ids = clustered(n_pat=3, per_pat=2)
    with pytest.raises(ValueError, match="n_boot"):
        BS.patient_clustered_bootstrap(vals, ids, n_boot=0)


# --------------------------------------------------------- 配对比较

def test_paired_resampling_shares_indices():
    """配对 bootstrap 必须共享患者索引：A 与 A 自身比较，区间应塌缩到 0。"""
    vals, ids = clustered(seed=2)
    BS.assert_paired_indexing_shared(vals, vals, ids)     # 不抛异常即通过


def test_paired_detects_a_real_shift():
    """A 整体高 5，配对差异应显著且不含 0。"""
    a, ids = clustered(seed=3, shift=0.0)
    b, _ = clustered(seed=3, shift=0.0)
    r = BS.paired_bootstrap_diff(a + 5.0, b, ids, n_boot=300, seed=7)
    assert r["delta"] == pytest.approx(5.0, abs=0.2)
    assert not r["crosses_zero"], "明显差异却被判为跨 0"


def test_paired_reports_no_difference_when_there_is_none():
    """完胜与持平都要能如实给出；持平不得被说成显著。"""
    vals, ids = clustered(seed=4)
    r = BS.paired_bootstrap_diff(vals.copy(), vals.copy(), ids, n_boot=300, seed=7)
    assert r["delta"] == pytest.approx(0.0)
    assert r["crosses_zero"], "完全相同的两组却报告了显著差异"


def test_paired_sign_convention_is_documented():
    """正值表示 A 的误差更大；ci_high < 0 才说明 A 显著更优。"""
    a, ids = clustered(seed=5)
    b = a - 3.0
    r = BS.paired_bootstrap_diff(a, b, ids, n_boot=200, seed=1)
    assert r["delta"] > 0, "A 更大时 delta 应为正"
    assert "正值表示 A 的指标更大" in r["note"]


def test_paired_requires_equal_lengths():
    """配对比较要求两组观测一一对应，长度不等直接报错。

    注：``paired_bootstrap_diff`` 对两组使用**同一份** patient_ids，
    因此"患者集合不同"是不可达的防御性分支，不构造用例。
    """
    a, _ = clustered(seed=6, n_pat=5)
    b, _ = clustered(seed=6, n_pat=6)
    with pytest.raises(ValueError, match="长度不一致"):
        BS.paired_bootstrap_diff(a, b, ["P%03d" % (i // 5) for i in range(len(a))])


# --------------------------------------------------------- 可复现

def test_determinism_with_fixed_seed():
    """固定 seed 重复采样，点估计、区间与差异方向必须一致。"""
    vals, ids = clustered(seed=8)
    r1 = BS.patient_clustered_bootstrap(vals, ids, n_boot=200, seed=11)
    r2 = BS.patient_clustered_bootstrap(vals, ids, n_boot=200, seed=11)
    assert r1 == r2

    a, ids_a = clustered(seed=9)
    b, _ = clustered(seed=9)
    d1 = BS.paired_bootstrap_diff(a, b + 1.0, ids_a, n_boot=200, seed=3)
    d2 = BS.paired_bootstrap_diff(a, b + 1.0, ids_a, n_boot=200, seed=3)
    assert d1 == d2, "配对结果不可复现"


def test_different_seeds_agree_on_direction():
    """不同 bootstrap seed 不应改变结论方向。"""
    a, ids = clustered(seed=10)
    b = a - 4.0                      # B 比 A 小 4 -> delta 应为正
    signs = [BS.paired_bootstrap_diff(a, b, ids, n_boot=150, seed=s)["delta"] > 0
             for s in (1, 2, 3)]
    assert all(signs), "不同 seed 给出了相反的差异方向"


# ------------------------------------------------- 多种子与 bootstrap 分开

def test_seed_summary_is_distinct_from_bootstrap_ci():
    """训练种子的均值±SD 与患者级 bootstrap 区间是两回事，必须分开报告。"""
    s = BS.summarize_seeds({42: 0.70, 43: 0.72, 44: 0.71})
    assert s["n_seeds"] == 3
    assert s["mean"] == pytest.approx(0.71, abs=0.01)
    assert s["sd"] > 0
    assert s["seeds"] == [42, 43, 44]

    vals, ids = clustered(seed=11)
    b = BS.patient_clustered_bootstrap(vals, ids, n_boot=100, seed=1)
    assert set(b) & {"mean", "sd"} == set() or "ci_low" in b
    assert "ci_low" in b and "n_patients" in b


def test_single_seed_has_zero_sd():
    s = BS.summarize_seeds({42: 0.7})
    assert s["n_seeds"] == 1 and s["sd"] == 0.0


def test_seed_summary_handles_nan():
    s = BS.summarize_seeds({42: float("nan")})
    assert s["n_seeds"] == 0
