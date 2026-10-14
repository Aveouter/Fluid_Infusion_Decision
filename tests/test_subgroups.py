"""issue #22 / RED：患者级误差分布、失败病例与临床亚组。

对应审稿意见：需给出逐患者误差分布并包含失败病例；代表性病例的选择标准需
说明；按性别、年龄、TBSA、吸入性损伤等分层报告性能；样本量小时不得作强结论。
"""

from __future__ import annotations

import numpy as np
import pytest

from utils import subgroups as SG


def base(sex=1.0, age=45.0, weight=60.0, tbsa=45.0, inhale=0.0):
    return [sex, age, 170.0, weight, 22.0, tbsa, 0.0, 0.0, 0.0, inhale]


# --------------------------------------------------------- 亚组划分

def test_subgroup_labels_cover_all_dimensions():
    lab = SG.subgroup_labels(base())
    assert set(lab) == {"sex", "age", "tbsa", "inhalation"}
    assert lab["sex"] == "male"
    assert lab["age"] == "40-59"
    assert lab["tbsa"] == "30-49"
    assert lab["inhalation"] == "injury_0"


@pytest.mark.parametrize("age,expect", [
    (10.0, "<18"), (18.0, "18-39"), (39.9, "18-39"),
    (40.0, "40-59"), (59.9, "40-59"), (60.0, "60+"), (90.0, "60+"),
])
def test_age_bands_are_half_open(age, expect):
    """年龄分层为左闭右开，边界不重叠、不留缝。"""
    assert SG.subgroup_labels(base(age=age))["age"] == expect


@pytest.mark.parametrize("tbsa,expect", [(10.0, "<30"), (30.0, "30-49"),
                                         (49.9, "30-49"), (50.0, "50+"), (99.0, "50+")])
def test_tbsa_bands(tbsa, expect):
    assert SG.subgroup_labels(base(tbsa=tbsa))["tbsa"] == expect


def test_missing_values_go_to_unknown_not_guessed():
    """缺失值归入 unknown，**不填补、不猜测**。

    否则亚组结论会建立在编造的分组上。
    """
    lab = SG.subgroup_labels([np.nan, np.nan, np.nan, np.nan, np.nan,
                              np.nan, np.nan, np.nan, np.nan, np.nan])
    assert lab["sex"] == "unknown"
    assert lab["age"] == "unknown"
    assert lab["tbsa"] == "unknown"
    assert lab["inhalation"] == "unknown"


def test_short_baseline_row_is_rejected():
    with pytest.raises(ValueError, match="至少需要"):
        SG.subgroup_labels([1.0, 45.0])


# ------------------------------------------- 先按患者聚合再分组统计

def test_group_metrics_aggregates_per_patient_first():
    """同一患者多个窗口必须先合并，患者等权。"""
    vals = np.array([10.0, 10.0, 10.0, 0.0, 0.0])   # 患者 A 三窗、B 两窗
    pids = ["A", "A", "A", "B", "B"]
    labs = ["x"] * 5
    g = SG.group_metrics(vals, pids, labs)["x"]
    assert g["n_patients"] == 2
    assert g["mean"] == pytest.approx((10.0 + 0.0) / 2), "未按患者等权"


def test_group_metrics_flags_small_groups():
    """样本过小的亚组仍报告，但必须标注不稳健。"""
    vals = np.arange(6.0)
    pids = [f"P{i}" for i in range(6)]
    small = SG.group_metrics(vals, pids, ["rare"] * 6)["rare"]
    assert small["n_patients"] == 6
    assert small["unstable"] is True

    big_vals = np.arange(40.0)
    big_pids = [f"Q{i}" for i in range(40)]
    big = SG.group_metrics(big_vals, big_pids, ["common"] * 40)["common"]
    assert big["unstable"] is False


def test_group_metrics_rejects_length_mismatch():
    with pytest.raises(ValueError, match="长度必须一致"):
        SG.group_metrics(np.zeros(3), ["A", "B"], ["x", "x", "x"])


def test_group_metrics_ignores_nan():
    vals = np.array([1.0, np.nan, 3.0])
    g = SG.group_metrics(vals, ["A", "A", "B"], ["x"] * 3)["x"]
    assert g["n_patients"] == 2


# --------------------------------------------------------- 选例规则

def test_case_selection_is_deterministic_and_includes_failures():
    """选例规则必须确定性，且**必须包含最差病例**——只展示成功案例是挑样。"""
    vals = np.array([1.0, 5.0, 3.0, 9.0, 7.0])
    pids = ["A", "B", "C", "D", "E"]
    c = SG.select_cases(vals, pids, n_each=2)
    # worst 按误差**降序**排列：D(9) 最差、E(7) 次之；best 按升序：A(1)、C(3)
    assert [p for p, _ in c["worst"]] == ["D", "E"]
    assert [p for p, _ in c["best"]] == ["A", "C"]
    # 重复调用结果一致
    assert SG.select_cases(vals, pids, n_each=2) == c


def test_case_selection_has_no_manual_picking():
    """规则只依赖数值排序与患者编号，不接受任何"指定病例"的入口。"""
    import inspect
    sig = inspect.signature(SG.select_cases)
    assert set(sig.parameters) == {"values", "patient_ids", "n_each"}


def test_case_selection_handles_empty_input():
    c = SG.select_cases(np.array([]), [], n_each=2)
    assert c == {"median": [], "best": [], "worst": []}


def test_case_selection_validates_n_each():
    with pytest.raises(ValueError, match="n_each"):
        SG.select_cases(np.array([1.0]), ["A"], n_each=0)


# ----------------------------------------------------- 逐患者误差分布

def test_per_patient_errors_reports_distribution():
    pred = np.array([[30.0], [10.0], [0.0]])
    true = np.array([[10.0], [10.0], [50.0]])
    pids = ["A", "A", "B"]
    s = SG.per_patient_errors(pred, true, pids)
    # 患者 A 两个窗口误差 20 与 0 -> 均值 10；患者 B 误差 50
    assert s["n_patients"] == 2
    assert s["patient_mean"] == pytest.approx((10.0 + 50.0) / 2)
    assert s["patient_max"] == pytest.approx(50.0)
    assert s["unit"] == "positive_flow_ml_per_h"


def test_per_patient_errors_skips_zero_flow():
    """零流单元不计入误差（与 #9 的对照口径一致）。"""
    pred = np.array([[999.0], [10.0]])
    true = np.array([[0.0], [10.0]])
    s = SG.per_patient_errors(pred, true, ["A", "A"])
    assert s["patient_mean"] == pytest.approx(0.0)


def test_per_patient_errors_shape_is_validated():
    with pytest.raises(ValueError, match="形状不一致"):
        SG.per_patient_errors(np.zeros((3, 1)), np.zeros((3, 2)), ["A"] * 3)


# --------------------------------------------------------- 失败病例

def test_failure_patients_are_ranked_by_severity():
    pred = np.array([[100.0], [10.0], [50.0]])
    true = np.array([[10.0], [10.0], [10.0]])
    pids = ["A", "B", "C"]
    rows = SG.failure_patients(pred, true, pids, threshold=20.0)
    # A 误差 90（最严重）、C 误差 40；按误差降序
    assert [r["patient_id"] for r in rows] == ["A", "C"]
    assert rows[0]["mean_abs_error"] == pytest.approx(90.0)
    assert rows[1]["mean_abs_error"] == pytest.approx(40.0)

    # 全部低于阈值时返回空列表，而不是硬凑几个"失败"
    assert SG.failure_patients(true, true, pids, threshold=20.0) == []


def test_failure_patients_reports_cell_counts():
    pred = np.array([[100.0], [100.0]])
    true = np.array([[10.0], [10.0]])
    rows = SG.failure_patients(pred, true, ["A", "A"], threshold=1.0)
    assert rows[0]["n_cells"] == 2
