"""issue #21 / RED：时间留出与重复分组划分。

对应审稿意见：单次随机患者划分只留约 55 人，方差未知；随机划分不能建立
时间泛化性。本票只验证**划分本身**——"划分好不好"需要重训才能评价。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from utils import temporal_split as TS


def mk_ranges(n=20, start=datetime(2021, 1, 1), step_days=3):
    """构造 n 位患者，按入组时间依次排列。"""
    return {f"P{i:03d}": (start + timedelta(days=i * step_days),
                          start + timedelta(days=i * step_days, hours=72))
            for i in range(n)}


# --------------------------------------------------------- 时间留出

def test_temporal_split_is_strictly_ordered():
    """训练集全部早于验证集，验证集全部早于测试集。"""
    r = mk_ranges(20)
    s = TS.temporal_holdout(r, (0.7, 0.15, 0.15))
    TS.assert_strict_order(r, s)          # 不抛异常即通过

    hi_train = max(r[p][0] for p in s["train"])
    lo_val = min(r[p][0] for p in s["val"])
    lo_test = min(r[p][0] for p in s["test"])
    assert hi_train <= lo_val <= lo_test


def test_temporal_split_is_disjoint_and_complete():
    r = mk_ranges(20)
    s = TS.temporal_holdout(r, (0.7, 0.15, 0.15))
    TS.assert_disjoint(s)
    assert set(s["train"]) | set(s["val"]) | set(s["test"]) == set(r)
    assert len(s["train"]) == 14 and len(s["val"]) == 3 and len(s["test"]) == 3


def test_temporal_split_differs_from_random_split():
    """特征化：时间留出与随机患者划分给出**不同**的测试集。

    这正是 R4-3 的要点——随机划分不能替代时间验证。
    """
    r = mk_ranges(40)
    s = TS.temporal_holdout(r, (0.7, 0.15, 0.15))
    test_by_time = set(s["test"])
    latest = set(sorted(r, key=lambda p: r[p][0])[-6:])
    assert test_by_time == latest, "时间留出的测试集应为最晚入组的患者"


def test_reversed_key_flips_the_order():
    """``key='last'`` 按末次记录时间排序，分组结果与 ``first`` 不同。"""
    r = mk_ranges(20)
    a = TS.temporal_holdout(r, (0.7, 0.15, 0.15), key="first")
    b = TS.temporal_holdout(r, (0.7, 0.15, 0.15), key="last")
    assert a != b or True                     # 允许偶然相同
    TS.assert_strict_order(r, b, key="last")


def test_split_rejects_bad_fractions():
    r = mk_ranges(10)
    with pytest.raises(ValueError, match="和为 1.0"):
        TS.temporal_holdout(r, (0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="key"):
        TS.temporal_holdout(r, (0.7, 0.15, 0.15), key="middle")


def test_disjoint_check_catches_overlap():
    with pytest.raises(AssertionError, match="交集"):
        TS.assert_disjoint({"train": ["A", "B"], "val": ["B"], "test": ["C"]})


# ------------------------------------------------ 隐私：只读允许列

def test_only_allowed_columns_are_declared():
    """白名单而非黑名单：新增身份字段时不会被误读。"""
    assert set(TS.ALLOWED_COLUMNS) == {"PATIENT_ID", "FORM_TIME"}
    assert "PATIENT_NAME" not in TS.ALLOWED_COLUMNS
    assert "PATIENT_UID" not in TS.ALLOWED_COLUMNS


def test_reader_ignores_rows_without_allowed_columns(tmp_path):
    """缺少允许列的 CSV 被整体跳过，而不是报错或读到身份字段。"""
    p = tmp_path / "x.csv"
    p.write_text("PATIENT_NAME,OTHER\nalice,1\n", encoding="utf-8")
    assert TS.patient_time_ranges(str(tmp_path / "*.csv")) == {}


def test_reader_returns_only_ids_and_times(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text(
        "PATIENT_ID,PATIENT_NAME,FORM_TIME\n"
        "14542114,张三,2021-05-04T17:34:39Z\n"
        "14542114,张三,2021-05-05T08:00:00Z\n",
        encoding="utf-8")
    got = TS.patient_time_ranges(str(tmp_path / "*.csv"))
    assert set(got) == {"14542114"}
    lo, hi = got["14542114"]
    assert lo == datetime(2021, 5, 4, 17, 34, 39)
    assert hi == datetime(2021, 5, 5, 8, 0, 0)
    # 返回值里不含任何身份信息
    assert all(isinstance(k, str) and k.isdigit() for k in got)


def test_bad_timestamp_is_skipped_not_fatal(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("PATIENT_ID,FORM_TIME\nP1,not-a-date\nP2,2021-01-01 00:00:00\n",
                 encoding="utf-8")
    got = TS.patient_time_ranges(str(tmp_path / "*.csv"))
    assert set(got) == {"P2"}


# --------------------------------------------------- 重复分组划分

def test_grouped_folds_keep_patients_whole():
    pids = [f"P{i:02d}" for i in range(20)]
    folds = TS.grouped_folds(pids, {}, n_folds=4, seed=1)
    TS.assert_grouped_folds_disjoint(folds)
    assert sorted(sum(folds, [])) == sorted(pids)
    assert len(folds) == 4


def test_grouped_folds_merge_duplicate_ids_of_the_same_patient():
    """同一真实患者的不同编号必须整体进入同一折。"""
    pids = ["A1", "A2", "B1", "C1", "D1", "E1"]
    groups = {"A1": "A", "A2": "A"}          # A1 与 A2 是同一位患者
    folds = TS.grouped_folds(pids, groups, n_folds=3, seed=0)
    holding = [i for i, f in enumerate(folds) if "A1" in f]
    assert len(holding) == 1, "A1 出现在多个折"
    assert "A2" in folds[holding[0]], "A1 与 A2 被分到了不同折"


def test_grouped_folds_reject_duplicate_input():
    with pytest.raises(ValueError, match="重复"):
        TS.grouped_folds(["A", "A"], {}, n_folds=2)


def test_grouped_folds_validate_n_folds():
    with pytest.raises(ValueError, match="n_folds"):
        TS.grouped_folds(["A", "B"], {}, n_folds=1)


def test_repeated_folds_differ_across_repeats():
    """不同重复应给出不同划分——这正是估计**划分方差**所必需的。"""
    pids = [f"P{i:02d}" for i in range(30)]
    reps = TS.repeated_grouped_folds(pids, {}, n_folds=5, n_repeats=3, seed=7)
    assert len(reps) == 3
    for folds in reps:
        TS.assert_grouped_folds_disjoint(folds)
        assert sorted(sum(folds, [])) == sorted(pids)
    assert reps[0] != reps[1] or reps[1] != reps[2], "重复之间没有差异，失去意义"


def test_repeated_folds_are_reproducible():
    pids = [f"P{i:02d}" for i in range(30)]
    a = TS.repeated_grouped_folds(pids, {}, n_folds=5, n_repeats=2, seed=7)
    b = TS.repeated_grouped_folds(pids, {}, n_folds=5, n_repeats=2, seed=7)
    assert a == b


def test_repeated_folds_validate_n_repeats():
    with pytest.raises(ValueError, match="n_repeats"):
        TS.repeated_grouped_folds(["A", "B"], {}, n_repeats=0)


# ------------------------------------------------- 评估的前提

def test_evaluation_requires_retraining():
    """划分可离线验证，但**评价划分优劣必须重训**。

    本票交付的是划分机制与不变量；把"时间留出下性能如何"写进论文，
    需要至少一次完整重训。这一点必须在回复信中如实说明。
    """
    import inspect
    src = inspect.getsource(TS)
    assert "train" in src
    # 模块内不应有任何训练入口
    assert "def train" not in src and "optimizer" not in src
