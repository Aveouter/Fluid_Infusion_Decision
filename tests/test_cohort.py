"""issue #24 / RED：入组、零时点与选择偏倚（R4-14）。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from utils import cohort as CO


def test_ledger_conserves_headcount():
    """人数必须守恒：候选 = 排除 + 最终。"""
    cands = [f"P{i:02d}" for i in range(20)]
    excl = {"P00": "non_icu_admission", "P01": "incomplete_records"}
    led = CO.cohort_ledger(cands, excl)
    assert led["n_candidates"] == 20
    assert led["n_excluded"] == 2
    assert led["n_final"] == 18
    assert led["by_reason"] == {"non_icu_admission": 1, "incomplete_records": 1}


def test_unknown_exclusion_reason_is_rejected():
    """不得让「其它」成为无法审计的黑洞。"""
    with pytest.raises(ValueError, match="未知排除原因"):
        CO.cohort_ledger(["P1"], {"P1": "misc"})


def test_excluded_must_be_a_candidate():
    with pytest.raises(ValueError, match="不在候选名单"):
        CO.cohort_ledger(["P1"], {"P9": "incomplete_records"})


def test_duplicate_candidates_are_rejected():
    with pytest.raises(ValueError, match="重复编号"):
        CO.cohort_ledger(["P1", "P1"], {})


def test_no_exclusion_keeps_everyone():
    led = CO.cohort_ledger(["P1", "P2"], {})
    assert led["n_final"] == 2 and led["by_reason"] == {}


# --------------------------------------------------------- 零时点

def test_zero_point_is_injury_time():
    t0 = datetime(2021, 5, 4, 10, 0)
    o = CO.time_zero_offsets(t0, admission=t0 + timedelta(hours=3))
    assert o["zero_point"] == "injury_time"
    assert o["admission_hour"] == pytest.approx(3.0)
    assert o["admission_in_window"] is True


def test_prehospital_period_is_flagged_not_ignored():
    """院前已给液时，第 0 小时的累计量**不为零** —— 必须显式标记。"""
    t0 = datetime(2021, 5, 4, 10, 0)
    o = CO.time_zero_offsets(t0, admission=t0 + timedelta(hours=2),
                             prehospital_fluids=True)
    assert o["prehospital_fluids"] is True
    assert o["cumulative_starts_at_zero"] is False
    assert "不得默认按零起算" in o["note"]

    clean = CO.time_zero_offsets(t0, admission=t0 + timedelta(hours=2))
    assert clean["cumulative_starts_at_zero"] is True


def test_admission_after_horizon_is_out_of_window():
    t0 = datetime(2021, 5, 4, 10, 0)
    o = CO.time_zero_offsets(t0, admission=t0 + timedelta(hours=100))
    assert o["admission_in_window"] is False


def test_admission_before_injury_is_rejected():
    t0 = datetime(2021, 5, 4, 10, 0)
    with pytest.raises(ValueError, match="零时点定义需要复核"):
        CO.time_zero_offsets(t0, admission=t0 - timedelta(hours=1))


# --------------------------------------------------------- 纳排判定

def test_paediatric_cutoff_is_deterministic():
    assert CO.eligibility(18.0)["eligible"] is True
    assert CO.eligibility(17.9)["eligible"] is False
    assert CO.eligibility(17.9)["reason"] == "paediatric"


def test_missing_age_is_excluded_not_guessed():
    """年龄缺失按排除处理 —— 否则纳排本身带入了假设。"""
    r = CO.eligibility(float("nan"))
    assert r["eligible"] is False
    assert "缺失" in r["note"]


def test_missingness_threshold_boundary():
    assert CO.eligibility(40.0, n_missing_baseline=10)["eligible"] is True
    r = CO.eligibility(40.0, n_missing_baseline=11)
    assert r["eligible"] is False
    assert r["reason"] == "incomplete_records"


# ------------------------------------------------- 与论文的对账

def test_manuscript_cohort_size_does_not_match_the_data():
    """特征化：论文写 364，pkl 是 363，原始记录目录有 449 个患者文件。

    **86 位患者在某一环节被排除，原因未记录。** 人数差异必须查清并说明。
    """
    r = CO.reconcile_with_manuscript(CO.PICKLE_COHORT_N)
    assert r["matches_pickle"] is True
    assert r["matches_manuscript"] is False
    assert r["discrepancy"] == 1
    assert CO.RAW_RECORD_N - CO.PICKLE_COHORT_N == 86
    assert "必须查清" in r["note"]


def test_exclusion_reasons_cover_the_documented_gaps():
    """排除原因词表必须覆盖审稿人点名的类别。"""
    for r in ("non_icu_admission", "incomplete_records", "paediatric"):
        assert r in CO.EXCLUSION_REASONS
