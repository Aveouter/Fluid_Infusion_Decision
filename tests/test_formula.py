"""issue #25 / RED：公式对照的量纲、误差口径与解读边界。

对应审稿意见：公式估计的是每公斤每 %TBSA 的 24 小时总量，而 BurnResu 训练于
临床医生行为；误差更小几乎是同义反复。绝对误差与相对误差混用不明确，
"40% 以内"的结果不显著。
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from utils import formula as FO

ROOT = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------- 量纲换算

def test_formula_unit_conversion():
    """``mL/kg/%TBSA`` → mL 需要体重与 TBSA 两个分母，缺一不可。"""
    # 2 mL/kg/%TBSA，60 kg，45% TBSA -> 2*60*45 = 5400 mL
    assert FO.formula_total_ml(2.0, 60.0, 45.0) == pytest.approx(5400.0)


def test_tbsa_is_a_percentage_not_a_fraction():
    """TBSA 按百分数传入（45 表示 45%），与数据字典一致。

    若误按分数（0.45）传入，结果会小 100 倍——这类单位错误正是
    R3-7 所指的"绝对误差与相对误差混用不明确"的来源之一。
    """
    as_percent = FO.formula_total_ml(2.0, 60.0, 45.0)
    as_fraction = FO.formula_total_ml(2.0, 60.0, 0.45)
    assert as_percent / as_fraction == pytest.approx(100.0)


def test_formula_rejects_bad_denominators():
    with pytest.raises(ValueError, match="体重必须为正"):
        FO.formula_total_ml(2.0, 0.0, 45.0)
    with pytest.raises(ValueError, match="TBSA 不得为负"):
        FO.formula_total_ml(2.0, 60.0, -1.0)


@pytest.mark.parametrize("unit", FO.REPORT_UNITS)
def test_report_units_are_consistent(unit):
    """同一体积在不同报告单位下的换算必须自洽。"""
    v = FO.formula_total_ml(2.0, 60.0, 45.0)
    got = FO.to_report_unit(v, unit, 60.0, 45.0)
    if unit == "ml":
        assert got == pytest.approx(v)
    elif unit == "ml_per_kg":
        assert got == pytest.approx(v / 60.0)
    else:
        assert got == pytest.approx(v / (60.0 * 45.0))


def test_report_unit_rejects_zero_denominator():
    with pytest.raises(ValueError, match="正的体重"):
        FO.to_report_unit(100.0, "ml_per_kg", 0.0, 45.0)
    with pytest.raises(ValueError, match="正的 TBSA"):
        FO.to_report_unit(100.0, "ml_per_kg_tbsa", 60.0, 0.0)


def test_unknown_report_unit_is_rejected():
    with pytest.raises(ValueError, match="unit 必须是"):
        FO.to_report_unit(100.0, "ml_per_h", 60.0, 45.0)


# --------------------------------------------------------- 误差口径

def test_absolute_and_relative_error_are_distinct():
    e = FO.absolute_and_relative_error(true_ml=1000.0, pred_ml=1400.0)
    assert e["ae_ml"] == pytest.approx(400.0)
    assert e["re"] == pytest.approx(0.40)


def test_zero_denominator_is_not_silently_zero():
    """真实总量为 0 时相对误差无定义，必须标 NaN 而不是记成 0。

    记成 0 会把"完全没预测对"写成"完美命中"。
    """
    e = FO.absolute_and_relative_error(true_ml=0.0, pred_ml=500.0)
    assert e["ae_ml"] == pytest.approx(500.0)
    assert np.isnan(e["re"])
    assert e["re_defined"] == 0.0


def test_summarize_reports_undefined_count():
    errs = [FO.absolute_and_relative_error(1000.0, 1200.0),
            FO.absolute_and_relative_error(0.0, 300.0),
            FO.absolute_and_relative_error(1000.0, 800.0)]
    s = FO.summarize_errors(errs)
    assert s["n"] == 3
    assert s["n_re_defined"] == 2
    assert s["n_re_undefined"] == 1
    assert s["ae_mean_ml"] == pytest.approx((200 + 300 + 200) / 3)
    assert s["re_mean"] == pytest.approx((0.2 + 0.2) / 2)


def test_within_threshold_reports_denominator():
    errs = [FO.absolute_and_relative_error(1000.0, 1300.0),   # re 0.30 ✓
            FO.absolute_and_relative_error(1000.0, 1700.0),   # re 0.70 ✗
            FO.absolute_and_relative_error(0.0, 100.0)]       # 未定义
    w = FO.within_threshold(errs, 0.40)
    assert w["n"] == 2
    assert w["n_within"] == 1
    assert w["fraction"] == pytest.approx(0.5)
    assert w["n_undefined"] == 1, "未定义样本未单独报出"


# ------------------------------------- 论文"40% 以内"的说法能否成立

def test_manuscript_40_percent_claim_holds_only_for_day1():
    """特征化：随包记录显示「40% 以内」只在 Day1 的晶体/胶体成立。

    ``ckpts/test_metrics.json`` 的 ``ae_re_tables.by_day_type``：

        Day1 : crystal RE 0.358, colloid 0.404, water 1.161
        Day2 : crystal 0.281, colloid 0.286, water 0.932
        Total: crystal **0.556**, colloid 0.395, water **1.104**

    Total 行的晶体是 55.6%、水是 110.4%，都远超 40%。因此"40% 以内"
    必须写明是**哪个时段、哪一类**，否则是选择性陈述。
    """
    p = ROOT / "ckpts" / "test_metrics.json"
    if not p.exists():
        pytest.skip("缺少 test_metrics.json")
    tables = json.load(open(p, encoding="utf-8"))["overall"]["ae_re_tables"]
    by_day = tables["by_day_type"]

    day1 = by_day["Day1"]
    assert day1["crystal"]["RE_ml_kg_mean"] <= 0.40
    assert day1["colloid"]["RE_ml_kg_mean"] <= 0.41

    total = by_day["Total"]
    assert total["crystal"]["RE_ml_kg_mean"] > 0.40, \
        "Total 晶体已 <= 40%，前提变化"
    assert total["water"]["RE_ml_kg_mean"] > 1.0, \
        "Total 水相的相对误差超过 100%"


def test_ml_comparison_tables_are_empty():
    """特征化：公式对照里的 ML 表**全是 null** —— 该对照并未真正完成。"""
    p = ROOT / "ckpts" / "test_metrics.json"
    if not p.exists():
        pytest.skip("缺少 test_metrics.json")
    tables = json.load(open(p, encoding="utf-8"))["overall"]["ae_re_tables"]
    assert tables["ml"] is None
    assert tables["ml_per_kg"] is None
    assert tables["ml_per_kg_tsba"] is None


def test_ae_and_re_are_reported_with_units():
    """汇总结果必须自带单位说明，避免 AE/RE 混用。"""
    s = FO.summarize_errors([FO.absolute_and_relative_error(100.0, 120.0)])
    assert "mL" in s["unit"]
    assert "re" in s["unit"]


# --------------------------------------------------------- 解读边界

def test_imitation_note_is_explicit():
    """对照的解读边界必须可被逐字引用（R4-7 / R4-8）。"""
    note = FO.imitation_not_efficacy_note()
    assert "重现" in note or "模拟" in note
    assert "不是临床疗效" in note
    assert "结局证据" in note


def test_manuscript_section_title_claims_efficacy():
    """特征化：论文小节标题仍作「Clinical Effectiveness」。

    与实际证据（只比较与医生行为的接近程度）不符，应由 #27 改写。
    """
    tex = ROOT / "BurnResu .tex"
    if not tex.exists():
        pytest.skip("本地无稿件")
    text = tex.read_text(encoding="utf-8", errors="replace")
    assert "Clinical Effectiveness" in text, "标题已改，请更新 #27 的清单"
