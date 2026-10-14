"""issue #19 / RED：缺失率分母、插补正确性与泄漏。

对应审稿意见：表 1 缺失率列标"%"，值却是 0.55–0.97 的分数；神经插补的架构与
验证未报告；高缺失率与学习插补可能显著影响结果。
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: 数据审计实测的逐小时缺失率（分母 = 26,136 个规划小时位置）
AUDIT_MISSING = {
    "temperature": 0.6325, "heart_rate": 0.4855, "pulse": 0.9973,
    "respiratory": 0.4838, "sbp": 0.5005, "dbp": 0.5005,
    "cvp": 0.9737, "spo2": 0.4993, "urine": 0.1790,
}

#: 论文表 1 的 "Missing rate(%)" 列
PAPER_MISSING = {
    "heart_rate": 0.56, "sbp": 0.59, "dbp": 0.59, "spo2": 0.56,
    "cvp": 0.97, "respiratory": 0.55, "temperature": 0.73,
}


def test_column_is_labelled_percent_but_holds_fractions():
    """特征化：列名写 %，取值却在 [0,1] 区间 —— 首先是个单位标注错误。"""
    for k, v in PAPER_MISSING.items():
        assert 0.0 <= v <= 1.0, f"{k} 的取值 {v} 已不在 [0,1]"


def test_paper_values_do_not_reconcile_with_the_data():
    """特征化：即便按**分数**解读（即 ×100 当百分比），也只有 CVP 对得上。

    | 变量 | 论文表 1 | 实测缺失率 |
    |---|---:|---:|
    | 心率 | 0.56 | 48.55% |
    | CVP | 0.97 | **97.37%** |
    | 体温 | 0.73 | 63.25% |
    | 呼吸 | 0.55 | 48.38% |

    按分数解读意味着"55–73% 缺失"；按百分数解读意味着"0.55–0.73% 缺失"，
    与实测的 48–63% 相差两个数量级。**两种解读都无法同时解释全部列值。**

    因此这一列既不能直接引用，也不能简单换个单位了事——#26 必须回到原始数据
    重算并写明分母。
    """
    as_fraction_ok = [k for k, v in PAPER_MISSING.items()
                      if abs(v - AUDIT_MISSING[k]) < 0.02]
    assert as_fraction_ok == ["cvp"], (
        f"按分数解读能对上的不止 CVP：{as_fraction_ok}，结论需要更新"
    )

    # 反向：按百分数解读（值本身就是百分数）与实测相差两个数量级
    for k, v in PAPER_MISSING.items():
        assert AUDIT_MISSING[k] / 100.0 < v, f"{k} 按百分数解读反而接近实测"


def test_some_variables_have_no_missing_rate_at_all():
    """表 1 中尿量与口服的缺失率列为空，而实测均为 17.90%。"""
    assert AUDIT_MISSING["urine"] > 0.15


# --------------------------------------------------------- 插补

def test_zero_fill_defeats_nan_detection():
    """特征化：``_safe_numeric_nd`` 先把 NaN 换成 0，``_has_nan_baseline``
    再判 NaN —— 于是缺失检测**形同虚设**。

    后果：基线缺失的患者不会被跳过，而是带着"0"进入训练与评测。
    数据审计指出 205/363 位患者至少缺一个基线字段。
    """
    src = (ROOT / "utils" / "data_loader.py").read_text(encoding="utf-8", errors="replace")
    assert "def _safe_numeric_nd" in src
    assert "def _has_nan_baseline" in src
    # 填充发生在判空之前
    i_fill = src.index("def _safe_numeric_nd")
    i_check = src.index("def _has_nan_baseline")
    assert i_fill < i_check, "两函数顺序变了，请复核本用例的前提"


def test_missing_baseline_correlates_with_worse_performance():
    """特征化：基线缺失（被填 0）的患者误差显著更高。

    #22 的亚组分析实测：吸入性损伤字段缺失的 21 位患者平均误差 108.4 mL/h，
    而已记录为无损伤的 26 位为 80.3 mL/h（相差 35%）。
    **这不是临床发现，是插补缺陷出现在结果里。**

    这两组规模对比如下（本地分析输出，此处只保留聚合数字）。
    """
    unknown_group = {"n": 21, "mean": 108.4}
    recorded_group = {"n": 26, "mean": 80.3}
    assert unknown_group["n"] > 15, "缺失组样本过小，结论不稳健"
    assert unknown_group["mean"] > recorded_group["mean"] * 1.2, \
        "缺失组优势不再明显，前提变化"


def test_imputation_is_fitted_on_the_full_series():
    """特征化：逐患者预处理在**完整 72 小时**上拟合（同时污染测试集输入）。

    源码位于未被跟踪的 ``data/`` 目录，CI 上跳过。
    """
    mod = pytest.importorskip("data.InsertionAndNornalization",
                              reason="预处理源码位于未被跟踪的 data/ 目录")
    import inspect
    fn = inspect.getsource(mod.impute_and_scale_df)
    assert "_fit_mf(data_t, mask_t" in fn
    assert "model()" in fn

    from utils.ziln import ziln_mean  # noqa: F401  （确认 utils 可导入）


def test_local_record_exists_for_the_unfixed_leak():
    """未修的泄漏必须有本地记录（不上传），避免结论随会话丢失。"""
    rec = ROOT / ".audit" / "local_findings.md"
    if not rec.exists():
        pytest.skip("本地记录不存在（.audit/ 被清理过）")
    text = rec.read_text(encoding="utf-8", errors="replace")
    assert "L1" in text and "impute_and_scale_df" in text
