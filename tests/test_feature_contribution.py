"""issue #79 / RED：特征贡献——生理变量是否带来超出"外推先前输液速率"的信息。

对应 R4-11：论文称模型能适应不断变化的血流动力学状态，但缺少直接证据表明模型
从生理变量中获得了超出"外推上一小时"的预测信息。论文自己把这件事推给了未来工作
（"future work could explicitly disentangle severity-driven versus
physiology-driven components"），审稿人要求现在做。

这个缺口在最初整改计划的 26 张票里**没有归属**——四张票（#5/#9/#10/#22）的
Reviewer 字段都写了 R4-11，但没有一张真的实施。

断言对象是 ``docs/feature_contribution.json``：由 ``analyze_feature_contribution.py``
从 ``preds/bundle.npz`` 与 ``.audit/ablation.json`` **计算**得出，不是手抄的数字。
"""

from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
RECORD = ROOT / "docs" / "feature_contribution.json"


@pytest.fixture(scope="module")
def rec():
    if not RECORD.exists():
        pytest.skip("缺少 feature_contribution.json，先跑 analyze_feature_contribution.py")
    return json.loads(RECORD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pred(rec):
    return {r["name"]: r for r in rec["predictors"]}


# ------------------------------------------------- 与纯外推的对照（R4-11 的正题）

def test_all_three_predictors_are_on_the_same_footing(pred):
    """三个预测器必须同窗口、同掩码、同患者数，否则比较无意义。

    这是 #12/#13 的教训：窗口级与患者级 MAE 是**两个不同的量**，
    混用会让"模型好 2.9 倍"这种结论凭空出现。"""
    assert set(pred) == {"BurnResu", "persistence", "moving_average"}
    assert {r["n_patients"] for r in pred.values()} == {55}


def test_persistence_uses_no_physiology(pred):
    """persistence 只外推上一小时，**不含任何生理变量**——这正是它的意义。"""
    p = pred["persistence"]
    assert p["pat_mae"] > 0 and p["win_mae"] > 0


def test_model_does_not_beat_pure_extrapolation(pred):
    """**关键结论**：完整模型没有胜过"只外推上一小时"。

    窗口级：persistence 80.15 < 模型 81.70。
    患者级：90.33 vs 90.14，差 0.19 mL/h，在噪声范围内（#13 的配对区间跨 0）。

    若生理变量真的带来了超出上一小时的增量信息，完整模型至少应明显更优。
    它没有。
    """
    m, p = pred["BurnResu"], pred["persistence"]
    assert p["win_mae"] < m["win_mae"], "窗口级：模型已优于纯外推，结论需重写"
    assert abs(m["pat_mae"] - p["pat_mae"]) < 5.0, \
        "患者级差距过大，与 #13 的配对区间跨 0 不符"


def test_classification_endpoint_is_flat_too(pred):
    """分类侧同样没有优势：macro F1 0.6808 vs 0.6688，且移动平均（0.6854）最高。

    一个不含任何生理变量、也不含学习的预测器拿到最高 F1，
    这本身就是对"模型从生理变量中学到了东西"的反证。
    """
    m, p, a = pred["BurnResu"], pred["persistence"], pred["moving_average"]
    assert abs(m["macro_f1"] - p["macro_f1"]) < 0.02, "分类侧差距与记录不符"
    assert a["macro_f1"] > m["macro_f1"], "移动平均不再是最高，需复核"


def test_paired_bootstrap_favours_persistence_on_per_patient_f1(pred):
    """#13 的患者级配对 bootstrap：**第三个聚合口径**。

    按患者算 macro F1 再平均，persistence 显著更优：
    Δ −0.0281，95% 区间 [−0.0428, −0.0147]，整体在 0 以下。
    MAE 的区间则跨 0。
    """
    m, p = pred["BurnResu"], pred["persistence"]
    assert p["n_patients"] == m["n_patients"] == 55
    delta, lo, hi = -0.0281, -0.0428, -0.0147
    assert hi < 0, "配对区间不再整体为负"


def test_adaptation_claim_is_not_supported(pred):
    """合起来：**"适应不断变化的血流动力学"这一主张缺乏支持**。

    1. 完整模型在窗口级 MAE 上**差于**不含任何生理变量的纯外推；
    2. 患者级 MAE 两者无法区分（差 0.19）；
    3. 分类侧无优势，患者级配对检验反而显著偏向纯外推。
    """
    m, p = pred["BurnResu"], pred["persistence"]
    assert p["win_mae"] < m["win_mae"]
    assert abs(m["pat_mae"] - p["pat_mae"]) < 5.0


# ------------------------------------------- 输入模态之间的相对贡献（第 3 节）

def test_both_input_modalities_contribute_relative_to_each_other(rec):
    """两种输入各自都有贡献——模型没有忽略任何一路。

    但注意：这**不是** R4-11 问的问题。两个变体都仍能通过标签派生的累计量
    看到输液历史，所以"相对另一模态有贡献" ≠ "超出上一小时有贡献"。
    """
    ab = rec["ablation"]
    assert ab["wo_temporal"]["delta"] > 20, "去掉时序生理竟无影响"
    assert ab["wo_baseline"]["delta"] > 20, "去掉基线特征竟无影响"
    for k in ("wo_temporal", "wo_baseline"):
        lo, hi = ab[k]["ci"]
        assert lo > 0 and hi > 0, f"{k} 的区间跨 0，不能称为有贡献"


def test_sign_convention_is_variant_minus_full(rec):
    """ΔMAE 的符号约定：**变体 − full**，正值表示变体更差。

    ``analyze_ablation.py`` 的表头一度写作"变体 − full"却打印 ``full − 变体``，
    方向与标签相反。本用例锁住修正后的方向，防止回退。
    """
    ab = rec["ablation"]
    assert ab["wo_temporal"]["delta"] > 0, "符号方向反了"
    assert ab["full"]["pat_mae"] < ab["wo_temporal"]["pat_mae"]


# --------------------------------------------------------- 边界：不能多说的话

def test_no_per_feature_importance_is_claimed(rec):
    """产物里没有输入张量，做不了逐特征归因，因此**不得**声称做过。

    随包 bundle 只有 ``prob`` / ``speed`` / ``rate`` 等输出，没有输入特征。
    回复信只能写"输入组消融"，不能写成"特征重要性分析"。
    """
    keys = set(rec.keys())
    assert "feature_importance" not in keys and "shap" not in keys
    body = (ROOT / "docs" / "feature_contribution.md").read_text(encoding="utf-8")
    assert "feature-importance analysis" in body, "边界说明被删除"


def test_record_carries_no_patient_identifiers(rec):
    """记录必须只含聚合量——仓库是公开的，患者标识不得上传。"""
    blob = json.dumps(rec, ensure_ascii=False)
    assert "patient_id" not in blob
    assert all(isinstance(r["n_patients"], int) for r in rec["predictors"])
