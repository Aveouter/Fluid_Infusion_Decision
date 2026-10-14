"""issue #81 / RED：评估协议 v2 的冻结常量、单一来源与哈希。

对应审稿意见：同一个指标名有多种算法、阈值散落各处，导致数字无法互认。
本模块把"口径必须唯一"这件事钉死：常量只有一个来源，文档不能与代码漂移，
定义一变哈希就变。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils import changepoints as CP
from utils import endpoints as EP
from utils import eval_protocol as P

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "eval_protocol.md"
RECORD = ROOT / "docs" / "eval_protocol.json"


# --------------------------------------------------------------- 单一来源

def test_tolerances_have_exactly_one_source():
    """容差曾有三组参数（200/250/50），v2 起只剩一处定义。"""
    assert EP.TOL_ABS_ML is P.TOL_ABS_ML
    assert EP.TOL_PCT is P.TOL_PCT
    assert P.TOL_ABS_ML == 200.0 and P.TOL_PCT == 0.10


def test_change_thresholds_have_exactly_one_source():
    assert CP.CHANGE_ABS_ML is P.CHANGE_ABS_ML
    assert CP.CHANGE_REL is P.CHANGE_REL
    assert CP.REL_BASE_ML is P.REL_BASE_ML


def test_metrics_module_re_exports_the_same_objects():
    """utils.metrics 的词表必须与协议模块是**同一个对象**，不是复制品。"""
    from utils import metrics as M
    for name in ("AGGREGATIONS", "AGGREGATION_LEVELS", "UNITS", "MASKS",
                 "FLUID_CLASSES", "EVAL_MODES", "ESTIMANDS", "HORIZON_ORIGINS",
                 "RATE_DEFINITIONS", "QUALITY_TRACKS", "EVIDENCE_TIERS"):
        assert getattr(M, name) is getattr(P, name), f"{name} 被复制而不是 re-export"
    assert M.PROTOCOL_VERSION == P.PROTOCOL_VERSION == "2.0"


# ------------------------------------------------------------------- 哈希

def test_hash_is_stable_and_order_independent():
    assert P.protocol_hash() == P.protocol_hash()
    assert P.PROTOCOL_HASH == P.protocol_hash()
    assert len(P.PROTOCOL_HASH) == 16


def test_hash_changes_when_a_frozen_constant_changes():
    """任何会改变指标含义的改动都必须改变哈希。"""
    payload = P.freeze_payload()
    baseline = P.protocol_hash(payload)

    perturbed = json.loads(json.dumps(payload))
    perturbed["tolerances"]["tol_abs_ml"] = 250.0
    assert P.protocol_hash(perturbed) != baseline

    perturbed = json.loads(json.dumps(payload))
    perturbed["positive_flow_epsilon_ml_per_h"] = 1e-6
    assert P.protocol_hash(perturbed) != baseline

    perturbed = json.loads(json.dumps(payload))
    perturbed["vocabularies"]["estimands"].append("not_a_real_estimand")
    assert P.protocol_hash(perturbed) != baseline


def test_assert_frozen_rejects_a_mismatched_hash():
    P.assert_frozen(P.PROTOCOL_HASH)                 # 自身一致 → 通过
    with pytest.raises(AssertionError, match="协议 hash 不匹配"):
        P.assert_frozen("0000000000000000")


# --------------------------------------------------------------- 时间算术

def test_recorded_hours_minus_one_equals_predicted_hours():
    """72 个记录小时只对应 71 个模型预测小时（首小时是观测锚点）。"""
    assert P.PREDICTED_HOURS_BY_HORIZON == {h: h - 1 for h in P.HORIZONS}
    assert P.HORIZONS == (24, 48, 72)


def test_estimand_vocabulary_covers_all_four_objects():
    """E1/E2/E2b/E3 必须都在词表里，且名字写明"真实初始状态"这类边界。"""
    for name in ("conditional_cumulative_volume",
                 "action_autoregressive_rollout_under_true_initial_state",
                 "observed_physiology_conditioned_rollout",
                 "pure_prediction_window_volume"):
        assert name in P.ESTIMANDS
    assert "fully_closed_loop" not in " ".join(P.ESTIMANDS), \
        "不得出现把真实初始状态的 rollout 说成闭环的名字"


def test_rate_definition_vocabulary_is_complete():
    for name in ("pos_mean", "flow_pred", "speed", "rate",
                 "action_rate_uncond", "action_rate_cond", "deterministic"):
        assert name in P.RATE_DEFINITIONS
    # 依赖概率头的口径必须被显式登记（否则调用方会以为可以直接算）
    assert set(P.PROBABILISTIC_RATE_DEFINITIONS) <= set(P.RATE_DEFINITIONS)
    assert set(P.DECODED_RATE_DEFINITIONS) <= set(P.RATE_DEFINITIONS)


def test_masks_separate_zero_missing_and_incomplete():
    """真实零值 / 缺失 / 结构不完整必须是三个不同的掩码，不得合并。"""
    for name in ("zero_flow", "missing", "incomplete"):
        assert name in P.MASKS


def test_statistical_contract_is_frozen():
    assert P.INDEPENDENCE_UNIT == "patient"
    assert P.N_OUTER_FOLDS >= 2
    assert P.N_SEEDS_MIN >= 3
    assert P.N_BOOTSTRAP >= 1000
    assert 0.0 < P.CI_LEVEL < 1.0


# ------------------------------------------------------ 文档与代码不漂移

def test_protocol_document_contains_the_current_hash():
    assert DOC.exists(), "缺少 docs/eval_protocol.md"
    text = DOC.read_text(encoding="utf-8")
    assert P.PROTOCOL_HASH in text, \
        "协议文档的哈希与代码不一致：先运行 python freeze_eval_protocol.py --write"
    assert P.PROTOCOL_VERSION in text


def test_generated_block_matches_the_code():
    """定界标记之间的内容必须逐字等于由常量渲染的结果。"""
    text = DOC.read_text(encoding="utf-8")
    b, e = P.FROZEN_BLOCK_BEGIN, P.FROZEN_BLOCK_END
    assert b in text and e in text
    inner = text.split(b, 1)[1].split(e, 1)[0].strip("\n")
    assert inner.strip() == P.render_frozen_block().strip()


def test_protocol_record_matches_the_code():
    assert RECORD.exists(), "缺少 docs/eval_protocol.json"
    got = json.loads(RECORD.read_text(encoding="utf-8"))
    assert got["protocol_hash"] == P.PROTOCOL_HASH
    assert got["protocol_version"] == P.PROTOCOL_VERSION
    assert got["frozen"] == P.freeze_payload()


def test_freeze_script_verifies_clean():
    """``freeze_eval_protocol.py`` 的校验模式必须通过（CI 也跑这一条）。"""
    import freeze_eval_protocol as FZ
    assert FZ.main([]) == 0


def test_freeze_script_detects_a_tampered_document(tmp_path):
    """文档被改动后校验必须失败——否则"冻结"只是口号。"""
    import freeze_eval_protocol as FZ

    doc = tmp_path / "eval_protocol.md"
    doc.write_text(DOC.read_text(encoding="utf-8").replace(
        P.PROTOCOL_HASH, "deadbeefdeadbeef"), encoding="utf-8")
    record = tmp_path / "eval_protocol.json"
    record.write_text(RECORD.read_text(encoding="utf-8"), encoding="utf-8")
    assert FZ.main(["--doc", str(doc), "--record", str(record)]) == 1


def test_protocol_document_states_the_two_required_disclaimers():
    """两条必须写明的声明：原稿未预设时域；主口径尚未冻结。"""
    text = DOC.read_text(encoding="utf-8")
    assert "§3.6" in text or "3.6" in text
    assert "未指定" in text or "没有预设" in text, "必须说明原稿未预设累计剂量时域"
    assert "proposed" in text, "必须说明回归主口径为 proposed、本轮未冻结"
    assert "不得追溯" in text or "不追溯" in text
