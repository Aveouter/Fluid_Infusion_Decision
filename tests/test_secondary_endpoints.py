"""issue #81 / RED：次要终点与事件匹配。

对应审稿意见：正流误差不能冒充全时段误差；变点时点必须一对一匹配、
不得重复命中；E1 与 E3 的误差恒等，不能被当作互为独立的证据。
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from tests import eval_fixtures as F
from utils import changepoints as CP
from utils import endpoints as EP


# --------------------------------------------------------- 全时段 vs 正流

def test_positive_flow_mae_is_not_all_hours_mae():
    """同一份数据：正流 50 mL/h、全时段 11.11 mL/h。

    零流单元会把全时段均值拉低，所以把 positive-flow 的数值标成 all-hours
    会**系统性夸大**误差，反之则掩盖。两者必须分开报。
    """
    trajs = F.zero_flow_hour()
    pos = EP.rate_error_summary(trajs, "speed", mask="positive_flow")
    allh = EP.rate_error_summary(trajs, "speed", mask="all")

    assert pos["mae_ml_per_h"] == pytest.approx(50.0)
    assert allh["mae_ml_per_h"] == pytest.approx(11.1111, abs=1e-3)
    assert pos["mask"] == "positive_flow" and allh["mask"] == "all"
    assert pos["n_targets"] == 2 and allh["n_targets"] == 9
    assert pos["mae_ml_per_h"] != pytest.approx(allh["mae_ml_per_h"])


def test_zero_flow_mask_isolates_the_should_have_stopped_cells():
    """零流掩码单独看"该停没停"。"""
    out = EP.rate_error_summary(F.zero_flow_hour(), "speed", mask="zero_flow")
    assert out["n_targets"] == 7          # 9 个单元里 2 个为正流
    assert out["mae_ml_per_h"] == pytest.approx(0.0)


def test_target_pooled_is_reported_but_flagged_as_auxiliary():
    out = EP.rate_error_summary(F.zero_flow_hour(), "speed", mask="all")
    assert "target_pooled" in out
    assert out["aggregation"] == "patient_mean"
    assert "仅作辅助" in out["note"]


def test_mask_kind_is_validated():
    with pytest.raises(ValueError, match="mask 必须是"):
        EP.rate_error_summary(F.zero_flow_hour(), "speed", mask="whatever")


# ------------------------------------------------------------------ SMAPE

def test_smape_hand_computed():
    """正流 SMAPE 的手算值。

    真值 100 / 预测 150 → 200·50/250 = **40%**（两个正流单元相同）。
    """
    out = EP.smape_summary(F.zero_flow_hour(), "speed")
    assert out["smape_percent"] == pytest.approx(40.0)
    assert out["n_targets"] == 2
    assert out["n_skipped"] == 0


def test_smape_skips_zero_denominator_and_counts_it():
    """分母为 0 的单元跳过**并计数**——按 0 计入会人为压低误差。"""
    pids, hours = [f"{F.SYNTH}S"], [0]
    from utils.trajectories import PredictionMatrix, assemble_trajectories
    true = np.array([[0.0, 0.0, 0.0]])
    pred = np.array([[0.0, 0.0, 0.0]])
    trajs = assemble_trajectories(
        pids, hours, np.zeros((1, 3)), true,
        predictions={"speed": PredictionMatrix(rates=pred)})
    out = EP.smape_summary(trajs, "speed", mask="all")
    assert out["available"] is False
    assert out["n_skipped"] == 3


# --------------------------------------------------------------- 时间稳定性

def test_total_variation_flags_a_constant_predictor():
    """恒定预测的总变差为 0：它"稳定"，但完全不做调整。

    真值 100 → 200 → 300（TV = 100 + 100 = 200），预测恒为 100（TV = 0）。
    ``tv_ratio = 0`` 时必须与 change-point recall 一起读，
    否则"稳定"会被误读成优点。
    """
    from utils.trajectories import PredictionMatrix, assemble_trajectories
    true = np.array([[100.0, 0, 0], [200.0, 0, 0], [300.0, 0, 0]])
    pred = np.array([[100.0, 0, 0]] * 3)
    trajs = assemble_trajectories(
        [f"{F.SYNTH}T"] * 3, [0, 1, 2], (true > 0).astype(float), true,
        predictions={"speed": PredictionMatrix(rates=pred)})

    out = EP.total_variation_summary(trajs, "speed")
    assert out["tv_true_ml"] == pytest.approx(200.0)
    assert out["tv_pred_ml"] == pytest.approx(0.0)
    assert out["tv_abs_diff_ml"] == pytest.approx(200.0)
    assert out["tv_ratio"] == pytest.approx(0.0)


def test_total_variation_requires_at_least_two_hours():
    # unequal_hours() 的第 2 位患者只有 1 个小时
    out = EP.total_variation_summary(F.unequal_hours()[1:], "speed")
    assert out["available"] is False


# --------------------------------------------------------------- R² 分母

def test_r2_reports_its_denominator():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.0, 2.0, 3.0, 5.0])
    out = __import__("utils.metrics", fromlist=["metrics"]).r2_with_denominator(y, p)
    assert out["denominator"] == "sum_sq_dev_from_true_mean_over_masked_targets"
    assert out["ss_tot"] == pytest.approx(5.0)
    assert out["ss_res"] == pytest.approx(1.0)
    assert out["r2"] == pytest.approx(0.8)


# ============================================================ 变点匹配

def test_max_cardinality_beats_nearest_first_greedy():
    """``{10,11}`` vs ``{9,10}``（±1h）：最大基数 2，最近邻贪心只有 1。

    贪心先配距离 0 的 10→10，剩下的真实变点 11 就无处可配。这正是 v2
    弃用贪心的原因。
    """
    true_cp, pred_cp = F.change_events_max_cardinality()
    m = CP.match_events(true_cp, pred_cp, tolerance=1)
    assert m["n_matched"] == 2
    assert m["max_multiplicity"] == 1

    # 朴素"最近邻优先"贪心：同样的数据只得 1 个 TP
    true_ts = list(np.where(true_cp[:, 0])[0])
    pred_ts = list(np.where(pred_cp[:, 0])[0])
    used, greedy = set(), 0
    for t in sorted(true_ts, key=lambda t: min(abs(t - u) for u in pred_ts)):
        cands = [u for u in pred_ts if u not in used and abs(u - t) <= 1]
        if cands:
            used.add(min(cands, key=lambda u: abs(u - t)))
            greedy += 1
    assert greedy == 1, "该反例应让最近邻贪心漏掉一个变点"
    assert greedy < m["n_matched"], "贪心与最大基数必须不同，否则反例失效"


def test_matching_equals_brute_force_maximum_on_random_instances():
    """线性贪心必须与穷举最大匹配完全一致。

    结论不是靠断言声明的：这里在小规模上穷举所有单射匹配，逐一比对。
    """
    def brute(true_ts, pred_ts, tol):
        for r in range(min(len(true_ts), len(pred_ts)), -1, -1):
            for tsub in itertools.combinations(true_ts, r):
                for psub in itertools.permutations(pred_ts, r):
                    if all(abs(a - b) <= tol for a, b in zip(tsub, psub)):
                        return r
        return 0

    rng = np.random.default_rng(0)
    for _ in range(400):
        T = int(rng.integers(6, 13))
        tol = int(rng.integers(0, 3))
        t = np.zeros((T, 1), bool)
        p = np.zeros((T, 1), bool)
        t[rng.random(T) < 0.35, 0] = True
        p[rng.random(T) < 0.35, 0] = True
        got = CP.match_events(t, p, tolerance=tol)["n_matched"]
        want = brute(list(np.where(t[:, 0])[0]), list(np.where(p[:, 0])[0]), tol)
        assert got == want, f"tol={tol} true={t[:,0]} pred={p[:,0]}"


def test_one_to_one_matching_forbids_duplicate_hits():
    """v1 的最近邻允许重复命中，会把 recall 报成 1.0。

    真实变点 {10, 11}，预测变点只有 {10}：v2 的召回是 1/2，
    而 v1 报 match_rate = 1.0——同一个预测变点"命中"了两个真实变点。
    """
    true_rates, pred_rates = F.change_rate_series_duplicate_hit()
    true_cp = CP.detect_changes(true_rates)
    pred_cp = CP.detect_changes(pred_rates)

    assert true_cp[:, 0].sum() == 2 and pred_cp[:, 0].sum() == 1

    v2 = CP.event_prf(true_cp, pred_cp, tolerance=1)
    assert v2["micro"]["recall"] == pytest.approx(0.5)
    assert v2["micro"]["precision"] == pytest.approx(1.0)

    # v1：注意参数顺序是 (pred, true)
    v1 = CP.change_timing_error(pred_rates, true_rates, tolerance=1)
    assert v1["match_rate"] == pytest.approx(1.0), \
        "该实例应展示 v1 的 match_rate 是上界（重复命中）"
    assert v1["match_rate"] > v2["micro"]["recall"], \
        "v1 的高估必须能被复算出来，否则无法说明为何弃用"


def test_event_prf_macro_ignores_channels_without_events():
    true_cp, pred_cp = F.change_events_exact()
    out = CP.event_prf(true_cp, pred_cp, tolerance=1)
    assert out["per_channel"][0]["n_true"] == 2
    assert out["macro"]["n_channels"] == 1
    assert out["max_multiplicity"] == 1


def test_timing_reports_lag_only_over_matched_pairs():
    true_cp, pred_cp = F.change_events_max_cardinality()
    out = CP.change_timing_v2(true_cp, pred_cp, tolerance=1)
    assert out["n_matched"] == 2
    assert out["n_unmatched_true"] == 0
    # 10→9 是提前 1 小时，11→10 也是提前 1 小时 → 平均滞后 −1
    assert out["mean_lag_hours"] == pytest.approx(-1.0)


def test_false_change_rate_is_reported_alongside_recall():
    """只报召回会偏袒"到处都在变"的预测器；误报率必须配对给出。"""
    true_cp, pred_cp = F.change_events_max_cardinality()
    out = CP.false_change_rate(true_cp, pred_cp, tolerance=1)
    assert out["n_pred_changes"] == 2
    assert out["n_false_changes"] == 0
    assert out["n_opportunities"] == 12        # 14 个单元里 2 个是真实变点
    assert out["false_change_rate"] == pytest.approx(0.0)


# ======================================================= E1 与 E3 恒等

def test_e1_and_e3_terminal_errors_are_identical():
    """``|(V₀+Ŝ) − (V₀+S)| = |Ŝ − S|``——E1 与 E3 的终末累计误差必然相同。

    因此 E3 **不能**作为 E1 误差的独立稳健性证据。E3 只在相对容差分母、
    累计量绝对水平、时域解释三处提供信息。
    """
    fx = F.anchor_identity()
    v0, t_tail, p_tail = fx["v0"], fx["true_tail"], fx["pred_tail"]

    e1_err = abs((v0 + p_tail.sum()) - (v0 + t_tail.sum()))
    e3_err = abs(p_tail.sum() - t_tail.sum())
    assert e1_err == pytest.approx(e3_err)

    # 累加顺序不影响结论
    assert abs((v0 + p_tail.sum()) - (v0 + t_tail.sum())) == \
        abs(p_tail.sum() - t_tail.sum())


def test_e1_and_e3_differ_in_levels_and_relative_denominator():
    """虽然误差相同，两者的**水平**与**相对容差分母**不同——这是 E3 的用处。"""
    fx = F.anchor_identity()
    v0, t_tail, p_tail = fx["v0"], fx["true_tail"], fx["pred_tail"]
    e1_true = v0 + t_tail.sum()
    e3_true = t_tail.sum()
    assert e1_true != pytest.approx(e3_true)
    assert (abs(p_tail.sum() - t_tail.sum()) / e3_true) != \
        pytest.approx(abs((v0 + p_tail.sum()) - e1_true) / e1_true)


# ==================================================== 速率口径必须分叉

def test_rate_definitions_diverge_when_probability_is_low():
    """``prob`` 低而 ``speed`` 高时，各口径必须给出不同的数。

    τ = 0.5 下：``speed`` = 800，``rate`` = 40，``action_rate_uncond`` = 0。
    真值该小时为 0 → 只有动作口径与"医生没输"这件事一致。
    """
    from utils.decoding import decode_action_rate, decode_bits

    speed = np.array([[800.0, 0.0, 0.0]])
    prob = np.array([[0.05, 0.0, 0.0]])
    thresholds = np.array([0.5, 0.5, 0.5])

    assert decode_action_rate(prob, speed, thresholds)[0, 0] == pytest.approx(0.0)
    assert decode_bits(prob, thresholds)[0, 0] == 0
    assert (prob * speed)[0, 0] == pytest.approx(40.0)
    assert speed[0, 0] == pytest.approx(800.0)
    assert len({0.0, 40.0, 800.0}) == 3, "三个口径必须给出不同的值"


def test_decoding_requires_thresholds_from_the_validation_set():
    """阈值必须显式给出；不接受"没有就默认 0.5"。"""
    from utils.decoding import decode_action_rate

    speed = np.array([[100.0, 100.0, 100.0]])
    prob = np.array([[0.6, 0.6, 0.6]])
    with pytest.raises(ValueError, match="thresholds 必填"):
        decode_action_rate(prob, speed, None)
    with pytest.raises(ValueError, match="不得回退 0.5"):
        decode_action_rate(prob, speed, np.array([]))


def test_decoding_rejects_out_of_range_or_mismatched_thresholds():
    from utils.decoding import decode_action_rate
    prob = np.array([[0.5, 0.5, 0.5]])
    speed = np.array([[1.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="阈值"):
        decode_action_rate(prob, speed, np.array([1.5, 0.5, 0.5]))
    with pytest.raises(ValueError, match="长度"):
        decode_action_rate(prob, speed, np.array([0.5, 0.5]))


# ============================================ 真实零值 vs 缺失 vs 不完整

def test_true_zero_and_missing_are_indistinguishable_in_labels():
    """这是**已知限制**的可执行记录，不是可以修的东西。

    哨兵 −1 在 :func:`utils.label_protocol.normalize` 中被置 0，因此"未记录"
    与"真实零输注"在标签矩阵里已经不可区分 → 目标侧缺失率无法从数据算出。
    协议不得假装能算它，只能在合成场景里演示语义。
    """
    zero, missing = F.zero_vs_missing()
    assert np.array_equal(zero, missing), \
        "若两者变得可区分，说明标签协议改了——必须同步修订协议的已知限制"
    assert zero[0, :3].sum() == 0


def test_structurally_incomplete_is_reported_not_filled():
    """结构不完整 → 该时域不可估计（NA + 原因），而不是补 0 或插值。"""
    trajs = F.thirty_hour_patient()
    t = trajs[0]
    c = t.completeness(72)
    assert c["complete"] is False
    assert c["hours_present"] == 30
    assert c["hours_required"] == 72
    assert len(c["missing_hours"]) == 42
