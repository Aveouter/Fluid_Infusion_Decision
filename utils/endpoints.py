"""预设终点的统一实现（issue #12，对应 R1-3 / R2-2 / R3-3）。

论文 Methods 列出的终点（累积剂量 MAE、±200 mL/±10% 容差准确率、正流 SMAPE、
变点时间、时间稳定性）在 Results 中未完整出现；其中容差终点**从未实现**
（``--tol_abs_ml`` / ``--tol_pct`` 是死配置，见 #6）。本模块把它们补齐，
并强制每个数字携带单位、分母与聚合口径。

速率终点的口径问题
------------------
审计发现同一份权重存在三个不同的"速率"（详见 #9 的更正与 #12 的裁定）：

- ``flow_pred``   ZILN 均值，含 ``(1-p0)`` 因子，单位 mL/h —— **模型对速率的
  直接输出**；实测取值范围 [19.0, 887.8]，从不接近 0。
- ``speed``       ``softplus(flow_pred)``，在 float32 下与 ``flow_pred``
  逐位相同（≥19 时 softplus 是恒等变换），论文报告的 MAE 用的正是它。
- ``rate``        ``prob × speed``，又乘了一次类型概率，**门控了两次**；
  它被用于 ``L_cum`` 与轨迹图。

三者都保留在本模块里，由调用方显式选择并在结果中标注，避免再次出现"表格用
一个口径、正文用另一个口径"的情况。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

import utils.metrics as M
from utils import label_protocol as lp

#: 容差终点的默认参数。论文方法写的是 ±200 mL 与 ±10%；
#: 代码里 ``--tol_abs_ml`` 的默认值 50.0 从未被使用（见 #6）。
#:
#: v2 起**唯一来源**是 :mod:`utils.eval_protocol`——此处只做 re-export，
#: 不再各自定义，避免"三组容差参数"的历史问题重现。
from utils import eval_protocol as EP  # noqa: E402
from utils.eval_protocol import TOL_ABS_ML, TOL_PCT  # noqa: E402  (re-export)

#: 可选的速率口径。
RATE_DEFINITIONS = ("flow_pred", "speed", "rate")


def _as_series(x, name="数组"):
    """接受 ``(T, C)``（单条序列）或 ``(N, T, C)``（一批序列）。

    时间轴一律取倒数第二维，因此两类输入的处理逻辑完全一致。
    """
    a = np.asarray(x, dtype=float)
    if a.ndim not in (2, 3):
        raise ValueError(f"{name} 应为 (T, C) 或 (N, T, C)，实际 {a.shape}")
    return a


# ------------------------------------------------------- 累积剂量

def cumulative_dose(true_rates: np.ndarray, pred_rates: np.ndarray,
                    step_hours: float = 1.0) -> Dict[str, np.ndarray]:
    """逐步累计体积 ``(N, C)``，单位 mL。

    与 ``data_loader._recompute_inst_cum_from_label`` 同口径：
    第 t 步累计 = Σ_{k<=t} rate_k × Δt。
    """
    true = _as_series(true_rates) * float(step_hours)
    pred = _as_series(pred_rates) * float(step_hours)
    # 时间轴是倒数第二维：2D 输入时退化为原来的 axis=0
    return {"true": np.cumsum(true, axis=-2), "pred": np.cumsum(pred, axis=-2)}


def cumulative_dose_mae(true_rates, pred_rates, step_hours: float = 1.0,
                        level: str = "final") -> Dict[str, float]:
    """累积剂量 MAE，单位 **mL**（与逐小时 mL/h 的 MAE 不是同一个量）。

    ``level="final"`` 只比终末累计量；``"all"`` 对每个时间点都算。
    """
    if level not in ("final", "all"):
        raise ValueError(f"level 应为 final/all，实际 {level}")
    cum = cumulative_dose(true_rates, pred_rates, step_hours)
    a, b = cum["true"], cum["pred"]
    if level == "final":
        a, b = a[..., -1:, :], b[..., -1:, :]
    diff = np.abs(a - b)
    return {"mae_ml": float(diff.mean()),
            "n": int(diff.size),
            "unit": "cumulative_ml",
            "level": level}


# ------------------------------------------------------- 容差终点

def tolerance_accuracy(true_rates, pred_rates, step_hours: float = 1.0,
                       tol_abs_ml: float = TOL_ABS_ML,
                       tol_pct: float = TOL_PCT) -> Dict[str, float]:
    """容差准确率：终末累计量同时满足 ±tol_abs_ml **且** ±tol_pct 的样本占比。

    两处易错，均显式处理：

    - **真实总量为 0 时**不使用百分比判据（除零），只判绝对容差；
    - **两个判据都要满足**（论文写的是"±200 mL 和 ±10%"），不是取其一。
    """
    cum = cumulative_dose(true_rates, pred_rates, step_hours)
    true_f, pred_f = cum["true"][..., -1, :], cum["pred"][..., -1, :]
    diff = np.abs(pred_f - true_f)

    abs_ok = diff <= float(tol_abs_ml)
    denom_zero = true_f <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom_zero, np.nan, diff / np.where(true_f == 0, 1.0, true_f))
    pct_ok = np.where(denom_zero, True, rel <= float(tol_pct))

    ok = abs_ok & pct_ok
    return {
        "accuracy": float(ok.mean()) if ok.size else float("nan"),
        "n": int(ok.size),
        "n_abs_ok": int(abs_ok.sum()),
        "n_pct_ok": int(pct_ok.sum()),
        "n_zero_denominator": int(denom_zero.sum()),
        "tol_abs_ml": float(tol_abs_ml),
        "tol_pct": float(tol_pct),
        "denominator": "per (sample, class) final cumulative volume",
    }


# ------------------------------------------------------- 正流 SMAPE

def smape_positive_flow(true_rates, pred_rates,
                        mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """仅在**预先定义**的正流掩码上计算 SMAPE；零流单元单独计数上报。

    SMAPE 定义：``2|p-t| / (|p|+|t|)``，分母为 0 时该单元跳过并计数，
    不静默按 0 处理（按 0 处理会人为压低误差）。
    """
    true = _as_series(true_rates, "true_rates")
    pred = _as_series(pred_rates, "pred_rates")
    if true.shape != pred.shape:
        raise ValueError(f"形状不一致：{true.shape} vs {pred.shape}")
    if mask is None:
        mask = true > 0
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != true.shape:
        raise ValueError("mask 形状与速率不一致")

    t, p = true[mask], pred[mask]
    denom = np.abs(t) + np.abs(p)
    valid = denom > 0
    skipped = int((~valid).sum())
    if valid.sum() == 0:
        return {"smape": float("nan"), "n": 0, "n_skipped": skipped,
                "mask": "positive_flow"}
    val = float(np.mean(2.0 * np.abs(p[valid] - t[valid]) / denom[valid]))
    return {"smape": val, "n": int(valid.sum()), "n_skipped": skipped,
            "mask": "positive_flow"}


# ------------------------------------------------------- 时间稳定性

def temporal_stability(pred_rates, true_rates, step_hours: float = 1.0) -> Dict[str, float]:
    """轨迹稳定性：逐步变化量的总变差，以及预测/真实的摆动比。

    模型若在相邻小时之间抖动，即便逐点误差不大，轨迹也不可用。
    ``oscillation_ratio`` > 1 表示预测比真实更抖。
    """
    true = _as_series(true_rates, "true_rates") * float(step_hours)
    pred = _as_series(pred_rates, "pred_rates") * float(step_hours)
    if true.shape != pred.shape:
        raise ValueError(f"形状不一致：{true.shape} vs {pred.shape}")

    def tv(a):
        return float(np.abs(np.diff(a, axis=-2)).sum()) if a.shape[-2] > 1 else 0.0

    tv_t, tv_p = tv(true), tv(pred)
    return {
        "total_variation_true_ml": tv_t,
        "total_variation_pred_ml": tv_p,
        "oscillation_ratio": (tv_p / tv_t) if tv_t > 0 else float("nan"),
        "unit": "cumulative_ml",
    }


# ------------------------------------------------------- 统一评估

def evaluate_all(true_labels: np.ndarray, pred_labels: np.ndarray,
                 step_hours: float = 1.0,
                 rate_definition: str = "rate") -> Dict[str, dict]:
    """在**标签形式** (N, T, 6) 上一次性算出全部终点。

    ``rate_definition`` 必须显式给出，并会写进每条结果的 ``rate_definition``
    字段——这是 #12 要求的"每个数字都能读出它用的是哪个口径"。
    """
    if rate_definition not in RATE_DEFINITIONS:
        raise ValueError(f"rate_definition 必须是 {RATE_DEFINITIONS} 之一")
    t = np.asarray(true_labels, dtype=float)
    p = np.asarray(pred_labels, dtype=float)
    if t.shape != p.shape or t.ndim != 3 or t.shape[-1] != 2 * lp.NUM_TYPES:
        raise ValueError(f"标签形状应为 (N, T, {2 * lp.NUM_TYPES}) 且一致，"
                         f"实际 {t.shape} vs {p.shape}")

    tr, pr = t[:, :, lp.RATE_SLICE], p[:, :, lp.RATE_SLICE]
    tb, pb = t[:, :, lp.BIT_SLICE], p[:, :, lp.BIT_SLICE]

    cls = M.classification_metrics(tb.reshape(-1, 3), pb.reshape(-1, 3))
    out: Dict[str, dict] = {
        "macro_f1": {"value": cls["macro"]["f1"], "aggregation": "macro",
                     "unit": "probability", "level": "window"},
        "micro_f1": {"value": cls["micro"]["f1"], "aggregation": "micro",
                     "unit": "probability", "level": "window"},
        "weighted_f1": {"value": cls["weighted"]["f1"], "aggregation": "weighted",
                        "unit": "probability", "level": "window"},
        "bit_accuracy": {"value": M.accuracy_bit(tb, pb), "aggregation": "micro",
                         "unit": "probability", "level": "window"},
        "mae_positive_flow": {
            "value": M.mae(tr, pr, unit="positive_flow_ml_per_h",
                           mask=M.positive_flow_mask(tr)),
            "aggregation": "none", "unit": "positive_flow_ml_per_h",
            "level": "window", "mask": "positive_flow"},
        "mae_step": {"value": M.mae(tr, pr, unit="step_ml_per_h"),
                     "aggregation": "none", "unit": "step_ml_per_h",
                     "level": "window", "mask": "all"},
    }

    # 累积/容差/SMAPE/稳定性按患者逐条算，再逐患者平均（患者等权）
    pids = np.zeros(len(t), dtype=int)          # 占位：调用方可用更高层封装分组
    cum_all = cumulative_dose(tr, pr, step_hours)
    out["cumulative_dose_mae"] = {
        "value": float(np.abs(cum_all["pred"][..., -1, :]
                              - cum_all["true"][..., -1, :]).mean()),
        "aggregation": "none", "unit": "cumulative_ml", "level": "window",
        "mask": "all"}
    tol = tolerance_accuracy(tr, pr, step_hours)
    out["tolerance_accuracy"] = {
        "value": tol["accuracy"], "aggregation": "none", "unit": "probability",
        "level": "window", "mask": "all",
        "note": f"±{tol['tol_abs_ml']} mL 且 ±{tol['tol_pct']:.0%}；"
                f"真值 0 的分母有 {tol['n_zero_denominator']} 个，只判绝对容差"}
    sm = smape_positive_flow(tr, pr)
    out["smape_positive_flow"] = {
        "value": sm["smape"], "aggregation": "none", "unit": "probability",
        "level": "window", "mask": "positive_flow",
        "note": f"分母为 0 跳过 {sm['n_skipped']} 个单元"}
    st = temporal_stability(pr, tr, step_hours)
    out["oscillation_ratio"] = {
        "value": st["oscillation_ratio"], "aggregation": "none",
        "unit": "ratio", "level": "window", "mask": "all",
        "note": "预测总变差 / 真实总变差；>1 表示预测比真实更抖"}

    for v in out.values():
        v["rate_definition"] = rate_definition
    return out


# ==========================================================================
# v2（issue #81）：患者优先的终点实现
# ==========================================================================
#
# 原则：**独立单位是患者**。每个终点都先对每位患者求值，再对患者等权平均；
# 目标加权（target_pooled）只作辅助，且必须显式标注。
#
# 本层返回**普通结果字典**（数值 + 分母 + 排除原因），不构造结果行——
# 行的 schema 由 utils/metrics.py 负责，装配由 utils/eval_engine.py 负责。
# 这样指标算术可以脱离 schema 单独手算校验。


def _by_class(x: np.ndarray, fluid_class: str) -> float:
    """按液体类别求和；``total`` 为三类之和。"""
    if fluid_class == "total":
        return float(np.sum(x))
    if fluid_class not in lp.LIQUID_TYPES:
        raise ValueError(f"未知的 fluid_class：{fluid_class}")
    return float(np.sum(x[lp.LIQUID_TYPES.index(fluid_class)]))


def patient_cumulative(traj, rate_definition: str, *, horizon: int,
                       origin: str = "hour_0_anchored", dt: float = 1.0
                       ) -> Dict[str, object]:
    """一位患者的逐类累计体积（mL），按冻结的估计对象计算。

    - ``hour_0_anchored``（E1）：第 0 小时用**已观测**的值补齐——它在决策
      时刻已知，对真值与预测是同一个数，因此不计入误差，也不是泄漏。
      于是真值与预测的差只来自**被预测的那部分**。
    - ``first_predicted_hour``（E3）：不锚定，只用轨迹里的小时求和。
    """
    if origin not in ("hour_0_anchored", "first_predicted_hour"):
        raise ValueError(
            f"origin 必须是 hour_0_anchored / first_predicted_hour，实际 {origin}")

    pred = traj.rates_for(rate_definition)
    sel = traj.select_hours(horizon)          # 实际存在、且小时序号 < horizon
    n_pred_hours = int(len(sel))

    true_class = (np.sum(traj.true_rates[sel], axis=0) * dt
                  if n_pred_hours else np.zeros(lp.NUM_TYPES))
    pred_class = (np.sum(pred[sel], axis=0) * dt
                  if n_pred_hours else np.zeros(lp.NUM_TYPES))

    anchored = False
    if origin == "hour_0_anchored" and 0 not in set(int(h) for h in traj.hours):
        if traj.observed_first is None:
            return {"available": False,
                    "na_reason": "hour_0_not_observed_for_anchored_estimand",
                    "n_predicted_hours": n_pred_hours}
        obs = np.asarray(traj.observed_first, dtype=float)[lp.RATE_SLICE] * dt
        true_class = true_class + obs          # 观测值同样是这一小时的真值
        pred_class = pred_class + obs
        anchored = True

    true_ml = float(np.sum(true_class))
    pred_ml = float(np.sum(pred_class))
    abs_err_class = np.abs(pred_class - true_class)
    return {
        "available": True,
        "true_ml": true_ml,
        "pred_ml": pred_ml,
        # 总量口径：|总量差|。**不是**逐类绝对误差之和——后者会把
        # "晶体少 500、胶体多 500" 也算成 1000 的误差，见 component_abs_err_ml。
        "abs_err_ml": abs(pred_ml - true_ml),
        "true_class_ml": true_class,
        "pred_class_ml": pred_class,
        "abs_err_class_ml": abs_err_class,
        # 分量口径：Σ_c |Δ_c|。只有它能为正于总量误差，从而暴露通道间抵消。
        "component_abs_err_ml": float(np.sum(abs_err_class)),
        "n_predicted_hours": n_pred_hours,
        "hour0_anchored": anchored,
    }


def horizon_eligible(traj, horizon: int, *, require_complete: bool = True,
                     rule: str = "all_hours_present") -> Dict[str, object]:
    """该患者在某时域下是否可用；不可用时给出原因。

    ``require_complete=True``（默认）要求轨迹**完整覆盖**该时域。这是
    防止"30 小时的部分轨迹被当作 48h/72h 求和"的闸门——缺的小时是缺的，
    不能靠"有多少算多少"补齐。

    ``rule`` 决定"完整"的含义：

    - ``all_hours_present``：该时域的小时都在（主轨）。
    - ``all_hours_eligible``：而且这些小时**全部通过质量掩码**（敏感性轨）。

    两轨必须用**不同的规则**，否则敏感性分析只是把同一批数字换个标题——
    那正是双轨设计要防的假象。
    """
    if not require_complete:
        return {"eligible": True, "na_reason": ""}
    qm = getattr(traj, "_quality_mask", None)
    c = traj.completeness(horizon, rule=rule, quality_mask=qm)
    if c["complete"]:
        return {"eligible": True, "na_reason": "", "rule": rule,
                "hour0_anchored": c["hour0_anchored"]}
    reason = ("insufficient_horizon" if rule == "all_hours_present"
              else "not_estimable_under_quality_mask")
    return {"eligible": False, "na_reason": reason, "rule": rule,
            "hours_present": c["hours_present"],
            "hours_required": c["hours_required"]}


def cumulative_summary(trajs, rate_definition: str, *, horizon: int,
                       origin: str = "hour_0_anchored", dt: float = 1.0,
                       require_complete: bool = True,
                       rule: str = "all_hours_present"
                       ) -> Dict[str, object]:
    """P1：累计剂量 MAE（mL），**先患者后平均**，并分液体并列报告。

    分液体报告是必须的：总量误差允许三个通道的正负误差互相抵消，
    而"总量对、液体种类错"是临床上不同的失败模式（见 component_sum_error）。
    """
    eligible, excluded = [], []
    for t in trajs:
        gate = horizon_eligible(t, horizon, require_complete=require_complete,
                                    rule=rule)
        if not gate["eligible"]:
            excluded.append((t.patient_id, {"na_reason": gate["na_reason"]}))
            continue
        r = patient_cumulative(t, rate_definition, horizon=horizon,
                               origin=origin, dt=dt)
        (eligible if r["available"] else excluded).append((t.patient_id, r))

    n_pred_hours = [r["n_predicted_hours"] for _, r in eligible]
    out = {
        "available": bool(eligible),
        "n_patients_eligible": len(eligible),
        "n_patients_excluded": len(excluded),
        "exclusion_reasons": sorted({r["na_reason"] for _, r in excluded}),
        "n_predicted_hours_total": int(np.sum(n_pred_hours)) if n_pred_hours else 0,
        "horizon_hours": int(horizon),
        "origin": origin,
        "estimand": ("conditional_cumulative_volume"
                     if origin == "hour_0_anchored"
                     else "pure_prediction_window_volume"),
        "per_class": {},
    }
    if not eligible:
        return out

    for fluid in ("total",) + tuple(lp.LIQUID_TYPES):
        k = None if fluid == "total" else lp.LIQUID_TYPES.index(fluid)
        errs = np.array([(r["abs_err_ml"] if k is None else r["abs_err_class_ml"][k])
                         for _, r in eligible], dtype=float)
        signed = np.array([(r["pred_ml"] - r["true_ml"]) if k is None
                           else (r["pred_class_ml"][k] - r["true_class_ml"][k])
                           for _, r in eligible], dtype=float)
        out["per_class"][fluid] = {
            "mae_ml": float(errs.mean()),
            "bias_ml": float(signed.mean()),
            "n_patients": int(errs.size),
            "n_targets": int(np.sum(n_pred_hours)),
        }
    return out


def tolerance_summary(trajs, rate_definition: str, *, horizon: int,
                       origin: str = "hour_0_anchored", dt: float = 1.0,
                      tol_abs_ml: float = TOL_ABS_ML,
                      tol_pct: float = TOL_PCT,
                      require_complete: bool = True,
                      rule: str = "all_hours_present") -> Dict[str, object]:
    """P2：容差准确率——**绝对与相对是两个独立主终点**，各自报分母。

    冻结规则：

    - 合格患者 = 该时域完整；
    - ``abs`` 在**全部**合格患者上算：零剂量患者也有"少输了多少"的绝对误差；
    - ``rel`` **排除真实累计总量为 0 的患者**——除以零不是"通过"；
    - ``joint`` 另行报告，**不得**替代上面两个；
    - 零剂量单列一个桶：人数 + 只判绝对容差的准确率。
    """
    rows = []
    for t in trajs:
        if not horizon_eligible(t, horizon, require_complete=require_complete,
                                    rule=rule)["eligible"]:
            continue
        r = patient_cumulative(t, rate_definition, horizon=horizon,
                               origin=origin, dt=dt)
        if r["available"]:
            rows.append((t.patient_id, r))
    if not rows:
        return {"available": False, "n_patients_eligible": 0,
                "exclusion_reasons": ["no_eligible_patient"]}

    true_v = np.array([r["true_ml"] for _, r in rows], dtype=float)
    pred_v = np.array([r["pred_ml"] for _, r in rows], dtype=float)
    abs_err = np.abs(pred_v - true_v)
    is_zero = true_v <= 0.0

    abs_ok = abs_err <= float(tol_abs_ml)
    safe = np.where(is_zero, 1.0, true_v)
    rel_err = abs_err / safe
    rel_ok = rel_err <= float(tol_pct)
    nonzero = ~is_zero

    return {
        "available": True,
        "tol_abs_ml": float(tol_abs_ml), "tol_pct": float(tol_pct),
        "horizon_hours": int(horizon), "origin": origin,
        "n_patients_eligible": int(true_v.size),
        "abs": {"value": float(abs_ok.mean()), "n_patients": int(abs_ok.size),
                "n_hits": int(abs_ok.sum())},
        "rel": {"value": (float(rel_ok[nonzero].mean()) if nonzero.any()
                          else float("nan")),
                "n_patients": int(nonzero.sum()),
                "n_hits": int(rel_ok[nonzero].sum()),
                "n_excluded_zero_dose": int(is_zero.sum())},
        "joint": {"value": (float((abs_ok[nonzero] & rel_ok[nonzero]).mean())
                            if nonzero.any() else float("nan")),
                  "n_patients": int(nonzero.sum())},
        "zero_dose": {
            "n_patients": int(is_zero.sum()),
            "accuracy_abs": (float(abs_ok[is_zero].mean()) if is_zero.any()
                             else float("nan")),
            "pred_total_ml": pred_v[is_zero].tolist(),
            "note": "真实累计总量为 0 的患者不进相对容差的分母；除零不是达标。",
        },
    }


def component_sum_error(trajs, rate_definition: str, *, horizon: int,
                           origin: str = "hour_0_anchored", dt: float = 1.0,
                        require_complete: bool = True,
                      rule: str = "all_hours_present") -> Dict[str, object]:
    """C2 的核心量：分量误差和 ``Σ_c |V̂_c − V_c|``（mL）。

    它与 ``|Σ_c V̂_c − Σ_c V_c|`` 不同：后者允许三个通道的正负误差互相抵消，
    于是"晶体少输 500 mL、胶体多输 500 mL"会显示为总量完全正确。
    两者一起看才能识别这种失败。
    """
    comp, tot = [], []
    for t in trajs:
        if not horizon_eligible(t, horizon, require_complete=require_complete,
                                    rule=rule)["eligible"]:
            continue
        r = patient_cumulative(t, rate_definition, horizon=horizon,
                               origin=origin, dt=dt)
        if not r["available"]:
            continue
        comp.append(float(r["component_abs_err_ml"]))
        tot.append(float(r["abs_err_ml"]))
    if not comp:
        return {"available": False}
    comp_a, tot_a = np.array(comp), np.array(tot)
    return {
        "available": True,
        "component_sum_mae_ml": float(comp_a.mean()),
        "total_abs_mae_ml": float(tot_a.mean()),
        "cancellation_gap_ml": float(comp_a.mean() - tot_a.mean()),
        "n_patients": int(comp_a.size),
        "unit": "mL",
        "note": "cancellation_gap = 分量误差和 − 总量误差；"
                ">0 说明存在通道间正负抵消。",
    }


# --------------------------------------------------------------- 速率误差

def rate_error_summary(trajs, rate_definition: str, *, mask: str = "all",
                       eps: float = None, fluid_class: str = "total"
                       ) -> Dict[str, object]:
    """逐小时速率误差（mL/h），**先患者后平均**。

    ``mask`` 必须是 ``all`` / ``positive_flow`` / ``zero_flow`` 之一：

    - ``all``：包含真实零流小时（含真实零输注）。这是"全时段"口径。
    - ``positive_flow``：只算真实速率 > ε 的单元。**它不是 all**——
      零流小时会把 all 的均值拉低，两者数值不同、问题不同。
    - ``zero_flow``：只算真值为零的单元，用于单独看"该停没停"。

    ``target_pooled`` 一并给出，但只作辅助：它把同一患者的多个小时当作独立
    样本，患者小时数不等时会偏向小时多的患者。

    ``fluid_class`` 取 ``total``（三类合并）或某一类，用于按液体分别报告——
    总量误差会掩盖"某一类系统性偏高、另一类偏低"。
    """
    if mask not in ("all", "positive_flow", "zero_flow"):
        raise ValueError(f"mask 必须是 all / positive_flow / zero_flow，实际 {mask}")
    if fluid_class != "total" and fluid_class not in lp.LIQUID_TYPES:
        raise ValueError(f"未知的 fluid_class：{fluid_class}")
    thr = EP.EPS_RATE_ML_PER_H if eps is None else float(eps)
    cols = (list(range(lp.NUM_TYPES)) if fluid_class == "total"
            else [lp.LIQUID_TYPES.index(fluid_class)])

    per_patient_mae, per_patient_rmse = [], []
    pooled_err, n_targets, n_patient_hours = [], 0, 0
    for t in trajs:
        pred = t.rates_for(rate_definition)
        if pred is None:
            continue
        true = t.true_rates[:, cols]
        pred = pred[:, cols]
        if mask == "positive_flow":
            sel = true > thr
        elif mask == "zero_flow":
            sel = true <= thr
        else:
            sel = np.ones_like(true, dtype=bool)
        if not sel.any():
            continue
        err = np.abs(pred[sel] - true[sel])
        per_patient_mae.append(float(np.mean(err)))
        per_patient_rmse.append(float(np.sqrt(np.mean(err ** 2))))
        pooled_err.append(err)
        n_targets += int(sel.sum())
        n_patient_hours += int(np.any(sel, axis=1).sum())

    if not per_patient_mae:
        return {"available": False, "mask": mask, "n_patients": 0,
                "na_reason": "no_target_matches_mask"}
    pooled = np.concatenate(pooled_err)
    return {
        "available": True,
        "mask": mask, "epsilon_ml_per_h": float(thr),
        "fluid_class": fluid_class,
        "unit": "mL/h",
        "mae_ml_per_h": float(np.mean(per_patient_mae)),
        "rmse_ml_per_h": float(np.mean(per_patient_rmse)),
        "n_patients": len(per_patient_mae),
        "n_patient_hours": n_patient_hours,
        "n_targets": n_targets,
        # 辅助口径：把 (小时, 类别) 当独立单元
        "target_pooled": {"mae_ml_per_h": float(pooled.mean()),
                          "rmse_ml_per_h": float(np.sqrt(np.mean(pooled ** 2))),
                          "n_targets": int(pooled.size)},
        "aggregation": "patient_mean",
        "note": "patient_mean：先按患者对有效小时/类别平均，再对患者平均。"
                "target_pooled 仅作辅助。",
    }


def smape_summary(trajs, rate_definition: str, *, mask: str = "positive_flow",
                  eps: float = None) -> Dict[str, object]:
    """SMAPE（%），分母为 ``|ŷ| + |y|``；分母为 0 的单元跳过**并计数**。

    按 0 计入会人为压低误差，所以跳过时必须报出 ``n_skipped``。
    """
    thr = EP.EPS_RATE_ML_PER_H if eps is None else float(eps)
    per_patient, n_skipped, n_used = [], 0, 0
    for t in trajs:
        pred = t.rates_for(rate_definition)
        if pred is None:
            continue
        true = t.true_rates
        sel = (true > thr) if mask == "positive_flow" else np.ones_like(true, bool)
        if not sel.any():
            continue
        a, b = true[sel], pred[sel]
        denom = np.abs(a) + np.abs(b)
        keep = denom > 0
        n_skipped += int((~keep).sum())
        if not keep.any():
            continue
        per_patient.append(float(np.mean(200.0 * np.abs(b[keep] - a[keep]) / denom[keep])))
        n_used += int(keep.sum())

    if not per_patient:
        return {"available": False, "mask": mask, "n_patients": 0,
                "n_skipped": n_skipped}
    return {
        "available": True, "mask": mask, "unit": "ratio",
        "smape_percent": float(np.mean(per_patient)),
        "n_patients": len(per_patient), "n_targets": n_used,
        "n_skipped": n_skipped,
        "aggregation": "patient_mean",
    }


# ------------------------------------------------------------- 分类（P3）

def multilabel_prf(y_true_bits: np.ndarray, y_pred_bits: np.ndarray,
                   *, class_names=None) -> Dict[str, object]:
    """P3：多标签逐类 P/R/F1 + macro/micro，含**无正例类**的冻结规则。

    冻结规则：主指标是**固定三个类别**的 Macro-F1，``zero_division=0``——
    与稿件已报的口径可比。某一类在真值里没有正例时该类的 F1 记 0 并进入
    均值，**不改变指标定义**；另给 ``present_class_macro_f1`` 作为敏感性
    （只在有正例的类上取均值），两者数值不同、不得互换。

    例：三类 F1 = 0.6 / 0.8 / 无正例 → 主指标 0.4667，敏感性 0.7000。
    """
    from sklearn.metrics import (f1_score, precision_score, recall_score)

    y = np.asarray(y_true_bits).astype(int)
    p = np.asarray(y_pred_bits).astype(int)
    if y.shape != p.shape:
        raise ValueError(f"形状不一致：{y.shape} vs {p.shape}")
    K = y.shape[1]
    if class_names:
        names = list(class_names)
    elif K == lp.NUM_TYPES:
        names = list(lp.LIQUID_TYPES)          # 本项目的液体口径
    else:
        names = [f"class_{k}" for k in range(K)]

    per_class, supports, f1s = {}, [], []
    for k in range(K):
        f1 = float(f1_score(y[:, k], p[:, k], zero_division=0))
        per_class[names[k]] = {
            "precision": float(precision_score(y[:, k], p[:, k], zero_division=0)),
            "recall": float(recall_score(y[:, k], p[:, k], zero_division=0)),
            "f1": f1,
            "support": int(y[:, k].sum()),
        }
        supports.append(per_class[names[k]]["support"])
        f1s.append(f1)

    present = [f for f, s in zip(f1s, supports) if s > 0]
    return {
        "available": True,
        "per_class": per_class,
        # 主指标：固定 K 类，zero_division=0
        "macro_f1": float(np.mean(f1s)),
        "macro_precision": float(precision_score(y, p, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, p, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, p, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
        "bit_accuracy": float((y == p).mean()),
        "n_targets": int(y.size), "n_patients_note": "调用方按患者聚合后填 n_patients",
        # 敏感性：只在有正例的类上取均值
        "present_class_macro_f1": (float(np.mean(present)) if present else float("nan")),
        "n_classes_in_main_macro": K,
        "n_classes_in_present_macro": len(present),
        "n_classes_without_positives": K - len(present),
        "macro_rule": "fixed_classes_zero_division_0",
    }


def multilabel_ranking(y_true_bits: np.ndarray, y_prob: np.ndarray,
                       *, n_bins: int = 10) -> Dict[str, object]:
    """逐类 ROC-AUC、Brier 与 ECE（macro）。

    某类在真值里只有单一标签时 AUC **无法计算**，记 ``None`` 并给出原因——
    不补造 0.5，也不把它算进 macro（那会把"不可算"变成"随机水平"）。
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score

    from utils import calibration as CAL

    y = np.asarray(y_true_bits).astype(int)
    q = np.asarray(y_prob, dtype=float)
    if y.shape != q.shape:
        raise ValueError(f"形状不一致：{y.shape} vs {q.shape}")

    aucs, briers, eces = [], [], []
    for k in range(y.shape[1]):
        if len(np.unique(y[:, k])) < 2:
            aucs.append(None)
        else:
            aucs.append(float(roc_auc_score(y[:, k], q[:, k])))
        briers.append(float(brier_score_loss(y[:, k], q[:, k])))
        eces.append(float(CAL.expected_calibration_error(
            y[:, k][:, None], q[:, k][:, None], n_bins=n_bins)["ece"]))

    finite = [a for a in aucs if a is not None]
    return {
        "available": True,
        "auc_per_class": aucs,
        "auc_macro": float(np.mean(finite)) if finite else float("nan"),
        "auc_macro_n_classes": len(finite),
        "auc_na_reason": ("single_class_labels_in_split" if len(finite) < len(aucs) else ""),
        "brier_per_class": briers,
        "brier_mean": float(np.mean(briers)),
        "ece_per_class": eces,
        "ece_macro": float(np.mean(eces)),
        "n_bins": int(n_bins),
    }


# --------------------------------------------------------------- 时间稳定性

def total_variation_summary(trajs, rate_definition: str, *,
                            dt: float = 1.0) -> Dict[str, object]:
    """总变差差异：``|TV(ŷ) − TV(y)|``，逐患者再平均（mL）。

    ``TV(r) = Σ_t |r_t − r_{t−1}|``。一个恒定预测的总变差是 0——它很"稳定"，
    但完全不做调整。因此本指标**必须与 change-point recall 配对展示**，
    否则"稳定"会被误读为优点。
    """
    diffs, ratios, tvs_t, tvs_p = [], [], [], []
    for t in trajs:
        pred = t.rates_for(rate_definition)
        if pred is None or t.n_hours < 2:
            continue
        true = t.true_rates

        def _tv(a):
            return float(np.sum(np.abs(np.diff(a, axis=0))) * dt)

        tv_t, tv_p = _tv(true), _tv(pred)
        tvs_t.append(tv_t)
        tvs_p.append(tv_p)
        diffs.append(abs(tv_p - tv_t))
        if tv_t > 0:
            ratios.append(tv_p / tv_t)

    if not diffs:
        return {"available": False, "n_patients": 0}
    return {
        "available": True, "unit": "mL",
        "tv_abs_diff_ml": float(np.mean(diffs)),
        "tv_true_ml": float(np.mean(tvs_t)),
        "tv_pred_ml": float(np.mean(tvs_p)),
        "tv_ratio": float(np.mean(ratios)) if ratios else float("nan"),
        "n_patients": len(diffs),
        "n_patients_with_ratio": len(ratios),
        "note": "ratio > 1 表示预测比真实更抖；ratio < 1 可能是「不敢调整」，"
                "两种读法都必须与 change-point recall 一起看。",
    }
