"""§2A 医学过程指标：C1、C2、N1（调速一致性）、PS-TC、N2（CETE）。

证据分级（每条结果都必须携带）
------------------------------
- **C-A / MUST**：可由现有预测与体重/TBSA 直接算出（C1、C2）。
- **C-B / CONDITIONAL**：需要真实、时间有效、质量合格的生理记录（PS-TC）。
  本仓库的建模产物**不满足**该前提，故 PS-TC 一律输出 NA + 原因。
- **N1 / N2**：``proposed_exploratory``——本研究的拟议定义，**未经临床验证**，
  不得取代 P1/P2/P3，也不得凭其数值宣称医学优越性。

不可算就报 NA
-------------
体重、TBSA 缺失或超出合理范围（体重 5–200 kg、TBSA 1–100%）时**不猜、不用 0 顶替**，
输出 ``NA + na_reason`` 并报出不可计算的比例。这是本模块与"先补一个数再说"
的分界线。

指南参考值不是金标准
--------------------
ABA 2024 的 2 mL/kg/%TBSA 是**起始估算建议**，面向成人 ≥20% TBSA、主要聚焦
伤后头 48 小时。它不是本队列的个体正确答案，也不是安全上限，**不得**用作
达标率的分母。本模块只把它作为 ``guidance_only`` 的参考行输出。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import eval_protocol as EP
from utils import label_protocol as lp

#: 合理的体重范围（kg）与 TBSA 范围（%）。超出即视为不可计算，而不是截断。
WEIGHT_RANGE_KG: Tuple[float, float] = (5.0, 200.0)
TBSA_RANGE_PCT: Tuple[float, float] = (1.0, 100.0)

#: 指南参考值（mL/kg/%TBSA）。**仅作参考行**，不是目标值。
GUIDANCE_ML_PER_KG_TBSA: float = 2.0

#: 可计算的分层维度。**都不是生理状态**——见 PS-TC 的说明。
AVAILABLE_STRATA: Tuple[str, ...] = ("hour_band", "observed_rate_level")

#: 证据分层标签。
TIER_MUST = "derived"
TIER_PROPOSED = "proposed_exploratory"
TIER_GUIDANCE = "guidance_only"
TIER_NOT_ASSESSABLE = "not_assessable"


def _normalizer(traj) -> Optional[float]:
    """返回 ``weight_kg × TBSA_percent``；任一缺失或不合理即返回 None。"""
    w, t = traj.weight_kg, traj.tbsa_percent
    if w is None or t is None:
        return None
    if not (WEIGHT_RANGE_KG[0] <= w <= WEIGHT_RANGE_KG[1]):
        return None
    if not (TBSA_RANGE_PCT[0] <= t <= TBSA_RANGE_PCT[1]):
        return None
    return float(w) * float(t)


# --------------------------------------------------------------------------
# C1：早期复苏液体暴露强度
# --------------------------------------------------------------------------

def early_resuscitation_exposure(trajs, rate_definition: str, *, horizon: int = 24,
                                 dt: float = 1.0,
                                 guidance: float = GUIDANCE_ML_PER_KG_TBSA,
                                 rule: str = "all_hours_present",
                                 require_complete: bool = True
                                 ) -> Dict[str, object]:
    """C1：``V_cryst(0–T) / (weight_kg × TBSA_percent)``，单位 mL/kg/%TBSA。

    同时给出真实临床值与模型的条件性预测值、逐患者差值、合格人数，以及
    **不可计算的比例**（体重/TBSA 缺失或超范围）——缺失本身是结论的一部分。

    三处必须写明的边界：

    1. 起点是否为**伤后**时间、院前转运输液、推注/维持液/口服是否计入，
       都会改变"总量"的含义；本函数只按传入轨迹求和，起点语义由调用方负责。
    2. 指南的 2 mL/kg/%TBSA 是**起始估算建议**，面向成人 ≥20% TBSA 且主要
       聚焦伤后 48h，**不得**套到整队列 72h，也不是个体"安全上限"。
    3. 这里是**观察性**暴露强度的复现，不是"模型给药是否更合适"。
    """
    rows, unavailable = [], 0
    unavailable_reasons: List[str] = []

    for traj in trajs:
        norm = _normalizer(traj)
        if norm is None:
            unavailable += 1
            reason = ("weight_or_tbsa_missing" if (traj.weight_kg is None
                                                   or traj.tbsa_percent is None)
                      else "weight_or_tbsa_out_of_range")
            unavailable_reasons.append(reason)
            continue
        # 晶体累计（0–T）：用真实速率；预测用同一时域的动作速率
        from utils.endpoints import horizon_eligible, patient_cumulative
        gate = horizon_eligible(traj, horizon, rule=rule,
                                require_complete=require_complete)
        if not gate["eligible"]:
            unavailable += 1
            unavailable_reasons.append(gate["na_reason"])
            continue
        r = patient_cumulative(traj, rate_definition, horizon=horizon,
                               origin="hour_0_anchored", dt=dt)
        if not r["available"]:
            unavailable += 1
            unavailable_reasons.append(r["na_reason"])
            continue
        k = lp.LIQUID_TYPES.index("crystal")
        rows.append({
            "patient_id": traj.patient_id,
            "true_ml_per_kg_tbsa": float(r["true_class_ml"][k]) / norm,
            "pred_ml_per_kg_tbsa": float(r["pred_class_ml"][k]) / norm,
            "true_ml": float(r["true_class_ml"][k]),
            "pred_ml": float(r["pred_class_ml"][k]),
            "weight_kg": float(traj.weight_kg),
            "tbsa_percent": float(traj.tbsa_percent),
        })

    n_total = len(trajs)
    out: Dict[str, object] = {
        "available": bool(rows),
        "horizon_hours": int(horizon),
        "unit": "mL/kg/%TBSA",
        "fluid_class": "crystal",
        "n_patients_total": n_total,
        "n_patients_eligible": len(rows),
        "n_patients_unavailable": unavailable,
        "missing_fraction": (unavailable / n_total) if n_total else float("nan"),
        "unavailable_reasons": sorted(set(unavailable_reasons)),
        "evidence_tier": TIER_MUST,
        "observation_vs_imputation": "observed",
        "clinical_definition_source": "Cartotto et al., ABA burn shock resuscitation "
                                      "guideline, J Burn Care Res 2024;45:565-589",
        "clinical_applicability": "adult, >=20% TBSA, predominantly first 48h post-injury",
        "interpretation_scope": "观察到的复苏暴露强度复现；不是最优处方，也不是疗效证据",
        # 指南参考行：独立、可追溯、明确不是目标
        "guidance": {
            "value_ml_per_kg_tbsa": float(guidance),
            "evidence_tier": TIER_GUIDANCE,
            "role": "reference_only",
            "note": "指南起始估算建议，**不是**个体金标准目标或安全上限；"
                    "不得用作达标率的分母。",
        },
    }
    if not rows:
        out["na_reason"] = "no_patient_with_usable_weight_and_tbsa"
        return out

    true_v = np.array([r["true_ml_per_kg_tbsa"] for r in rows], dtype=float)
    pred_v = np.array([r["pred_ml_per_kg_tbsa"] for r in rows], dtype=float)
    out.update({
        "true_mean": float(true_v.mean()),
        "pred_mean": float(pred_v.mean()),
        "mean_diff": float((pred_v - true_v).mean()),
        "mean_abs_diff": float(np.abs(pred_v - true_v).mean()),
        "true_median": float(np.median(true_v)),
        "true_p10": float(np.percentile(true_v, 10)),
        "true_p90": float(np.percentile(true_v, 90)),
        "n_above_guidance_true": int((true_v > guidance).sum()),
        "n_above_guidance_pred": int((pred_v > guidance).sum()),
        "per_patient": rows,
        "note": "n_above_guidance_* 只描述分布位置，**不构成达标率**——"
                "指南值不是目标。",
    })
    return out


# --------------------------------------------------------------------------
# C2：分液体剂量组成忠实度
# --------------------------------------------------------------------------

def per_fluid_dose_error(trajs, rate_definition: str, *, horizon: int,
                         origin: str = "hour_0_anchored", dt: float = 1.0,
                         require_complete: bool = True) -> Dict[str, object]:
    """C2：逐液体累计剂量误差 + 总量误差 + 分量误差和。"""
    from utils.endpoints import component_sum_error, cumulative_summary

    per_fluid = cumulative_summary(trajs, rate_definition, horizon=horizon,
                                   origin=origin, dt=dt,
                                   require_complete=require_complete)
    comp = component_sum_error(trajs, rate_definition, horizon=horizon,
                               origin=origin, dt=dt,
                               require_complete=require_complete)
    return {
        "available": bool(per_fluid.get("available")) and bool(comp.get("available")),
        "horizon_hours": int(horizon), "origin": origin,
        "per_fluid": per_fluid.get("per_class", {}),
        "component_sum": comp,
        "n_patients": per_fluid.get("n_patients_eligible", 0),
        "n_predicted_hours": per_fluid.get("n_predicted_hours_total", 0),
        "evidence_tier": TIER_MUST,
        "observation_vs_imputation": "observed",
        "interpretation_scope": "识别「总量一致但液体种类相互抵消」的失败模式",
        "note": "逐类误差与总量误差并列；component_sum 为正说明存在通道间抵消。",
    }


# --------------------------------------------------------------------------
# N1：调速一致性（可计算、描述性）
# --------------------------------------------------------------------------

def _hour_band(hour: int) -> str:
    if hour < 24:
        return "0-23h"
    if hour < 48:
        return "24-47h"
    return "48h+"


def _rate_level(rate: float) -> str:
    if rate <= EP.EPS_RATE_ML_PER_H:
        return "zero"
    if rate <= 100.0:
        return "0-100"
    return ">100"


def titration_concordance(trajs, rate_definition: str, *,
                          delta: float = None,
                          stratum: Optional[str] = None,
                          strata: Optional[Dict[str, Sequence[str]]] = None
                          ) -> Dict[str, object]:
    """N1：真实变化事件上的**方向一致率**。

    定义（对每一步 ``t → t+1``、每一类 ``c``）：

    - 真实变化 ``Δr = r_{t+1,c} − r_{t,c}``；
    - 预测变化 ``Δr̂ = r̂_{t+1,c} − r_{t,c}``——基准是 ``t`` 时刻**真实且已知**
      的速率，因此这衡量的正是"下一步会往哪个方向调"；
    - 变化事件 ``E = {|Δr| > δ}``，``δ`` 取自冻结的变点阈值；
    - ``DC = mean_{E} 1[sign(Δr̂) = sign(Δr)]``。

    **必须与误报率一起读**：一个"每个小时都在乱变"的预测器可以拿到很高的
    DC，所以同时给出非变化时段的 ``false_change_rate`` 与变化时段的 MAE。

    解释边界：这是"是否更好地预测了医生在某一分层下的调速动作"，
    **不是**"模型正确响应了休克/纠正了低尿量"。低尿量患者未必需要加液。

    ``stratum`` 只接受 :data:`AVAILABLE_STRATA` 中的**非生理**维度；生理状态
    分层属于 PS-TC，缺失时在那里报 NA，**不在这里用小时段冒充**。
    """
    if delta is None:
        delta = EP.CHANGE_ABS_ML
    if stratum is not None and stratum not in AVAILABLE_STRATA:
        raise ValueError(
            f"stratum 只支持 {AVAILABLE_STRATA}；生理状态分层属于 PS-TC，"
            "不得用小时段冒充")

    events: List[dict] = []
    total_steps = 0
    for traj in trajs:
        pred = traj.rates_for(rate_definition)
        if pred is None or traj.n_hours < 2:
            continue
        true = traj.true_rates
        labels = None
        if strata is not None and traj.patient_id in strata:
            labels = list(strata[traj.patient_id])
        for i in range(traj.n_hours - 1):
            hour = int(traj.hours[i])
            total_steps += 1
            d_true = true[i + 1] - true[i]
            d_pred = pred[i + 1] - true[i]          # 基准 = t 时刻真实速率
            for c in range(lp.NUM_TYPES):
                if abs(d_true[c]) <= delta:
                    continue
                if labels is not None:
                    tag = labels[i]
                elif stratum == "hour_band":
                    tag = _hour_band(hour)
                elif stratum == "observed_rate_level":
                    tag = _rate_level(float(true[i].sum()))
                else:
                    tag = "all"
                events.append({
                    "stratum": tag,
                    "sign_match": int(np.sign(d_pred[c]) == np.sign(d_true[c])),
                    "abs_err": float(abs(pred[i + 1, c] - true[i + 1, c])),
                })

    n_change_cells = sum(1 for _ in events)
    out: Dict[str, object] = {
        "available": bool(events),
        "delta_ml_per_h": float(delta),
        "stratum": stratum or "all",
        "n_steps": int(total_steps),
        "n_change_events": int(n_change_cells),
        "n_patients": len(trajs),
        "evidence_tier": TIER_PROPOSED,
        "clinical_applicability": "not_clinically_validated",
        "expert_review_status": "not_reviewed",
        "interpretation_scope": "是否更好地预测医生在该分层下的调速动作；"
                               "不是「模型正确响应了休克」",
    }
    if not events:
        out["na_reason"] = "no_true_change_events"
        return out

    by_stratum: Dict[str, List[dict]] = {}
    for e in events:
        by_stratum.setdefault(e["stratum"], []).append(e)

    def _summary(rows: List[dict]) -> Dict[str, float]:
        match = np.array([r["sign_match"] for r in rows], dtype=float)
        err = np.array([r["abs_err"] for r in rows], dtype=float)
        return {"direction_concordance": float(match.mean()),
                "change_hour_mae_ml_per_h": float(err.mean()),
                "n_events": int(match.size)}

    out["overall"] = _summary(events)
    out["per_stratum"] = {k: _summary(v) for k, v in sorted(by_stratum.items())}
    # 误报：非变化单元上预测却变了
    fp = 0
    opportunities = 0
    for traj in trajs:
        pred = traj.rates_for(rate_definition)
        if pred is None or traj.n_hours < 2:
            continue
        true = traj.true_rates
        for i in range(traj.n_hours - 1):
            d_true = true[i + 1] - true[i]
            d_pred = pred[i + 1] - true[i]
            for c in range(lp.NUM_TYPES):
                if abs(d_true[c]) > delta:
                    continue
                opportunities += 1
                if abs(d_pred[c]) > delta:
                    fp += 1
    out["false_change_rate"] = (fp / opportunities) if opportunities else float("nan")
    out["n_false_changes"] = int(fp)
    out["n_non_change_cells"] = int(opportunities)
    return out


def ps_tc(trajs, rate_definition: str, *, physiology_strata=None,
          **kwargs) -> Dict[str, object]:
    """PS-TC：**生理状态**分层的调速一致率。

    与 :func:`titration_concordance` 的唯一区别是分层变量必须是真的生理状态
    （低尿量、血流动力学、乳酸等），且必须**在 t 时刻可见**。

    本仓库不满足该前提，因此这里**不计算、只声明**：建模产物中的尿量是
    z-score 且按体重归一化后的值（不是 mL/kg/h），且没有 MAP、乳酸、碱剩余。
    用小时段或观测速率水平顶替生理状态会得到一个名字正确、含义错误的指标，
    所以本函数在缺少分层时返回 ``NA + na_reason``，而不是退化成 N1。
    """
    if physiology_strata is None:
        return {
            "available": False,
            "na_reason": "no_valid_physiology_state_available",
            "evidence_tier": TIER_NOT_ASSESSABLE,
            "observation_vs_imputation": "unavailable",
            "clinical_applicability": "requires observed urine output (mL/kg/h) or "
                                      "haemodynamic variables available before t",
            "missing_fraction": 1.0,
            "note": "建模产物中的尿量已 z-score 且按体重归一化，不是 mL/kg/h；"
                    "无 MAP / 乳酸 / 碱剩余。生理状态分层不可得，"
                    "本终点**不可估计**——不得用小时段冒充。",
        }
    return titration_concordance(trajs, rate_definition,
                                 strata=physiology_strata, **kwargs)


# --------------------------------------------------------------------------
# N2：累计暴露轨迹误差（CETE）
# --------------------------------------------------------------------------

def cumulative_exposure_trajectory_error(trajs, rate_definition: str, *,
                                         horizons: Sequence[int] = (24, 48, 72),
                                         dt: float = 1.0,
                                         require_complete: bool = True,
                                         rule: str = "all_hours_present"
                                         ) -> Dict[str, object]:
    """N2：CETE，累计暴露**轨迹**误差（不只是终末总量）。

    ``CETE_i(T) = [Σ_k |V̂(t_k) − V(t_k)| · Δt_k] / [T · weight_i]``，单位 mL/kg，
    其中 ``T = Σ_k Δt_k``（小时）。

    与只看终末 ``|V̂(T) − V(T)|`` 的区别：CETE 会暴露"早期过量、晚期不足"
    这类在终末总量上互相抵消的轨迹失败。

    同时给出**未按体重归一化**的版本（mL），供体重缺失时使用；
    体重不可用时归一化版本按患者跳过并报出人数。

    边界：``eval_mode`` 决定它的读法——教师强制下的 CETE 是**条件预测轨迹的
    忠实度**，不是自主闭环下的长期暴露。
    """
    out: Dict[str, object] = {
        "available": False, "horizons": {},
        "evidence_tier": TIER_PROPOSED,
        "clinical_applicability": "not_clinically_validated",
        "expert_review_status": "not_reviewed",
        "interpretation_scope": "条件预测轨迹的忠实度；不是自主闭环长期暴露",
        "unit_normalized": "mL/kg", "unit_absolute": "mL",
    }
    for horizon in horizons:
        from utils.endpoints import horizon_eligible

        per_patient_norm, per_patient_abs, hours_used = [], [], []
        n_skipped_weight = 0
        n_excluded = 0
        for traj in trajs:
            pred = traj.rates_for(rate_definition)
            if pred is None:
                continue
            # 与 P1/P2/C2 用同一套资格规则：部分轨迹不得冒充该时域
            if not horizon_eligible(traj, horizon,
                                    require_complete=require_complete,
                                    rule=rule)["eligible"]:
                n_excluded += 1
                continue
            sel = traj.select_hours(horizon)
            if len(sel) < 1:
                continue
            # 第 0 小时若不在轨迹里，则由决策时刻已观测的值补齐（与 E1 同规则）
            anchored = (0 not in set(int(h) for h in traj.hours)
                        and traj.observed_first is not None)
            anchor = (np.asarray(traj.observed_first, float)[lp.RATE_SLICE]
                      if anchored else np.zeros(lp.NUM_TYPES))
            cum_true = anchor + np.cumsum(traj.true_rates[sel], axis=0)
            cum_pred = anchor + np.cumsum(pred[sel], axis=0)
            abs_err = np.abs(cum_pred - cum_true).sum(axis=1)     # (K,) mL
            T_hours = float(len(sel) + (1 if anchored else 0))
            if T_hours <= 0:
                continue
            per_patient_abs.append(float(np.sum(abs_err * dt) / T_hours))
            hours_used.append(int(len(sel)))
            norm = _normalizer(traj)
            if norm is None:
                n_skipped_weight += 1
                continue
            per_patient_norm.append(float(np.sum(abs_err * dt) / (T_hours * norm)))

        entry: Dict[str, object] = {
            "n_patients_absolute": len(per_patient_abs),
            "n_patients_normalized": len(per_patient_norm),
            "n_patients_excluded_incomplete": n_excluded,
            "n_skipped_missing_weight": n_skipped_weight,
            "horizon_hours": int(horizon),
            "predicted_hours_min": int(min(hours_used)) if hours_used else 0,
            "predicted_hours_max": int(max(hours_used)) if hours_used else 0,
        }
        if per_patient_abs:
            entry["cete_ml"] = float(np.mean(per_patient_abs))
        else:
            entry["cete_ml"] = None
            entry["na_reason"] = "no_eligible_patient"
        if per_patient_norm:
            entry["cete_ml_per_kg"] = float(np.mean(per_patient_norm))
        else:
            entry["cete_ml_per_kg"] = None
            if "na_reason" not in entry:
                entry["na_reason"] = "weight_unavailable_for_normalisation"
        out["horizons"][int(horizon)] = entry
        out["available"] = out["available"] or bool(per_patient_abs)
    return out
