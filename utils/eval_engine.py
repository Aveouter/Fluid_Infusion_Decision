"""统一评估引擎（issue #81，Gate A）。

职责边界
--------
本模块**只做三件事**：

1. **前置校验**：标签不变量、小时有序性、口径/模式词表、协议哈希；
2. **命名**：把各终点算出的数值装成带完整身份的结果行（§4 的 18 字段规范键）；
3. **一致性检查**：比较不同方法之前的门禁（协议、估计对象、评估模式、患者集合）。

它**不含任何指标算术**：MAE、F1、累计量、匹配都在 ``utils/endpoints.py``、
``utils/changepoints.py``、``utils/clinical_endpoints.py`` 里，
``evaluate_v2.py`` 也不含算术（有测试用 AST 检查这一点）。
这样"同一个数字到底怎么算的"只有一个答案。

引擎不知道预测来自哪个模型
--------------------------
:class:`PredictionSet` 只携带**已解码**的速率/标签/概率与两个身份字段
（``rate_definition``、``eval_mode``）。TCN、GRU、Transformer、ZILN 与持久性
基线在引擎眼里没有区别——这正是"同一批真值与预测输入，无论由哪个模型产生，
都能确定性地算出相同定义、相同分母、可追溯的结果"的实现方式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from utils import changepoints as CP
from utils import clinical_endpoints as CL
from utils import endpoints as EP
from utils import eval_protocol as PROTO
from utils import label_protocol as lp
from utils import metrics as M
from utils.metrics import MetricSpec, MetricsExporter, register
from utils.trajectories import (PatientTrajectory, PredictionMatrix,
                                assemble_trajectories, audit_trajectories)


# --------------------------------------------------------------------------
# 输入
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictionSet:
    """一次评估的全部输入。**预测值必须已经解码。**"""

    patient_ids: np.ndarray                 # (N,) str
    time_index: np.ndarray                  # (N,) int，自零时点起的小时序号
    true_bits: np.ndarray                   # (N,3)
    true_rates: np.ndarray                  # (N,3) mL/h
    pred_rates: np.ndarray                  # (N,3) mL/h，已解码
    rate_definition: str
    eval_mode: str
    pred_bits: Optional[np.ndarray] = None  # (N,3)
    pred_prob: Optional[np.ndarray] = None  # (N,3)，AUC/Brier/校准与阈值解码用
    observed_first: Optional[np.ndarray] = None   # (N,6) 或 None
    thresholds: Optional[np.ndarray] = None
    fold: str = "test"

    def validate(self) -> None:
        for name, allowed in (("rate_definition", PROTO.RATE_DEFINITIONS),
                              ("eval_mode", PROTO.EVAL_MODES)):
            got = getattr(self, name)
            if got not in allowed:
                raise ValueError(f"未知的 {name}：{got!r}；可选 {allowed}")
        if self.rate_definition == "na":
            raise ValueError("rate_definition 不得为 'na'：每个数字都必须说明口径")
        if self.eval_mode == "na":
            raise ValueError("eval_mode 不得为 'na'：教师强制与自回归不得混排")

        n = len(self.patient_ids)
        for name in ("time_index", "true_bits", "true_rates", "pred_rates"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} 的行数 {len(getattr(self, name))} 与 "
                                 f"patient_ids {n} 不一致")
        if self.rate_definition in PROTO.PROBABILISTIC_RATE_DEFINITIONS \
                and self.pred_prob is None:
            raise ValueError(
                f"口径 {self.rate_definition} 依赖分类概率，但未提供 pred_prob")


# --------------------------------------------------------------------------
# 指标注册表：名字 → 冻结定义
# --------------------------------------------------------------------------
#
# 逐条登记，使"同一个名字只能是同一种定义"。``fluid_class`` / ``horizon``
# 是多类别指标**合法变化**的维度，登记时留空（None）表示不约束；单位、聚合、
# 层级、掩码必须钉死。

def _reg(name: str, unit: str, aggregation: str, level: str, mask: str,
         *, fluid_class: Optional[str] = None, tier: str = "na",
         scope: str = "", needs_rate: bool = False) -> MetricSpec:
    return register(MetricSpec(
        name=name, unit=unit, aggregation=aggregation, level=level, mask=mask,
        fluid_class=fluid_class,
        evidence_tier=tier, interpretation_scope=scope,
        requires_rate_definition=needs_rate,
    ))


def _register_all() -> None:
    if M.registry():
        return

    # ---- 主要终点 ----
    _reg("p1_cumulative_dose_mae", "mL", "patient_mean", "patient",
         "eligible_patient", needs_rate=True)
    _reg("p1_cumulative_dose_bias", "mL", "patient_mean", "patient",
         "eligible_patient", needs_rate=True)
    _reg("p2_tolerance_abs_200ml", "ratio", "patient_share", "patient",
         "eligible_patient", needs_rate=True)
    _reg("p2_tolerance_rel_10pct", "ratio", "patient_share", "patient",
         "eligible_patient", needs_rate=True)
    _reg("p2_tolerance_joint", "ratio", "patient_share", "patient",
         "eligible_patient", needs_rate=True)
    _reg("p2_zero_dose_accuracy_abs", "ratio", "patient_share", "patient",
         "zero_dose_patient", needs_rate=True)
    _reg("p2_zero_dose_n_patients", "count", "patient_share", "patient",
         "zero_dose_patient", needs_rate=True)
    _reg("p2_rel_excluded_zero_dose", "count", "patient_share", "patient",
         "zero_dose_patient", needs_rate=True)
    _reg("p3_macro_f1", "probability", "macro", "window", "all")
    _reg("p3_present_class_macro_f1", "probability", "macro", "window", "all")
    _reg("p3_micro_f1", "probability", "micro", "window", "all")
    _reg("p3_weighted_f1", "probability", "weighted", "window", "all")
    _reg("p3_bit_accuracy", "probability", "micro", "window", "all")
    _reg("p3_f1", "probability", "per_class", "window", "all")
    _reg("p3_precision", "probability", "per_class", "window", "all")
    _reg("p3_recall", "probability", "per_class", "window", "all")
    _reg("p3_support", "count", "per_class", "window", "all")

    # ---- 速率误差 ----
    _reg("all_hours_mae", "mL/h", "patient_mean", "window", "all", needs_rate=True)
    _reg("all_hours_rmse", "mL/h", "patient_mean", "window", "all", needs_rate=True)
    _reg("positive_flow_mae", "mL/h", "patient_mean", "window", "positive_flow",
         needs_rate=True)
    _reg("positive_flow_rmse", "mL/h", "patient_mean", "window", "positive_flow",
         needs_rate=True)
    _reg("zero_flow_mae", "mL/h", "patient_mean", "window", "zero_flow",
         needs_rate=True)
    _reg("positive_flow_smape", "ratio", "patient_mean", "window",
         "positive_flow", needs_rate=True)
    _reg("target_pooled_mae", "mL/h", "target_pooled", "window", "all",
         needs_rate=True)

    # ---- 排序与校准 ----
    _reg("macro_auc", "probability", "macro", "window", "all")
    _reg("auc", "probability", "per_class", "window", "all")
    _reg("brier_mean", "probability", "macro", "window", "all")
    _reg("ece_macro", "probability", "macro", "window", "all")

    # ---- 变点 ----
    _reg("change_point_precision", "ratio", "micro", "window", "matched_events",
         needs_rate=True)
    _reg("change_point_recall", "ratio", "micro", "window", "matched_events",
         needs_rate=True)
    _reg("change_point_f1", "ratio", "micro", "window", "matched_events",
         needs_rate=True)
    _reg("change_hour_mae", "mL/h", "patient_mean", "window", "true_change_hour",
         needs_rate=True)
    _reg("no_change_hour_mae", "mL/h", "patient_mean", "window", "no_change_hour",
         needs_rate=True)
    _reg("false_change_rate", "ratio", "event_micro", "window", "no_change_hour",
         needs_rate=True)
    _reg("change_timing_mean_lag", "hours", "event_micro", "window",
         "matched_events", needs_rate=True)
    _reg("change_timing_unmatched_true", "count", "event_micro", "window",
         "matched_events", needs_rate=True)

    # ---- 时间稳定性 ----
    _reg("total_variation_abs_diff", "mL", "patient_mean", "window", "all",
         needs_rate=True)
    _reg("total_variation_ratio", "ratio", "patient_mean", "window", "all",
         needs_rate=True)
    _reg("r2_positive_flow", "ratio", "patient_mean", "window", "positive_flow",
         needs_rate=True)

    # ---- 医学层 ----
    _reg("c1_crystal_exposure", "mL/kg/%TBSA", "patient_mean", "patient",
         "eligible_patient", fluid_class="crystal", tier="derived",
         scope="观察到的暴露强度复现；不是最优处方或疗效证据", needs_rate=True)
    _reg("c1_crystal_exposure_pred", "mL/kg/%TBSA", "patient_mean", "patient",
         "eligible_patient", fluid_class="crystal", tier="derived",
         scope="模型条件性预测的暴露强度；与真实值并列，差值本身不是优劣判断",
         needs_rate=True)
    _reg("c1_crystal_exposure_abs_diff", "mL/kg/%TBSA", "patient_mean", "patient",
         "eligible_patient", fluid_class="crystal", tier="derived",
         scope="|预测−真实| 的暴露强度差；不是「是否更合适」的度量", needs_rate=True)
    _reg("c1_guidance_reference", "mL/kg/%TBSA", "patient_mean", "patient",
         "eligible_patient", fluid_class="crystal", tier="guidance_only",
         scope="指南起始估算建议，不是个体金标准目标，不得用作达标率分母")
    _reg("c1_missing_fraction", "ratio", "patient_mean", "patient",
         "eligible_patient", fluid_class="crystal", tier="derived")
    _reg("c2_component_sum_mae", "mL", "patient_mean", "patient",
         "eligible_patient", tier="derived", needs_rate=True)
    _reg("c2_cancellation_gap", "mL", "patient_mean", "patient",
         "eligible_patient", tier="derived", needs_rate=True)
    _reg("n1_direction_concordance", "ratio", "event_macro", "window",
         "true_change_hour", tier="proposed_exploratory",
         scope="是否更好地预测医生在该分层下的调速动作；不是模型响应了休克",
         needs_rate=True)
    _reg("n1_false_change_rate", "ratio", "event_micro", "window",
         "no_change_hour", tier="proposed_exploratory", needs_rate=True)
    _reg("n1_change_hour_mae", "mL/h", "event_macro", "window",
         "true_change_hour", tier="proposed_exploratory", needs_rate=True)
    _reg("ps_tc_direction_concordance", "ratio", "event_macro", "window",
         "true_change_hour", tier="not_assessable",
         scope="生理状态分层不可得，本终点不可估计")
    _reg("n2_cete_ml", "mL", "patient_mean", "patient", "eligible_patient",
         tier="proposed_exploratory",
         scope="条件预测轨迹的忠实度；不是自主闭环长期暴露", needs_rate=True)
    _reg("n2_cete_ml_per_kg", "mL/kg", "patient_mean", "patient",
         "eligible_patient", tier="proposed_exploratory", needs_rate=True)

    # ---- 数据审计（计划登记的 NA 原因类也走这里） ----
    _reg("audit_n_patients", "count", "patient_mean", "patient", "all")
    _reg("audit_n_patient_hours", "count", "patient_mean", "patient", "all")
    _reg("audit_n_targets", "count", "patient_mean", "patient", "all")
    _reg("audit_horizon_complete", "count", "patient_mean", "patient",
         "eligible_patient")
    _reg("audit_zero_flow_hours", "count", "patient_mean", "patient", "zero_flow")
    _reg("audit_zero_dose_patients", "count", "patient_mean", "patient",
         "zero_dose_patient")
    _reg("subgroup_positive_flow_mae", "mL/h", "patient_mean", "patient",
         "positive_flow", needs_rate=True)
    _reg("audit_population_delta_patients", "count", "patient_mean", "patient",
         "eligible_patient")
    _reg("audit_population_delta_hours", "count", "patient_mean", "patient", "all")


_register_all()


# --------------------------------------------------------------------------
# 前置校验与一致性门禁
# --------------------------------------------------------------------------

def assert_comparable(records: Sequence[M.MetricRecord], *,
                      field_name: str = "estimand") -> None:
    """比较不同方法前：不得跨 estimand 或评eval_mode 汇总。

    把两个不同估计对象（例如条件累计量与自主 rollout）放进同一列比较，
    是这个项目已经犯过一次的错。这里做成机器检查。
    """
    values = {getattr(r, field_name) for r in records}
    if len(values) > 1:
        raise AssertionError(
            f"同一张表里出现了多个 {field_name}：{sorted(values)}；"
            "不同估计对象/评估模式不得汇总或配对比较")
    hashes = {r.protocol_hash for r in records}
    if len(hashes) > 1:
        raise AssertionError(f"同一张表里出现多个协议哈希：{sorted(hashes)}")


def paired_patients(a: Sequence[str], b: Sequence[str]) -> List[str]:
    """两个方法共有的患者（配对 bootstrap 的前提）。"""
    return sorted(set(map(str, a)) & set(map(str, b)))


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------

def evaluate(pred: PredictionSet, *,
             model_id: str,
             run_id: str,
             split_or_outer_fold: str,
             train_seed: Optional[int] = None,
             quality_mask: Optional[np.ndarray] = None,
             horizons: Sequence[int] = None,
             data_fingerprint: str = "",
             quality_track: str = "main",
             base_by_patient: Optional[Dict[str, Sequence[float]]] = None,
             note: str = "") -> MetricsExporter:
    """跑完全部终点，返回带完整身份的导出器。

    ``quality_mask``（``(N,)`` bool）只把某些**小时**标记为不合格，
    **不增删患者**——双轨敏感性分析的风险正是"换了掩码就悄悄换了一批人"。
    """
    pred.validate()
    horizons = tuple(horizons) if horizons else PROTO.HORIZONS
    for h in horizons:
        if int(h) not in PROTO.HORIZONS:
            raise ValueError(f"时域 {h} 不在冻结的 {PROTO.HORIZONS} 中")

    trajs = assemble_trajectories(
        pred.patient_ids, pred.time_index, pred.true_bits, pred.true_rates,
        predictions={pred.rate_definition: PredictionMatrix(
            rates=pred.pred_rates, bits=pred.pred_bits, prob=pred.pred_prob)},
        fold=pred.fold,
        observed_first=pred.observed_first,
    )
    if quality_mask is not None:
        qm = np.asarray(quality_mask, dtype=bool)
        if qm.shape[0] != len(pred.patient_ids):
            raise ValueError(f"quality_mask 长度 {qm.shape[0]} 与行数 "
                             f"{len(pred.patient_ids)} 不一致")
        _attach_quality_mask(trajs, pred, qm)
    if base_by_patient:
        from utils.trajectories import attach_baseline
        attach_baseline(trajs, base_by_patient)

    ex = MetricsExporter(split_or_outer_fold, seed=train_seed,
                         model_id=model_id, run_id=run_id,
                         quality_track=quality_track,
                         data_fingerprint=data_fingerprint)
    # 完整性规则由**是否给了质量掩码**决定：
    # - 主轨：小时都在即可（all_hours_present）
    # - 敏感性轨：而且这些小时必须全部通过掩码（all_hours_eligible）
    # 两轨用不同规则，敏感性分析才会真的改变分母；否则它只是把同一批数字
    # 换个标题——这比不做敏感性分析更危险。
    rule = ("all_hours_eligible" if quality_mask is not None
            else "all_hours_present")

    # 注意：``quality_track`` 是导出器级别的身份（构造时给定），不进逐行 kwargs。
    ctx = dict(eval_mode=pred.eval_mode, rate_definition=pred.rate_definition,
               run_note=note)

    # 不把全局掩码传进去：每条轨迹自带按患者切好的那一份，
    # 传全局的会因长度不符而报错（或更糟：被广播成错误的结果）。
    audit = audit_trajectories(trajs, horizons=horizons, rule=rule)
    _emit_audit(ex, trajs, audit, horizons, ctx, rule=rule)
    if not trajs:
        ex.add_na("p1_cumulative_dose_mae", na_reason="no_patients",
                  unit="mL", aggregation="patient_mean", level="patient",
                  mask="eligible_patient", **_row_ctx(ctx))
        return ex

    for h in horizons:
        _emit_cumulative(ex, trajs, pred, int(h), ctx, rule=rule)
        _emit_tolerance(ex, trajs, pred, int(h), ctx, rule=rule)
    _emit_classification(ex, trajs, pred, ctx)
    _emit_rate_errors(ex, trajs, pred, ctx)
    _emit_change_points(ex, trajs, pred, ctx)
    _emit_stability(ex, trajs, pred, ctx)
    _emit_subgroups(ex, trajs, pred, ctx)
    _emit_clinical(ex, trajs, pred, horizons, ctx, rule=rule)
    return ex


def _emit_subgroups(ex, trajs, pred, ctx) -> None:
    """按临床亚组报告正流 MAE（sex / age / TBSA / 吸入性损伤）。

    亚组一律**先按患者聚合**再统计，且随行报出 ``n_patients`` 与小样本标记：
    样本量过小时只作描述，**不作优越性主张**。没有基线特征时不产出任何行——
    宁可不报，也不猜分组。
    """
    from utils import subgroups as SG

    if trajs and trajs[0].base_row is None:
        # 没有基线就分不了组。**不猜**，但也不静默少几行——说明原因。
        ex.add_na("subgroup_positive_flow_mae",
                  na_reason="baseline_features_not_available",
                  unit="mL/h", aggregation="patient_mean", level="patient",
                  mask="positive_flow", n_patients=len(trajs), **_row_ctx(ctx))
        return

    usable = [t for t in trajs if t.pred.get(pred.rate_definition) is not None
              and t.base_row is not None]
    if not usable:
        return

    # 误差的**算术**在 utils/subgroups.py 里，这里只做装配：引擎自己算 MAE
    # 就会长出第二条计算路径，而那正是"同名指标多种算法"复发的方式。
    values, pids = SG.per_patient_error_values(
        np.concatenate([t.pred[pred.rate_definition].rates for t in usable], axis=0),
        np.concatenate([t.true_rates for t in usable], axis=0),
        # 该接口按**行**接收编号（一行一个小时），不是每位患者一个
        [t.patient_id for t in usable for _ in range(t.n_hours)],
        mask=np.concatenate([t.true_rates > PROTO.EPS_RATE_ML_PER_H
                             for t in usable], axis=0))
    if not pids:
        return

    by_pid = {t.patient_id: t.base_row for t in usable}
    rows_ref = [by_pid[p] for p in pids]
    labels = SG.subgroup_labels(rows_ref[0])
    for dim in labels:
        dim_labels = [SG.subgroup_labels(b)[dim] for b in rows_ref]
        for row in SG.subgroup_rows(np.asarray(values), pids, dim_labels,
                                    metric_name="subgroup_positive_flow_mae",
                                    unit="mL/h", mask="positive_flow"):
            ex.add("subgroup_positive_flow_mae", row["value"], unit="mL/h",
                   aggregation="patient_mean", level="patient",
                   mask="positive_flow", n_patients=row["n_patients"],
                   **_row_ctx(ctx, f"{dim}={row['subgroup']}；{row['note']}"
                                   if row["note"] else f"{dim}={row['subgroup']}"))


def _attach_quality_mask(trajs, pred: PredictionSet, qm: np.ndarray) -> None:
    """把逐窗掩码切片挂到各轨迹上（供 ``all_hours_eligible`` 使用）。"""
    by_patient: Dict[str, List[bool]] = {}
    for i, pid in enumerate(map(str, pred.patient_ids)):
        by_patient.setdefault(pid, []).append(bool(qm[i]))
    for traj in trajs:
        traj._quality_mask = np.asarray(by_patient.get(traj.patient_id, []), bool)


# --------------------------------------------------------------------------
# 各终点的行装配（只做命名与格式化，不含算术）
# --------------------------------------------------------------------------


def _row_ctx(ctx: dict, note: str = None) -> dict:
    """把运行级上下文摊成行级 kwargs。

    ``note`` 缺省时用运行级说明；一旦某行有自己的说明（例如"跳过了 N 个分母
    为 0 的单元"），以该行的为准——两者不能同时出现，否则 ``add()`` 会因
    关键字重复而报错。
    """
    out = {k: v for k, v in ctx.items() if k != "run_note"}
    out["note"] = ctx.get("run_note", "") if note is None else note
    return out


def _emit_audit(ex, trajs, audit, horizons, ctx, *, rule: str) -> None:
    # 审计行统一用「窗口/小时数不是样本量」这句说明，覆盖运行级 note。
    c = _row_ctx(ctx, audit.get("note", ""))
    ex.add("audit_n_patients", audit["n_patients"], unit="count",
           aggregation="patient_mean", level="patient", mask="all",
           n_patients=audit["n_patients"], **c)
    ex.add("audit_n_patient_hours", audit["n_patient_hours"], unit="count",
           aggregation="patient_mean", level="patient", mask="all",
           n_patients=audit["n_patients"],
           n_patient_hours=audit["n_patient_hours"], **c)
    ex.add("audit_n_targets", audit["n_targets"], unit="count",
           aggregation="patient_mean", level="patient", mask="all",
           n_patients=audit["n_patients"], n_targets=audit["n_targets"], **c)
    ex.add("audit_zero_flow_hours", audit["n_zero_flow_hours"], unit="count",
           aggregation="patient_mean", level="patient", mask="zero_flow",
           n_patients=audit["n_patients"], **c)
    ex.add("audit_zero_dose_patients", audit["n_zero_dose_patients"],
           unit="count", aggregation="patient_mean", level="patient",
           mask="zero_dose_patient", n_patients=audit["n_patients"], **c)
    for h in horizons:
        ex.add("audit_horizon_complete", audit["horizon_complete"].get(int(h), 0),
               unit="count", aggregation="patient_mean", level="patient",
               mask="eligible_patient", horizon_hours=int(h),
               n_patients=audit["n_patients"], **c)

    # 族群变化必须成为可见的一行：敏感性轨若换了合格规则，读者要能立刻看到
    # 各时域少了几个人，而不是自己对比两张表才发觉分母已经不同。
    if rule != "all_hours_present":
        from utils.trajectories import population_delta
        reference = audit_trajectories(trajs, horizons=horizons,
                                       rule="all_hours_present")
        delta = population_delta(reference, audit)
        # 患者总数不变**正是设计意图**（掩码只标小时、不增删患者），所以
        # 真正要报的是**各时域合格人数**少了几位——那才是分母的变化。
        for h in horizons:
            lost = delta["delta_horizon_complete"].get(int(h), 0)
            na = (f"主轨 {reference['horizon_complete'].get(int(h), 0)} 人合格，"
                  f"本轨 {audit['horizon_complete'].get(int(h), 0)} 人")
            ex.add("audit_population_delta_patients", lost, unit="count",
                   aggregation="patient_mean", level="patient",
                   mask="eligible_patient", horizon_hours=int(h),
                   n_patients=audit["n_patients"],
                   **_row_ctx(ctx, na))
        # 小时维度的变化用**合格小时数**衡量（总小时数不会变）
        n_eligible = int(sum(
            int(np.asarray(t._quality_mask, bool).sum()) for t in trajs
            if getattr(t, "_quality_mask", None) is not None))
        ex.add("audit_population_delta_hours",
               audit["n_patient_hours"] - n_eligible, unit="count",
               aggregation="patient_mean", level="patient", mask="all",
               n_patients=audit["n_patients"],
               **_row_ctx(ctx, f"总小时 {audit['n_patient_hours']}，"
                               f"其中合格 {n_eligible}"))


def _emit_cumulative(ex, trajs, pred, horizon, ctx, *, rule: str) -> None:
    out = EP.cumulative_summary(trajs, pred.rate_definition, horizon=horizon,
                                origin="hour_0_anchored", rule=rule)
    est = out["estimand"]
    if not out["available"]:
        ex.add_na("p1_cumulative_dose_mae",
                  na_reason=",".join(out["exclusion_reasons"]) or "no_eligible_patient",
                  unit="mL", aggregation="patient_mean", level="patient",
                  mask="eligible_patient", horizon_hours=horizon,
                  horizon_origin="hour_0_anchored", estimand=est,
                  n_patients=0, **_row_ctx(ctx))
        return
    for fluid, stat in out["per_class"].items():
        ex.add("p1_cumulative_dose_mae", stat["mae_ml"], unit="mL",
               aggregation="patient_mean", level="patient",
               mask="eligible_patient", fluid_class=fluid,
               horizon_hours=horizon, horizon_origin="hour_0_anchored",
               estimand=est, n_patients=stat["n_patients"],
               n_targets=stat["n_targets"],
               n_patient_hours=out["n_predicted_hours_total"], **_row_ctx(ctx))
        ex.add("p1_cumulative_dose_bias", stat["bias_ml"], unit="mL",
               aggregation="patient_mean", level="patient",
               mask="eligible_patient", fluid_class=fluid,
               horizon_hours=horizon, horizon_origin="hour_0_anchored",
               estimand=est, n_patients=stat["n_patients"],
               n_targets=stat["n_targets"], **_row_ctx(ctx))
    if out["n_patients_excluded"]:
        ex.add_na("p1_cumulative_dose_mae_excluded",
                  na_reason=",".join(out["exclusion_reasons"]),
                  unit="count", aggregation="patient_mean", level="patient",
                  mask="incomplete", horizon_hours=horizon, estimand=est,
                  n_patients=out["n_patients_excluded"], **_row_ctx(ctx))


def _emit_tolerance(ex, trajs, pred, horizon, ctx, *, rule: str) -> None:
    out = EP.tolerance_summary(trajs, pred.rate_definition, horizon=horizon,
                               origin="hour_0_anchored", rule=rule)
    est = "conditional_cumulative_volume"
    if not out["available"]:
        ex.add_na("p2_tolerance_abs_200ml", na_reason="no_eligible_patient",
                  unit="ratio", aggregation="patient_share", level="patient",
                  mask="eligible_patient", horizon_hours=horizon, estimand=est,
                  n_patients=0, **_row_ctx(ctx))
        return
    common = dict(unit="ratio", aggregation="patient_share", level="patient",
                  mask="eligible_patient", horizon_hours=horizon,
                  horizon_origin="hour_0_anchored", estimand=est,
                  n_patients=out["abs"]["n_patients"], **_row_ctx(ctx))
    ex.add("p2_tolerance_abs_200ml", out["abs"]["value"], **common)
    ex.add("p2_tolerance_rel_10pct", out["rel"]["value"], **common)
    ex.add("p2_tolerance_joint", out["joint"]["value"], **common)
    ex.add("p2_zero_dose_n_patients", out["zero_dose"]["n_patients"],
           unit="count", aggregation="patient_share", level="patient",
           mask="zero_dose_patient", horizon_hours=horizon, estimand=est, **_row_ctx(ctx))
    ex.add("p2_rel_excluded_zero_dose", out["rel"]["n_excluded_zero_dose"],
           unit="count", aggregation="patient_share", level="patient",
           mask="zero_dose_patient", horizon_hours=horizon, estimand=est, **_row_ctx(ctx))
    zd = out["zero_dose"]["accuracy_abs"]
    if zd == zd:                                   # 非 NaN
        ex.add("p2_zero_dose_accuracy_abs", zd, unit="ratio",
               aggregation="patient_share", level="patient",
               mask="zero_dose_patient", horizon_hours=horizon, estimand=est,
               n_patients=out["zero_dose"]["n_patients"], **_row_ctx(ctx))
    else:
        ex.add_na("p2_zero_dose_accuracy_abs", na_reason="no_zero_dose_patients",
                  unit="ratio", aggregation="patient_share", level="patient",
                  mask="zero_dose_patient", horizon_hours=horizon, estimand=est,
                  n_patients=0, **_row_ctx(ctx))


def _emit_classification(ex, trajs, pred, ctx) -> None:
    bits_true = np.concatenate([t.true_bits for t in trajs], axis=0)
    mat = trajs[0].pred.get(pred.rate_definition)
    if mat is None or mat.bits is None:
        ex.add_na("p3_macro_f1", na_reason="pred_bits_not_provided",
                  unit="probability", aggregation="macro", level="window",
                  mask="all", **_row_ctx(ctx))
        return
    bits_pred = np.concatenate([t.pred[pred.rate_definition].bits for t in trajs],
                               axis=0)
    prf = EP.multilabel_prf(bits_true.astype(int), bits_pred.astype(int))
    # 阈值逐类不同，**不做平均**：一个"平均阈值"不对应任何一类，反而会让读者
    # 以为存在单一阈值。宏平均行留空并附逐类清单；逐类行各带自己的阈值。
    thr_list = (list(map(float, pred.thresholds))
                if pred.thresholds is not None else None)
    thr_note = ("逐类阈值 " + "/".join(f"{t:.3f}" for t in thr_list)
                if thr_list else "无概率头：未使用分类阈值")
    thr = None
    n_pat = len(trajs)
    ex.add("p3_macro_f1", prf["macro_f1"], unit="probability",
           aggregation="macro", level="window", mask="all", threshold=thr,
           n_patients=n_pat, n_targets=prf["n_targets"], **_row_ctx(ctx))
    ex.add("p3_present_class_macro_f1", prf["present_class_macro_f1"],
           unit="probability", aggregation="macro", level="window", mask="all",
           threshold=thr, n_patients=n_pat, n_targets=prf["n_targets"],
           **_row_ctx(ctx, f"敏感性：只在 {prf['n_classes_in_present_macro']} 个有正例的类上取均值"))
    ex.add("p3_micro_f1", prf["micro_f1"], unit="probability",
           aggregation="micro", level="window", mask="all", threshold=thr,
           n_patients=n_pat, n_targets=prf["n_targets"], **_row_ctx(ctx))
    ex.add("p3_weighted_f1", prf["weighted_f1"], unit="probability",
           aggregation="weighted", level="window", mask="all", threshold=thr,
           n_patients=n_pat, n_targets=prf["n_targets"], **_row_ctx(ctx))
    ex.add("p3_bit_accuracy", prf["bit_accuracy"], unit="probability",
           aggregation="micro", level="window", mask="all", threshold=thr,
           n_patients=n_pat, n_targets=prf["n_targets"], **_row_ctx(ctx))
    for k, (name, stat) in enumerate(prf["per_class"].items()):
        thr_k = thr_list[k] if thr_list else None
        ex.add("p3_f1", stat["f1"], unit="probability", aggregation="per_class",
               level="window", mask="all", fluid_class=name, threshold=thr_k,
               n_patients=n_pat, n_targets=prf["n_targets"], **_row_ctx(ctx))
        ex.add("p3_precision", stat["precision"], unit="probability",
               aggregation="per_class", level="window", mask="all",
               fluid_class=name, threshold=thr_k, n_patients=n_pat,
               **_row_ctx(ctx))
        ex.add("p3_recall", stat["recall"], unit="probability",
               aggregation="per_class", level="window", mask="all",
               fluid_class=name, threshold=thr_k, n_patients=n_pat,
               **_row_ctx(ctx))
        ex.add("p3_support", stat["support"], unit="count",
               aggregation="per_class", level="window", mask="all",
               fluid_class=name, n_patients=n_pat, **_row_ctx(ctx))

    if pred.pred_prob is not None:
        prob = np.concatenate([t.pred[pred.rate_definition].prob for t in trajs],
                              axis=0)
        rank = EP.multilabel_ranking(bits_true.astype(int), prob)
        if rank["auc_macro"] == rank["auc_macro"]:
            ex.add("macro_auc", rank["auc_macro"], unit="probability",
                   aggregation="macro", level="window", mask="all", threshold=thr,
                   n_patients=n_pat, **_row_ctx(ctx, rank["auc_na_reason"]))
        else:
            ex.add_na("macro_auc", na_reason="no_class_with_two_label_values",
                      unit="probability", aggregation="macro", level="window",
                      mask="all", n_patients=n_pat, **_row_ctx(ctx))
        for name, auc in zip(lp.LIQUID_TYPES, rank["auc_per_class"]):
            if auc is None:
                ex.add_na("auc", na_reason="single_class_labels_in_split",
                          unit="probability", aggregation="per_class",
                          level="window", mask="all", fluid_class=name,
                          n_patients=n_pat, **_row_ctx(ctx))
            else:
                ex.add("auc", auc, unit="probability", aggregation="per_class",
                       level="window", mask="all", fluid_class=name,
                       n_patients=n_pat, **_row_ctx(ctx))
        ex.add("brier_mean", rank["brier_mean"], unit="probability",
               aggregation="macro", level="window", mask="all",
               n_patients=n_pat, **_row_ctx(ctx))
        ex.add("ece_macro", rank["ece_macro"], unit="probability",
               aggregation="macro", level="window", mask="all",
               n_patients=n_pat, **_row_ctx(ctx, f"{rank['n_bins']} 等宽分箱"))
    else:
        # 无概率头的基线（持久性、移动平均）：这些终点**不可计算**。
        # 必须显式成行报 NA + 原因，而不是悄悄少几行——少了行读者会以为
        # 忘了报，NA 才说明"这个量对这类方法不存在"。
        reason = "no_probability_head_for_this_method"
        for name in ("macro_auc", "brier_mean", "ece_macro"):
            ex.add_na(name, na_reason=reason, unit="probability",
                      aggregation="macro", level="window", mask="all",
                      n_patients=n_pat, **_row_ctx(ctx))
        for fluid in lp.LIQUID_TYPES:
            ex.add_na("auc", na_reason=reason, unit="probability",
                      aggregation="per_class", level="window", mask="all",
                      fluid_class=fluid, n_patients=n_pat, **_row_ctx(ctx))


def _emit_rate_errors(ex, trajs, pred, ctx) -> None:
    n_pat = len(trajs)
    for fluid in ("total",) + tuple(lp.LIQUID_TYPES):
        for kind, name in (("all", "all_hours_mae"), ("positive_flow", "positive_flow_mae"),
                           ("zero_flow", "zero_flow_mae")):
            out = EP.rate_error_summary(trajs, pred.rate_definition, mask=kind,
                                        fluid_class=fluid)
            if not out["available"]:
                ex.add_na(name, na_reason="no_target_matches_mask", unit="mL/h",
                          aggregation="patient_mean", level="window",
                          mask=kind, fluid_class=fluid, n_patients=0, **_row_ctx(ctx))
                continue
            ex.add(name, out["mae_ml_per_h"], unit="mL/h",
                   aggregation="patient_mean", level="window", mask=kind,
                   fluid_class=fluid, n_patients=out["n_patients"],
                   n_targets=out["n_targets"],
                   n_patient_hours=out["n_patient_hours"], **_row_ctx(ctx))
        r = EP.rate_error_summary(trajs, pred.rate_definition, mask="all",
                                  fluid_class=fluid)
        if r["available"]:
            ex.add("all_hours_rmse", r["rmse_ml_per_h"], unit="mL/h",
                   aggregation="patient_mean", level="window", mask="all",
                   fluid_class=fluid, n_patients=r["n_patients"],
                   n_targets=r["n_targets"], **_row_ctx(ctx))
            ex.add("target_pooled_mae", r["target_pooled"]["mae_ml_per_h"],
                   unit="mL/h", aggregation="target_pooled", level="window",
                   mask="all", fluid_class=fluid, n_patients=r["n_patients"],
                   n_targets=r["n_targets"],
                   **_row_ctx(ctx, "辅助口径：把 (小时, 类别) 当独立单元，患者小时数不等时会偏向小时多者"))

    sm = EP.smape_summary(trajs, pred.rate_definition, mask="positive_flow")
    if sm["available"]:
        ex.add("positive_flow_smape", sm["smape_percent"], unit="ratio",
               aggregation="patient_mean", level="window", mask="positive_flow",
               n_patients=sm["n_patients"], n_targets=sm["n_targets"],
               **_row_ctx(ctx, f"跳过分母为 0 的 {sm['n_skipped']} 个单元"))
    else:
        ex.add_na("positive_flow_smape", na_reason="no_positive_flow_target",
                  unit="ratio", aggregation="patient_mean", level="window",
                  mask="positive_flow", n_patients=0, **_row_ctx(ctx))


def _emit_change_points(ex, trajs, pred, ctx) -> None:
    true_cp, pred_cp = [], []
    for t in trajs:
        mat = t.pred.get(pred.rate_definition)
        if mat is None:
            continue
        true_cp.append(CP.detect_changes(t.true_rates))
        pred_cp.append(CP.detect_changes(mat.rates))
    if not true_cp:
        return
    tc = np.concatenate(true_cp, axis=0)
    pc = np.concatenate(pred_cp, axis=0)
    tol = PROTO.MATCH_TOLERANCE_PRIMARY

    def _emit_ratio(name, value, na_reason, note, n_targets, n_patients=None):
        """有限值 → 数值行；None/NaN → NA + 原因。

        变化点指标天然会遇到分母为 0（没有真实变点、或预测一个变点都没报），
        那时指标**无定义**，必须写 NA 而不是 0——0 会被读成"全错"。
        """
        kw = dict(unit="ratio", aggregation="micro", level="window",
                  mask="matched_events", n_patients=(n_patients or len(trajs)),
                  **_row_ctx(ctx, note))
        if value is None or not np.isfinite(value):
            ex.add_na(name, na_reason=na_reason, n_targets=n_targets, **kw)
        else:
            ex.add(name, float(value), n_targets=n_targets, **kw)

    for tolerance in (0, tol):
        prf = CP.event_prf(tc, pc, tolerance=tolerance)
        suffix = "_exact" if tolerance == 0 else ""
        note = "" if tolerance else "容差 0 小时（逐小时精确；主报告用 ±1h）"
        n_true, n_pred = prf["micro"]["n_true"], prf["micro"]["n_pred"]
        _emit_ratio(f"change_point_precision{suffix}", prf["micro"]["precision"],
                    "no_predicted_change_events", note, n_pred)
        _emit_ratio(f"change_point_recall{suffix}", prf["micro"]["recall"],
                    "no_true_change_events", note, n_true)
        _emit_ratio(f"change_point_f1{suffix}", prf["micro"]["f1"],
                    "no_true_or_predicted_change_events", note, n_true)

    timing = CP.change_timing_v2(tc, pc, tolerance=tol)
    if timing["n_matched"]:
        ex.add("change_timing_mean_lag", timing["mean_lag_hours"], unit="hours",
               aggregation="event_micro", level="window", mask="matched_events",
               n_patients=len(trajs), n_targets=timing["n_matched"],
               **_row_ctx(ctx, "正数 = 预测滞后；只统计一对一匹配上的事件"))
    else:
        ex.add_na("change_timing_mean_lag", na_reason="no_matched_change_events",
                  unit="hours", aggregation="event_micro", level="window",
                  mask="matched_events", n_patients=len(trajs), **_row_ctx(ctx))
    ex.add("change_timing_unmatched_true", timing["n_unmatched_true"],
           unit="count", aggregation="event_micro", level="window",
           mask="matched_events", n_patients=len(trajs),
           **_row_ctx(ctx, "未匹配的真实变点不按 0 计入时延均值"))

    fr = CP.false_change_rate(tc, pc, tolerance=tol)
    if np.isfinite(fr["false_change_rate"]):
        ex.add("false_change_rate", fr["false_change_rate"], unit="ratio",
               aggregation="event_micro", level="window", mask="no_change_hour",
               n_patients=len(trajs), n_targets=fr["n_opportunities"],
               **_row_ctx(ctx, "必须与 change-point recall 配对读，否则偏向'到处乱变'的预测"))
    else:
        ex.add_na("false_change_rate", na_reason="no_non_change_cells",
                  unit="ratio", aggregation="event_micro", level="window",
                  mask="no_change_hour", n_patients=len(trajs), **_row_ctx(ctx))

    strat = CP.stratified_metrics(np.concatenate([t.pred[pred.rate_definition].rates for t in trajs]),
                                  np.concatenate([t.true_rates for t in trajs]), tc)
    for key, mask_name in (("changed", "true_change_hour"),
                           ("stable", "no_change_hour")):
        st = strat.get(key, {})
        name = "change_hour_mae" if key == "changed" else "no_change_hour_mae"
        if st.get("n_cells"):
            ex.add(name, st["mae"], unit="mL/h", aggregation="patient_mean",
                   level="window", mask=mask_name, n_patients=len(trajs),
                   n_targets=st["n_cells"], **_row_ctx(ctx))


def _emit_stability(ex, trajs, pred, ctx) -> None:
    tv = EP.total_variation_summary(trajs, pred.rate_definition)
    if tv["available"]:
        ex.add("total_variation_abs_diff", tv["tv_abs_diff_ml"], unit="mL",
               aggregation="patient_mean", level="window", mask="all",
               n_patients=tv["n_patients"], **_row_ctx(ctx))
        if tv["tv_ratio"] == tv["tv_ratio"]:
            ex.add("total_variation_ratio", tv["tv_ratio"], unit="ratio",
                   aggregation="patient_mean", level="window", mask="all",
                   n_patients=tv["n_patients_with_ratio"],
                   **_row_ctx(ctx, "必须与 change-point recall 配对读；<1 可能是'不敢调整'"))
    # R² 只在正流上，且必须带分母说明
    pool_t, pool_p = [], []
    for t in trajs:
        mat = t.pred.get(pred.rate_definition)
        if mat is None:
            continue
        m = t.true_rates > PROTO.EPS_RATE_ML_PER_H
        if m.any():
            pool_t.append(t.true_rates[m])
            pool_p.append(mat.rates[m])
    if pool_t:
        r2 = M.r2_with_denominator(np.concatenate(pool_t), np.concatenate(pool_p))
        if r2["r2"] == r2["r2"]:
            ex.add("r2_positive_flow", r2["r2"], unit="ratio",
                   aggregation="patient_mean", level="window",
                   mask="positive_flow", n_patients=len(trajs),
                   n_targets=r2["n"], **_row_ctx(ctx, r2["denominator"]))


def _emit_clinical(ex, trajs, pred, horizons, ctx, *, rule: str) -> None:
    c1 = CL.early_resuscitation_exposure(trajs, pred.rate_definition, horizon=24)
    if c1["available"]:
        ex.add("c1_crystal_exposure", c1["true_mean"], unit="mL/kg/%TBSA",
               aggregation="patient_mean", level="patient",
               mask="eligible_patient", fluid_class="crystal",
               horizon_hours=24, evidence_tier=CL.TIER_MUST,
               clinical_definition_source=c1["clinical_definition_source"],
               clinical_applicability=c1["clinical_applicability"],
               interpretation_scope=c1["interpretation_scope"],
               observation_vs_imputation="observed",
               missing_fraction=c1["missing_fraction"],
               denominator_patients=c1["n_patients_eligible"],
               n_patients=c1["n_patients_eligible"], **_row_ctx(ctx, "真实临床暴露值"))
        ex.add("c1_crystal_exposure_pred", c1["pred_mean"], unit="mL/kg/%TBSA",
               aggregation="patient_mean", level="patient",
               mask="eligible_patient", fluid_class="crystal",
               horizon_hours=24, evidence_tier=CL.TIER_MUST,
               clinical_definition_source=c1["clinical_definition_source"],
               clinical_applicability=c1["clinical_applicability"],
               interpretation_scope=c1["interpretation_scope"],
               observation_vs_imputation="observed",
               denominator_patients=c1["n_patients_eligible"],
               n_patients=c1["n_patients_eligible"],
               **_row_ctx(ctx, "模型条件性预测值；与真实值并列报告"))
        ex.add("c1_crystal_exposure_abs_diff", c1["mean_abs_diff"],
               unit="mL/kg/%TBSA", aggregation="patient_mean", level="patient",
               mask="eligible_patient", fluid_class="crystal",
               horizon_hours=24, evidence_tier=CL.TIER_MUST,
               clinical_applicability=c1["clinical_applicability"],
               interpretation_scope=c1["interpretation_scope"],
               n_patients=c1["n_patients_eligible"],
               **_row_ctx(ctx, "逐患者 |预测−真实| 的平均；差值不是临床优劣判断"))
        ex.add("c1_guidance_reference", c1["guidance"]["value_ml_per_kg_tbsa"],
               unit="mL/kg/%TBSA", aggregation="patient_mean", level="patient",
               mask="eligible_patient", fluid_class="crystal",
               horizon_hours=24, evidence_tier=CL.TIER_GUIDANCE,
               interpretation_scope=c1["guidance"]["note"],
               clinical_applicability=c1["clinical_applicability"],
               n_patients=c1["n_patients_eligible"], **_row_ctx(ctx))
        ex.add("c1_missing_fraction", c1["missing_fraction"], unit="ratio",
               aggregation="patient_mean", level="patient",
               mask="eligible_patient", fluid_class="crystal",
               evidence_tier=CL.TIER_MUST, n_patients=c1["n_patients_total"],
               **_row_ctx(ctx, f"不可计算 {c1['n_patients_unavailable']} 人：" f"{c1['unavailable_reasons']}"))
    else:
        ex.add_na("c1_crystal_exposure", na_reason=c1["na_reason"],
                  unit="mL/kg/%TBSA", aggregation="patient_mean", level="patient",
                  mask="eligible_patient", fluid_class="crystal",
                  horizon_hours=24, evidence_tier=CL.TIER_MUST,
                  observation_vs_imputation="unavailable",
                  missing_fraction=c1["missing_fraction"], n_patients=0, **_row_ctx(ctx))

    comp = EP.component_sum_error(trajs, pred.rate_definition, horizon=horizons[-1],
                               rule=rule)
    if comp["available"]:
        ex.add("c2_component_sum_mae", comp["component_sum_mae_ml"], unit="mL",
               aggregation="patient_mean", level="patient",
               mask="eligible_patient", horizon_hours=int(horizons[-1]),
               evidence_tier="derived", n_patients=comp["n_patients"],
               **_row_ctx(ctx, comp["note"]))
        ex.add("c2_cancellation_gap", comp["cancellation_gap_ml"], unit="mL",
               aggregation="patient_mean", level="patient",
               mask="eligible_patient", horizon_hours=int(horizons[-1]),
               evidence_tier="derived", n_patients=comp["n_patients"],
               **_row_ctx(ctx, ">0 说明存在通道间正负抵消（总量对、组成错）"))

    n1 = CL.titration_concordance(trajs, pred.rate_definition)
    if n1["available"]:
        ex.add("n1_direction_concordance", n1["overall"]["direction_concordance"],
               unit="ratio", aggregation="event_macro", level="window",
               mask="true_change_hour", evidence_tier=CL.TIER_PROPOSED,
               clinical_applicability="not_clinically_validated",
               expert_review_status="not_reviewed",
               interpretation_scope=n1["interpretation_scope"],
               n_patients=n1["n_patients"], n_targets=n1["n_change_events"], **_row_ctx(ctx))
        ex.add("n1_change_hour_mae",
               n1["overall"]["change_hour_mae_ml_per_h"], unit="mL/h",
               aggregation="event_macro", level="window", mask="true_change_hour",
               evidence_tier=CL.TIER_PROPOSED,
               interpretation_scope=n1["interpretation_scope"],
               n_patients=n1["n_patients"], n_targets=n1["n_change_events"], **_row_ctx(ctx))
        if n1["false_change_rate"] == n1["false_change_rate"]:
            ex.add("n1_false_change_rate", n1["false_change_rate"], unit="ratio",
                   aggregation="event_micro", level="window",
                   mask="no_change_hour", evidence_tier=CL.TIER_PROPOSED,
                   n_patients=n1["n_patients"], n_targets=n1["n_non_change_cells"],
                   **_row_ctx(ctx, "必须与 N1 一致率配对读"))
    else:
        ex.add_na("n1_direction_concordance", na_reason=n1["na_reason"],
                  unit="ratio", aggregation="event_macro", level="window",
                  mask="true_change_hour", evidence_tier=CL.TIER_PROPOSED,
                  n_patients=len(trajs), **_row_ctx(ctx))

    ps = CL.ps_tc(trajs, pred.rate_definition)
    ex.add_na("ps_tc_direction_concordance", na_reason=ps["na_reason"],
              unit="ratio", aggregation="event_macro", level="window",
              mask="true_change_hour", evidence_tier=CL.TIER_NOT_ASSESSABLE,
              observation_vs_imputation="unavailable",
              clinical_applicability=ps["clinical_applicability"],
              missing_fraction=ps["missing_fraction"],
              interpretation_scope=ps["note"], n_patients=len(trajs), **_row_ctx(ctx))

    cete = CL.cumulative_exposure_trajectory_error(
        trajs, pred.rate_definition, horizons=horizons, rule=rule)
    for h in horizons:
        entry = cete["horizons"].get(int(h), {})
        if entry.get("cete_ml") is not None:
            ex.add("n2_cete_ml", entry["cete_ml"], unit="mL",
                   aggregation="patient_mean", level="patient",
                   mask="eligible_patient", horizon_hours=int(h),
                   evidence_tier=CL.TIER_PROPOSED,
                   clinical_applicability="not_clinically_validated",
                   interpretation_scope=cete["interpretation_scope"],
                   n_patients=entry["n_patients_absolute"], **_row_ctx(ctx))
        if entry.get("cete_ml_per_kg") is not None:
            ex.add("n2_cete_ml_per_kg", entry["cete_ml_per_kg"], unit="mL/kg",
                   aggregation="patient_mean", level="patient",
                   mask="eligible_patient", horizon_hours=int(h),
                   evidence_tier=CL.TIER_PROPOSED,
                   interpretation_scope=cete["interpretation_scope"],
                   n_patients=entry["n_patients_normalized"],
                   **_row_ctx(ctx, f"跳过 {entry['n_skipped_missing_weight']} 位无体重患者"))
        elif entry.get("na_reason"):
            ex.add_na("n2_cete_ml_per_kg", na_reason=entry["na_reason"],
                      unit="mL/kg", aggregation="patient_mean", level="patient",
                      mask="eligible_patient", horizon_hours=int(h),
                      evidence_tier=CL.TIER_PROPOSED,
                      missing_fraction=1.0, n_patients=0, **_row_ctx(ctx))
