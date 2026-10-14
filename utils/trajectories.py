"""窗口 → 患者轨迹的装配、完整性与审计（issue #81，Gate A）。

为什么单独成模块
----------------
评估的独立单位是**患者**，而模型产出的是**逐窗口**的预测。两者之间隔着三步
容易出错的重建：

1. 同一患者的窗口必须能拼成一条**按小时有序、无重复、无空洞**的轨迹；
2. 患者不能被拆分到多个 split/fold（否则泄漏）；
3. 时域完整性必须显式判定——30 小时的部分轨迹**不是** 48h/72h。

这三件事此前没有任何一处负责，于是"窗口数当作样本量""部分轨迹冒充满时域"
都无法被机器发现。

真实零值 / 缺失 / 结构不完整
----------------------------
按协议 v2 的要求，三者**语义不同、不得合并**：

- **真实零值**（``rate == 0`` 且该小时确实被记录）：是"医生这一小时没输"，是数据。
- **缺失**（哨兵 ``-1``，在 :func:`utils.label_protocol.normalize` 中被置 0）：
  与真实零值在标签矩阵里**已经不可区分**，本模块不假装能还原它。
- **结构不完整**（该患者根本没有第 t 个小时）：由 :func:`completeness` 判定，
  结果是"该时域不可估计"，而不是"那一小时是 0"。

本模块**不做任何插值**：缺的小时就是缺的，绝不补。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from utils import label_protocol as lp

#: 时域完整性判定规则。
COMPLETENESS_RULES = (
    "all_hours_present",     # 该患者的轨迹覆盖 0..H-1 且连续
    "all_hours_eligible",    # 同上，且这些小时全部通过质量掩码
)


@dataclass(frozen=True)
class PredictionMatrix:
    """一种速率口径下的逐窗口预测。

    ``rates`` 是**已解码**的速率（mL/h）——评估引擎不关心它是怎么来的。
    ``bits``/``prob`` 可选：前者用于分类指标，后者用于 AUC/Brier/校准。
    """

    rates: np.ndarray
    bits: Optional[np.ndarray] = None
    prob: Optional[np.ndarray] = None

    def __post_init__(self):
        r = np.asarray(self.rates, dtype=float)
        if r.ndim != 2 or r.shape[1] != lp.NUM_TYPES:
            raise ValueError(f"rates 形状应为 (N, {lp.NUM_TYPES})，实际 {r.shape}")
        object.__setattr__(self, "rates", r)
        for name in ("bits", "prob"):
            v = getattr(self, name)
            if v is None:
                continue
            v = np.asarray(v, dtype=float)
            if v.shape != r.shape:
                raise ValueError(f"{name} 形状 {v.shape} 与 rates {r.shape} 不一致")
            object.__setattr__(self, name, v)


@dataclass
class PatientTrajectory:
    """一位患者的完整评估单元。

    ``hours`` 是自零时点起的小时序号（0 基），**升序且无重复**。
    ``pred`` 以速率口径为键；每种口径的数组与 ``hours`` 逐行对齐。
    """

    patient_id: str
    fold: str
    hours: np.ndarray                 # (T,) int
    true_bits: np.ndarray             # (T,3) float
    true_rates: np.ndarray            # (T,3) float, mL/h
    row_index: np.ndarray             # (T,) int，各行在原窗口表中的下标
    pred: Dict[str, PredictionMatrix] = field(default_factory=dict)
    observed_first: Optional[np.ndarray] = None   # (6,) 决策时刻即可见的首小时标签
    weight_kg: Optional[float] = None
    tbsa_percent: Optional[float] = None
    base_row: Optional[np.ndarray] = None

    # ------------------------------------------------------------
    @property
    def n_hours(self) -> int:
        return int(len(self.hours))

    @property
    def hour_start(self) -> int:
        return int(self.hours[0]) if self.n_hours else -1

    @property
    def hour0_available(self) -> bool:
        """第 0 小时是否**在决策时刻即可知**（累计锚点 E1 的前提）。

        两种情形都算：第 0 小时本身出现在窗口表里（``hours`` 含 0），
        或它由 ``observed_first``（决策时刻的已观测标签）提供。
        都不是时，E1 的锚定口径不可用，只能退回 E3。
        """
        return bool(0 in set(int(h) for h in self.hours)) or self.observed_first is not None

    def rate_definitions(self) -> List[str]:
        return sorted(self.pred)

    def rates_for(self, rate_definition: str) -> Optional[np.ndarray]:
        m = self.pred.get(rate_definition)
        return None if m is None else m.rates

    # ------------------------------------------------------------
    def completeness(self, horizon: int, rule: str = "all_hours_present",
                     quality_mask: Optional[np.ndarray] = None) -> Dict[str, object]:
        """该患者在时域 ``horizon`` 上是否完整。

        ``horizon`` 是**记录小时数**（24/48/72）。判定要求小时序号恰好覆盖
        ``[0, horizon)``：**部分轨迹一律判为不完整**，由调用方转成
        ``NA + na_reason="insufficient_horizon"``，绝不按现有小时求和后
        冒充满时域。

        例外只有一处，且是协议明确规定的：若第 0 小时由 ``observed_first``
        提供（决策时刻即已观测，不是预测），则它不算缺失——此时要求覆盖的是
        ``[1, horizon)``，共 ``horizon - 1`` 个**预测**小时。
        """
        if rule not in COMPLETENESS_RULES:
            raise ValueError(f"rule 必须是 {COMPLETENESS_RULES} 之一，实际 {rule}")
        if horizon < 1:
            raise ValueError(f"horizon 必须 >= 1，实际 {horizon}")

        have = set(int(h) for h in self.hours)
        required = set(range(horizon))
        anchored = 0 not in have and self.hour0_available
        if anchored:
            required.discard(0)
        missing = sorted(required - have)
        present = len(required & have)

        complete = not missing
        if complete and rule == "all_hours_eligible":
            # 逐小时质量掩码与**本轨迹**的 hours 对齐。调用方可以显式传入，
            # 否则用轨迹自带的那一份（由 eval_engine 按患者切片挂上）。
            # 传长度不符的掩码直接报错，不做广播猜测。
            qm = quality_mask
            if qm is None:
                qm = getattr(self, "_quality_mask", None)
            if qm is None:
                complete = False
            else:
                qm = np.asarray(qm, dtype=bool).ravel()
                if qm.size != self.n_hours:
                    raise ValueError(
                        f"质量掩码长度 {qm.size} 与患者 {self.patient_id} 的 "
                        f"{self.n_hours} 个小时不一致；掩码必须与该轨迹对齐")
                sel = np.isin(self.hours, sorted(required))
                complete = bool(qm[sel].all()) if sel.any() else False

        return {
            "complete": bool(complete),
            "hours_present": int(present),
            "hours_required": int(len(required)),
            "missing_hours": missing,
            "hour0_anchored": bool(anchored),
            "rule": rule,
        }

    def select_hours(self, horizon: int) -> np.ndarray:
        """返回 ``[0, horizon)`` 内实际存在的行下标（**不补缺**）。"""
        return np.where(self.hours < horizon)[0]


# --------------------------------------------------------------------------
# 装配
# --------------------------------------------------------------------------

def assemble_trajectories(patient_ids: Sequence[str],
                          time_index: Sequence[int],
                          true_bits: np.ndarray,
                          true_rates: np.ndarray,
                          *,
                          predictions: Optional[Dict[str, PredictionMatrix]] = None,
                          fold: str = "test",
                          require_contiguous: bool = True,
                          observed_first: Optional[np.ndarray] = None,
                          ) -> List[PatientTrajectory]:
    """把逐窗口的数组装配成逐患者的轨迹，并逐项校验。

    ``time_index`` 是各窗口**目标小时**自零时点起的序号（0 基）。要求同一
    患者内部严格升序；``require_contiguous=True`` 时还要求相邻相差 1，
    否则说明窗口表本身有洞，应当先修数据而不是继续算指标。

    校验（不通过即抛错）：

    - 三个数组长度一致；
    - 每位患者的行按小时严格升序、无重复；
    - 标签满足 ``bit = 1 ⟺ rate > 0``；
    - 预测数组与真值形状一致。
    """
    pids = [str(p) for p in patient_ids]
    hours_all = np.asarray(time_index, dtype=int)
    bits = np.asarray(true_bits, dtype=float)
    rates = np.asarray(true_rates, dtype=float)
    n = len(pids)

    if not (len(hours_all) == len(bits) == len(rates) == n):
        raise ValueError(
            f"长度不一致：patient_ids={n} time_index={len(hours_all)} "
            f"true_bits={len(bits)} true_rates={len(rates)}")
    for name, arr in (("true_bits", bits), ("true_rates", rates)):
        if arr.ndim != 2 or arr.shape[1] != lp.NUM_TYPES:
            raise ValueError(f"{name} 形状应为 (N, {lp.NUM_TYPES})，实际 {arr.shape}")

    # 核心不变量：bit = 1 ⟺ rate > 0
    expected = (rates > 0).astype(float)
    if not np.array_equal(bits, expected):
        bad = int(np.sum(bits != expected))
        raise ValueError(f"标签不满足 bit=1 ⟺ rate>0，有 {bad} 个单元不符")

    preds = predictions or {}
    for rd, mat in preds.items():
        if mat.rates.shape[0] != n:
            raise ValueError(f"预测 {rd!r} 的行数 {mat.rates.shape[0]} 与真值 {n} 不一致")

    order: Dict[str, List[int]] = {}
    for i, p in enumerate(pids):
        order.setdefault(p, []).append(i)

    trajs: List[PatientTrajectory] = []
    for pid in sorted(order):
        idx = np.asarray(order[pid], dtype=int)
        hrs = hours_all[idx]
        if len(np.unique(hrs)) != len(hrs):
            dup = sorted({int(h) for h in hrs if (hrs == h).sum() > 1})
            raise ValueError(f"患者 {pid} 的小时序号重复：{dup[:5]}")
        sort_order = np.argsort(hrs, kind="stable")
        idx, hrs = idx[sort_order], hrs[sort_order]
        if require_contiguous and len(hrs) > 1:
            gaps = np.diff(hrs)
            if np.any(gaps != 1):
                where = [int(hrs[i]) for i in np.where(gaps != 1)[0][:5]]
                raise ValueError(
                    f"患者 {pid} 的窗口表有空洞（要求连续小时）：空洞起点 {where}。"
                    "先修数据，不要插值补齐。")

        trajs.append(PatientTrajectory(
            patient_id=pid, fold=fold, hours=hrs,
            true_bits=bits[idx], true_rates=rates[idx], row_index=idx,
            pred={rd: PredictionMatrix(
                rates=mat.rates[idx],
                bits=None if mat.bits is None else mat.bits[idx],
                prob=None if mat.prob is None else mat.prob[idx],
            ) for rd, mat in preds.items()},
            observed_first=(None if observed_first is None
                            else np.asarray(observed_first, dtype=float)[idx[0]]),
        ))
    return trajs


def assert_disjoint_fold(trajs: Sequence[PatientTrajectory]) -> None:
    """断言一位患者只出现在一个 fold 里（跨 fold 泄漏的机器检查）。"""
    seen: Dict[str, str] = {}
    for t in trajs:
        prev = seen.get(t.patient_id)
        if prev is not None and prev != t.fold:
            raise AssertionError(
                f"患者 {t.patient_id} 同时出现在 fold {prev!r} 与 {t.fold!r}，存在泄漏")
        seen[t.patient_id] = t.fold


def attach_baseline(trajs: Sequence[PatientTrajectory],
                    base_by_patient: Dict[str, Sequence[float]],
                    weight_index: int = 3, tbsa_index: int = 5) -> None:
    """把体重与 TBSA 挂到轨迹上（C1 需要）。

    索引默认取自 :data:`utils.subgroups.BASE_COLUMNS`：体重 3、TBSA 5。
    缺失一律留 ``None``——不猜、不用 0 顶替，由调用方转成 NA + 原因。
    """
    for t in trajs:
        row = base_by_patient.get(t.patient_id)
        if row is None:
            continue
        arr = np.asarray(row, dtype=float).ravel()
        t.base_row = arr

        def _pick(i):
            if i >= arr.size:
                return None
            v = float(arr[i])
            return v if np.isfinite(v) else None

        t.weight_kg = _pick(weight_index)
        t.tbsa_percent = _pick(tbsa_index)


# --------------------------------------------------------------------------
# 审计
# --------------------------------------------------------------------------

def audit_trajectories(trajs: Sequence[PatientTrajectory],
                       horizons: Sequence[int] = (24, 48, 72),
                       rule: str = "all_hours_present",
                       quality_mask=None) -> Dict[str, object]:
    """每个 split 的患者数、小时数、目标数、阳性/零流分布与各时域完整人数。

    这是 #81 强制要求的数据审计口径：**窗口总数不是独立样本量**，所以这里
    同时给出 ``n_patients``（独立单位）与 ``n_patient_hours``（观测小时）。
    """
    if not trajs:
        return {"n_patients": 0, "n_patient_hours": 0, "n_targets": 0,
                "horizon_complete": {int(h): 0 for h in horizons}}

    bits = np.concatenate([t.true_bits for t in trajs], axis=0)
    rates = np.concatenate([t.true_rates for t in trajs], axis=0)
    hours = np.concatenate([t.hours for t in trajs], axis=0)

    zero_flow = ~np.any(bits > 0, axis=1)
    zero_dose = [t.patient_id for t in trajs
                 if float(t.true_rates.sum()) <= 0.0]

    complete = {}
    for h in horizons:
        complete[int(h)] = sum(
            1 for t in trajs if t.completeness(h, rule=rule,
                                               quality_mask=quality_mask)["complete"])

    return {
        "n_patients": len(trajs),
        "n_patient_hours": int(len(hours)),
        "n_targets": int(bits.size),
        "horizon_complete": complete,
        "n_zero_flow_hours": int(zero_flow.sum()),
        "zero_flow_ratio": float(zero_flow.mean()),
        "per_class_positive_cells": [int((bits[:, k] > 0).sum())
                                     for k in range(lp.NUM_TYPES)],
        "n_zero_dose_patients": len(zero_dose),
        "hour_min": int(hours.min()), "hour_max": int(hours.max()),
        "folds": sorted({t.fold for t in trajs}),
        "note": "n_targets 是 (小时, 类别) 单元数；独立单位是 n_patients，"
                "窗口/小时总数不得当作样本量。",
    }


def population_delta(before: Dict[str, object],
                     after: Dict[str, object]) -> Dict[str, object]:
    """两套资格规则之间的族群变化，**必须成为结果里可见的一行**。

    双轨敏感性分析的风险是"换了掩码就悄悄换了一批人"，那样两条结论不可比。
    这里把差异摊开：多少患者、多少小时、各时域各少了多少人。
    """
    def _h(d):
        return {int(k): int(v) for k, v in (d.get("horizon_complete") or {}).items()}

    hb, ha = _h(before), _h(after)
    return {
        "delta_patients": int(before.get("n_patients", 0)) - int(after.get("n_patients", 0)),
        "delta_patient_hours": int(before.get("n_patient_hours", 0))
        - int(after.get("n_patient_hours", 0)),
        "delta_horizon_complete": {h: hb.get(h, 0) - ha.get(h, 0)
                                   for h in sorted(set(hb) | set(ha))},
        "note": "患者在敏感性轨中因掩码转为不合格的数量；两轨的族群必须显式对比，"
                "不得默认相同。",
    }
