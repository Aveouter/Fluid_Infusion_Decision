"""指标口径协议（v2）：每个数字都必须能被完整地识别出来。

背景
----
v1（issue #7）把「口径」提升为强制字段，方向是对的，但不够：它只要求声明
``aggregation`` / ``unit`` / ``level``，于是两个**时域不同**、**掩码不同**
或**模型不同**的数字仍可能共享同一个键，在展平时互相覆盖。审计里同一个
指标名有 7 处算法、正流 MAE 能报出 29.0 与 87.2 两个数，正是这样发生的。

v2（issue #81，Gate A）做三件事：

1. **身份补全**：规范键扩展到 18 个字段（模型、run、时域、锚点、估计对象、
   液体类别、掩码、评估模式、速率口径、质量轨、协议哈希…）。键冲突直接
   抛错，不再静默覆盖。
2. **NA 纪律**：`value = None` 当且仅当 `na_reason` 非空；非空值必须有限。
   v1 允许 ``float("nan")`` 一路写到 JSON，而裸 ``NaN`` 不是合法 JSON。
3. **词表封闭 + 注册表**：单位/聚合/掩码/口径等逐项校验；已注册的指标名
   必须与其注册的 spec 一致，防止同一个名字被两处用成两种定义。

与 v1 的关系
------------
``mae()`` / ``MAE_UNITS`` / ``classification_metrics`` / ``patient_then_average``
等 v1 接口**行为保持不变**，作为兼容层保留（训练侧 ``run_epoch.py`` 仍在用）。
注意 ``MAE_UNITS`` 是 v1 专用：它把掩码编进了单位名（``positive_flow_ml_per_h``），
v2 用 ``UNITS`` + 独立的 ``mask`` 字段取代，**v2 代码路径不调用** ``mae()``。

单位与掩码必须分开，这是 v2 的核心修复之一：``mL`` 与 ``mL/h`` 是不同量纲，
``positive_flow`` 是掩码而不是单位。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from utils import eval_protocol as EP

#: 协议版本（唯一来源：:mod:`utils.eval_protocol`）。
PROTOCOL_VERSION = EP.PROTOCOL_VERSION

#: 行词表（唯一来源）。分类平均**只用** :data:`CLASSIFICATION_AVERAGES`。
AGGREGATIONS = EP.AGGREGATIONS
AGGREGATION_LEVELS = EP.AGGREGATION_LEVELS
UNITS = EP.UNITS
MASKS = EP.MASKS
FLUID_CLASSES = EP.FLUID_CLASSES
EVAL_MODES = EP.EVAL_MODES
ESTIMANDS = EP.ESTIMANDS
HORIZON_ORIGINS = EP.HORIZON_ORIGINS
RATE_DEFINITIONS = EP.RATE_DEFINITIONS
QUALITY_TRACKS = EP.QUALITY_TRACKS
EVIDENCE_TIERS = EP.EVIDENCE_TIERS
OBSERVATION_SOURCES = EP.OBSERVATION_SOURCES

#: sklearn 支持的平均方式——``classification_metrics`` 只遍历这些。
CLASSIFICATION_AVERAGES = ("macro", "micro", "weighted", "samples")

#: v1 兼容：逐小时/累计/正流三种回归口径的**单位名**。
#:
#: .. deprecated:: 2.0
#:    这里把掩码编进了单位名，v2 用 ``UNITS`` + ``mask`` 取代。
#:    仅保留给 ``mae()`` 与训练侧 ``run_epoch.py`` 使用。
MAE_UNITS = ("step_ml_per_h", "cumulative_ml", "positive_flow_ml_per_h")

#: 规范键的全部身份字段。**顺序即键的拼接顺序**，改动等于改变结果身份。
IDENTITY_FIELDS = (
    "model_id", "run_id", "level", "metric_name", "aggregation", "unit",
    "horizon_hours", "horizon_origin", "estimand", "fluid_class", "mask",
    "eval_mode", "rate_definition", "split_or_outer_fold", "train_seed",
    "threshold", "quality_track", "protocol_hash",
)


# --------------------------------------------------------------------------
# 指标注册表：让"同一个名字只能是同一种定义"
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSpec:
    """一个指标的冻结定义。注册后，任何与它不符的行都会被拒绝。"""

    name: str
    unit: str
    aggregation: str
    level: str
    mask: str
    #: ``None`` = 不约束。液体类别与（未在此列出的）时域是同一指标**合法变化**
    #: 的维度，登记时留空；单位/聚合/层级/掩码则必须钉死。
    fluid_class: Optional[str] = None
    evidence_tier: str = "na"
    interpretation_scope: str = ""
    requires_rate_definition: bool = False

    def as_expected(self) -> Dict[str, object]:
        expected = {
            "unit": self.unit, "aggregation": self.aggregation,
            "level": self.level, "mask": self.mask,
        }
        if self.fluid_class is not None:
            expected["fluid_class"] = self.fluid_class
        return expected


_REGISTRY: Dict[str, MetricSpec] = {}


def register(spec: MetricSpec) -> MetricSpec:
    """登记一个指标定义。重复登记同名指标即报错（防止两处定义漂移）。"""
    if spec.name in _REGISTRY:
        raise ValueError(f"指标 {spec.name!r} 已注册，不得重复定义")
    if spec.unit not in UNITS:
        raise ValueError(f"注册 {spec.name!r} 用了未知单位 {spec.unit!r}")
    if spec.aggregation not in AGGREGATIONS:
        raise ValueError(f"注册 {spec.name!r} 用了未知聚合 {spec.aggregation!r}")
    if spec.level not in AGGREGATION_LEVELS:
        raise ValueError(f"注册 {spec.name!r} 用了未知层级 {spec.level!r}")
    if spec.mask not in MASKS:
        raise ValueError(f"注册 {spec.name!r} 用了未知掩码 {spec.mask!r}")
    _REGISTRY[spec.name] = spec
    return spec


def registered(name: str) -> Optional[MetricSpec]:
    return _REGISTRY.get(name)


def registry() -> Dict[str, MetricSpec]:
    """已注册指标的只读副本。"""
    return dict(_REGISTRY)


# --------------------------------------------------------------------------
# 结果行
# --------------------------------------------------------------------------

@dataclass
class MetricRecord:
    """单条指标记录。任何进入论文或图表的数字都应能被这一结构完整描述。

    字段顺序：前 12 个与 v1 一致（``split`` / ``seed`` 保持原名，使按位置或
    关键字构造的既有调用继续可用），v2 的新字段一律**追加**在末尾。
    ``split_or_outer_fold`` / ``train_seed`` 是它们的 #81 别名，导出时两个
    名字都会写出且必须一致。
    """

    metric_name: str
    value: Optional[float]        # None ⇔ NA（此时 na_reason 必须非空）
    aggregation: str
    unit: str
    level: str
    split: str                    # 见属性 split_or_outer_fold
    seed: Optional[int] = None    # 见属性 train_seed
    n_patients: Optional[int] = None
    n_windows: Optional[int] = None
    mask: str = "all"
    protocol_version: str = PROTOCOL_VERSION
    note: str = ""

    # ---------------- v2 追加（全部有默认值） ----------------
    na_reason: str = ""
    horizon_hours: Optional[int] = None
    horizon_origin: str = "na"
    estimand: str = "na"
    fluid_class: str = "na"
    eval_mode: str = "na"
    rate_definition: str = "na"
    n_patient_hours: Optional[int] = None
    n_targets: Optional[int] = None
    threshold: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    data_fingerprint: str = ""
    model_id: str = ""
    run_id: str = ""
    quality_track: str = "main"
    protocol_hash: str = EP.PROTOCOL_HASH

    # ---------------- §2A 医学层的证据字段 ----------------
    evidence_tier: str = "na"
    clinical_definition_source: str = ""
    clinical_applicability: str = ""
    observation_vs_imputation: str = "na"
    missing_fraction: Optional[float] = None
    denominator_patients: Optional[int] = None
    expert_review_status: str = "not_reviewed"
    interpretation_scope: str = ""

    # ---- #81 字段名的只读别名（导出时两个名字都必须出现且一致） ----
    @property
    def split_or_outer_fold(self) -> str:
        return self.split

    @property
    def train_seed(self) -> Optional[int]:
        return self.seed

    @property
    def is_na(self) -> bool:
        return self.value is None

    # ------------------------------------------------------------
    def validate(self) -> None:
        """校验字段合法性与 NA 纪律。不通过即抛错，不做静默修正。"""
        if self.aggregation not in AGGREGATIONS + ("none",):
            raise ValueError(f"未知的 aggregation：{self.aggregation}")
        if self.unit not in UNITS:
            raise ValueError(f"未知的 unit：{self.unit}")
        if self.level not in AGGREGATION_LEVELS:
            raise ValueError(f"未知的 level：{self.level}")
        for name, allowed in (("mask", MASKS), ("fluid_class", FLUID_CLASSES),
                              ("eval_mode", EVAL_MODES), ("estimand", ESTIMANDS),
                              ("horizon_origin", HORIZON_ORIGINS),
                              ("rate_definition", RATE_DEFINITIONS),
                              ("quality_track", QUALITY_TRACKS),
                              ("evidence_tier", EVIDENCE_TIERS),
                              ("observation_vs_imputation", OBSERVATION_SOURCES)):
            got = getattr(self, name)
            if got not in allowed:
                raise ValueError(f"未知的 {name}：{got}")

        # NA 纪律：value 与 na_reason 必须同时给出或同时缺席
        if self.value is None:
            if not self.na_reason:
                raise ValueError(
                    f"指标 {self.metric_name!r} 的 value 为 NA 但未给出 na_reason")
        else:
            if self.na_reason:
                raise ValueError(
                    f"指标 {self.metric_name!r} 既有数值又声明了 na_reason "
                    f"{self.na_reason!r}，二者互斥")
            if not math.isfinite(float(self.value)):
                raise ValueError(
                    f"指标 {self.metric_name!r} 的 value 非有限值（{self.value}）；"
                    "不可估计必须写成 value=None + na_reason")

        # 区间：三个都有或都没有
        ci = [self.ci_low, self.ci_high]
        if any(c is not None for c in ci) and any(c is None for c in ci):
            raise ValueError("ci_low 与 ci_high 必须同时给出")
        if self.ci_low is not None and self.value is not None:
            if not (self.ci_low <= float(self.value) <= self.ci_high):
                raise ValueError(
                    f"区间 [{self.ci_low}, {self.ci_high}] 不包含点估计 {self.value}")

        # 样本规模自洽：**单元级**指标里每位患者至少贡献一个单元，故
        # n_targets >= n_patients。事件级指标（变点、匹配）不受此约束——
        # 2 位患者只有 1 次真实变点是正常情形，硬套会误报。
        CELL_LEVEL_MASKS = ("all", "positive_flow", "zero_flow")
        if (self.value is not None and self.mask in CELL_LEVEL_MASKS
                and self.n_patients is not None and self.n_targets is not None
                and 0 < self.n_targets < self.n_patients):
            raise ValueError(
                f"n_targets ({self.n_targets}) 小于 n_patients ({self.n_patients})："
                "单元级指标里每位患者至少贡献一个单元")

        # 注册表交叉校验：同名指标只能是同一种定义
        spec = registered(self.metric_name)
        if spec is not None:
            for key, want in spec.as_expected().items():
                got = getattr(self, key)
                if got != want:
                    raise ValueError(
                        f"指标 {self.metric_name!r} 的 {key}={got!r} 与其注册定义 "
                        f"{want!r} 不一致；同名指标不得有两种口径")
            if spec.requires_rate_definition and self.rate_definition == "na":
                raise ValueError(
                    f"指标 {self.metric_name!r} 依赖速率口径，必须声明 rate_definition")

    def identity(self) -> Dict[str, object]:
        """规范键的 18 个字段。"""
        return {k: getattr(self, k) for k in IDENTITY_FIELDS}

    def identity_key(self) -> str:
        return "|".join(str(self.identity()[k]) for k in IDENTITY_FIELDS)

    def to_dict(self) -> Dict[str, object]:
        """导出为字典。同时给出 #81 字段名与 v1 别名，二者必然一致。"""
        row = asdict(self)
        row["split_or_outer_fold"] = self.split_or_outer_fold
        row["train_seed"] = self.train_seed
        row["identity_key"] = self.identity_key()
        return row


# --------------------------------------------------------------------------
# 导出器
# --------------------------------------------------------------------------

class MetricsExporter:
    """把指标按统一 schema 落盘，供表格与绘图统一取数。

    构造时给定一次运行的身份（模型、run、split、seed、质量轨、数据指纹），
    逐行只需给出该行特有的字段。
    """

    def __init__(self, split: str, seed: Optional[int] = None, *,
                 model_id: str = "", run_id: str = "",
                 quality_track: str = "main", data_fingerprint: str = ""):
        if quality_track not in QUALITY_TRACKS:
            raise ValueError(f"未知的 quality_track：{quality_track}")
        self.split = split
        self.seed = seed
        self.model_id = model_id
        self.run_id = run_id
        self.quality_track = quality_track
        self.data_fingerprint = data_fingerprint
        self.records: List[MetricRecord] = []

    # ------------------------------------------------------------
    def add(self, metric_name: str, value: Optional[float], *,
            aggregation: str = "none", unit: str = "probability",
            level: str = "window", mask: str = "all", note: str = "",
            n_patients: Optional[int] = None, n_windows: Optional[int] = None,
            na_reason: str = "", horizon_hours: Optional[int] = None,
            horizon_origin: str = "na", estimand: str = "na",
            fluid_class: str = "na", eval_mode: str = "na",
            rate_definition: str = "na",
            n_patient_hours: Optional[int] = None,
            n_targets: Optional[int] = None,
            threshold: Optional[float] = None,
            ci_low: Optional[float] = None, ci_high: Optional[float] = None,
            evidence_tier: str = "na", clinical_definition_source: str = "",
            clinical_applicability: str = "",
            observation_vs_imputation: str = "na",
            missing_fraction: Optional[float] = None,
            denominator_patients: Optional[int] = None,
            expert_review_status: str = "not_reviewed",
            interpretation_scope: str = "") -> MetricRecord:
        rec = MetricRecord(
            metric_name=metric_name, value=value, aggregation=aggregation,
            unit=unit, level=level, split=self.split, seed=self.seed,
            n_patients=n_patients, n_windows=n_windows, mask=mask, note=note,
            na_reason=na_reason, horizon_hours=horizon_hours,
            horizon_origin=horizon_origin, estimand=estimand,
            fluid_class=fluid_class, eval_mode=eval_mode,
            rate_definition=rate_definition, n_patient_hours=n_patient_hours,
            n_targets=n_targets, threshold=threshold, ci_low=ci_low,
            ci_high=ci_high, data_fingerprint=self.data_fingerprint,
            model_id=self.model_id, run_id=self.run_id,
            quality_track=self.quality_track,
            evidence_tier=evidence_tier,
            clinical_definition_source=clinical_definition_source,
            clinical_applicability=clinical_applicability,
            observation_vs_imputation=observation_vs_imputation,
            missing_fraction=missing_fraction,
            denominator_patients=denominator_patients,
            expert_review_status=expert_review_status,
            interpretation_scope=interpretation_scope,
        )
        rec.validate()
        self.records.append(rec)
        return rec

    def add_na(self, metric_name: str, *, na_reason: str, **kwargs) -> MetricRecord:
        """显式记录一个不可估计的终点。

        不允许把不可估计写成 0 或直接省略：分母变化与缺失原因是结论的一部分。
        """
        if not na_reason:
            raise ValueError(f"指标 {metric_name!r} 声明为 NA 必须给出 na_reason")
        return self.add(metric_name, None, na_reason=na_reason, **kwargs)

    # ------------------------------------------------------------
    def as_dict(self, key_fields: Sequence[str] = IDENTITY_FIELDS) -> Dict[str, float]:
        """按规范键展平。**键冲突即抛错**，不再静默覆盖。

        v1 用的是 ``name|aggregation|unit|level``——时域、掩码、模型、评估模式
        都不在键里，正是"两个不同东西挤成一行"的通道。
        """
        out: Dict[str, float] = {}
        seen: Dict[str, MetricRecord] = {}
        for rec in self.records:
            key = "|".join(str(rec.identity()[k]) for k in key_fields)
            if key in out:
                prev = seen[key]
                raise ValueError(
                    f"指标键冲突：{key}\n"
                    f"  已有：value={prev.value} rate_definition={prev.rate_definition} "
                    f"note={prev.note!r}\n"
                    f"  新增：value={rec.value} rate_definition={rec.rate_definition} "
                    f"note={rec.note!r}\n"
                    "两个不同的数字不能共享同一个身份。")
            seen[key] = rec
            out[key] = rec.value
        return out

    def na_ledger(self) -> List[Dict[str, object]]:
        """本次导出的 NA 台账：每个不可估计的终点及其原因。"""
        return [{"metric_name": r.metric_name, "na_reason": r.na_reason,
                 "mask": r.mask, "horizon_hours": r.horizon_hours,
                 "estimand": r.estimand, "fluid_class": r.fluid_class,
                 "model_id": r.model_id, "rate_definition": r.rate_definition}
                for r in self.records if r.is_na]

    def to_json(self) -> str:
        return json.dumps([r.to_dict() for r in self.records],
                          ensure_ascii=False, indent=2, allow_nan=False)

    def write(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


# --------------------------------------------------------------------------
# 分类指标：所有聚合方式一次算全（v1 行为保持不变）
# --------------------------------------------------------------------------

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_prob: Optional[np.ndarray] = None) -> Dict[str, dict]:
    """返回 ``{aggregation: {precision, recall, f1, auc?}}``，各口径全部给出。

    多标签任务下必须看全部口径：只报一个数字是审稿意见的根源之一。
    只遍历 :data:`CLASSIFICATION_AVERAGES`（sklearn 支持的那些），
    因为 :data:`AGGREGATIONS` 还含 patient_* / event_* 等行级词表。
    """
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score)

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"形状不一致：{y_true.shape} vs {y_pred.shape}")

    out: Dict[str, dict] = {}
    per_class: Dict[str, float] = {}
    for agg in CLASSIFICATION_AVERAGES:
        try:
            out[agg] = {
                "precision": float(precision_score(y_true, y_pred, average=agg, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average=agg, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, average=agg, zero_division=0)),
            }
        except ValueError as exc:                      # 例如某类全为 0
            out[agg] = {"precision": float("nan"), "recall": float("nan"),
                        "f1": float("nan"), "error": str(exc)}

    K = y_true.shape[1]
    for k in range(K):
        per_class[f"class_{k}"] = {
            "precision": float(precision_score(y_true[:, k], y_pred[:, k], zero_division=0)),
            "recall": float(recall_score(y_true[:, k], y_pred[:, k], zero_division=0)),
            "f1": float(f1_score(y_true[:, k], y_pred[:, k], zero_division=0)),
            "support": int(y_true[:, k].sum()),
        }
    out["per_class"] = per_class

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        aucs = []
        for k in range(K):
            if len(np.unique(y_true[:, k])) < 2:
                aucs.append(float("nan"))
                continue
            aucs.append(float(roc_auc_score(y_true[:, k], y_prob[:, k])))
        out["auc_per_class"] = aucs
        out["auc_macro"] = float(np.nanmean(aucs)) if aucs else float("nan")

    return out


def accuracy_bit(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """逐位准确率（把所有 (样本, 类别) 位一起算）。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


# --------------------------------------------------------------------------
# 回归指标
# --------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray, *,
        unit: str, mask: Optional[np.ndarray] = None) -> float:
    """带单位声明的 MAE（**v1 兼容接口**）。

    ``unit`` 必须取自 :data:`MAE_UNITS`。注意这里把掩码编进了单位名，
    v2 代码请改用 :func:`mean_abs_error`（``unit`` 用 :data:`UNITS`，
    掩码由独立的 ``mask`` 字段承载）。
    """
    if unit not in MAE_UNITS:
        raise ValueError(f"unit 必须是 {MAE_UNITS} 之一，实际 {unit}")
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_abs_error(y_true, y_pred, *, unit: str,
                   mask: Optional[np.ndarray] = None,
                   weights: Optional[np.ndarray] = None) -> float:
    """v2 的 MAE：``unit`` 取 :data:`UNITS`，掩码与单位分离。

    仍然返回 ``nan`` 表示"没有可用样本"；**调用方必须把它转成
    ``value=None + na_reason``**，不得把 nan 当作 0 写进结果。
    """
    if unit not in UNITS:
        raise ValueError(f"unit 必须是 {UNITS} 之一，实际 {unit}")
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"形状不一致：{a.shape} vs {b.shape}")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != a.shape:
            raise ValueError(f"mask 形状 {mask.shape} 与数据 {a.shape} 不一致")
        a, b = a[mask], b[mask]
        if weights is not None:
            weights = np.asarray(weights, dtype=float)[mask]
    if a.size == 0:
        return float("nan")
    err = np.abs(a - b)
    if weights is None:
        return float(err.mean())
    return float(np.sum(err * weights) / np.sum(weights))


def positive_flow_mask(y_true_rate: np.ndarray, eps: float = None) -> np.ndarray:
    """真实速率**高于** ``eps`` 的掩码（逐元素）。

    冻结时 ``eps`` 取 :data:`utils.eval_protocol.EPS_RATE_ML_PER_H`（默认 0.0）。
    真实零值（``rate == 0``）与缺失（哨兵，已在标签协议中置 0）都不在掩码内，
    但二者**语义不同**，报告时必须分开说明。
    """
    thr = EP.EPS_RATE_ML_PER_H if eps is None else float(eps)
    return np.asarray(y_true_rate, dtype=float) > thr


def zero_flow_mask(y_true_rate: np.ndarray, eps: float = None) -> np.ndarray:
    """真实速率恰为零（真实零输注，**不是**缺失）的掩码。"""
    thr = EP.EPS_RATE_ML_PER_H if eps is None else float(eps)
    return np.asarray(y_true_rate, dtype=float) <= thr


def r2_with_denominator(y_true, y_pred, *,
                        mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """R² 及其分母，避免"R² 是在哪个子集、对哪个均值算的"说不清。

    分母固定为 ``sum_sq_dev_from_true_mean_over_masked_targets``——即
    ``SS_tot = Σ(y - mean(y))²`` 只在掩码内计算。
    """
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"形状不一致：{a.shape} vs {b.shape}")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        a, b = a[mask], b[mask]
    if a.size == 0:
        return {"r2": float("nan"), "ss_res": float("nan"),
                "ss_tot": float("nan"), "n": 0, "mean_true": float("nan"),
                "denominator": "ss_tot=0_or_empty"}
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return {
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "ss_res": ss_res, "ss_tot": ss_tot, "n": int(a.size),
        "mean_true": float(a.mean()),
        "denominator": "sum_sq_dev_from_true_mean_over_masked_targets",
    }


# --------------------------------------------------------------------------
# 聚合层级：窗口级 vs 先按患者再平均（v1 行为保持不变）
# --------------------------------------------------------------------------

def window_average(values: np.ndarray,
                   weights: Optional[np.ndarray] = None) -> float:
    """直接对所有窗口求平均（微平均）。

    同一患者的多个窗口会被当作独立样本——在患者间窗口数不等时，
    窗口多的患者贡献更大，通常**不是**想要的科研口径。
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan")
    if weights is None:
        return float(values.mean())
    w = np.asarray(weights, dtype=float)
    return float(np.sum(values * w) / np.sum(w))


def patient_then_average(values: np.ndarray, patient_ids: Sequence[str]) -> float:
    """先按患者聚合（取均值），再对患者求平均（宏平均）。

    这是患者级独立性的正确口径：每位患者权重相同，与窗口数无关。
    """
    values = np.asarray(values, dtype=float)
    ids = list(patient_ids)
    if len(values) != len(ids):
        raise ValueError(f"values 与 patient_ids 长度不一致：{len(values)} vs {len(ids)}")
    per_patient: Dict[str, List[float]] = {}
    for v, pid in zip(values, ids):
        if np.isnan(v):
            continue
        per_patient.setdefault(pid, []).append(float(v))
    if not per_patient:
        return float("nan")
    means = [float(np.mean(vs)) for vs in per_patient.values()]
    return float(np.mean(means))


def patient_mean(values: np.ndarray, patient_ids: Sequence[str]) -> np.ndarray:
    """每位患者一个值（患者内取均值），返回按患者编号排序的向量。

    患者级 bootstrap 需要"一位患者一个值"这一形式；直接对逐窗口值重采样
    会把同一患者的若干小时当成独立样本，人为压窄区间。
    """
    values = np.asarray(values, dtype=float)
    ids = [str(p) for p in patient_ids]
    if len(values) != len(ids):
        raise ValueError(f"values 与 patient_ids 长度不一致：{len(values)} vs {len(ids)}")
    grouped: Dict[str, List[float]] = {}
    for v, pid in zip(values, ids):
        if np.isnan(v):
            continue
        grouped.setdefault(pid, []).append(float(v))
    return np.array([float(np.mean(grouped[k])) for k in sorted(grouped)], dtype=float)


# --------------------------------------------------------------------------
# 零流 / 类别缺失的显式处理
# --------------------------------------------------------------------------

def describe_zero_handling(y_true: np.ndarray, y_true_rate: np.ndarray) -> dict:
    """报告零流与全零类的规模，避免"静默返回误导性均值"。"""
    y_true = np.asarray(y_true).astype(int)
    rate = np.asarray(y_true_rate, dtype=float)
    n = int(y_true.shape[0])
    all_zero = ~np.any(y_true > 0, axis=1)
    per_class_positive = [int(y_true[:, k].sum()) for k in range(y_true.shape[1])]
    return {
        "n_rows": n,
        "zero_flow_rows": int(all_zero.sum()),
        "zero_flow_ratio": (float(all_zero.mean()) if n else 0.0),
        "per_class_positive": per_class_positive,
        "classes_without_positive": [k for k, c in enumerate(per_class_positive) if c == 0],
        "positive_flow_cells": int((rate > 0).sum()),
        "note": "多标签下各类阳性数之和可超过行数；无输注由三类全零定义，"
                "不能由占比余数推得（见 utils/label_protocol.py）。",
    }
