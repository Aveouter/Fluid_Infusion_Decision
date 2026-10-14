"""指标口径协议（v1）：聚合方式、单位与聚合层级都必须显式声明。

背景（issue #7，对应 R1-3 / R2-3 / R3-3 / R4-15）
------------------------------------------------
论文里同一个数字在不同表格中口径不一致：表 2 的 BurnResu 行 AUC=0.816 与
召回=0.813 属 macro 口径（与表 4 的 Macro Average 一致），但同一行的
F1=0.743 既不是 macro（表 4 给 0.696），也不是任何单一类别，而**随包代码
只计算 macro**，无法产出该值。

本模块把口径变成强制字段，避免同类问题再次发生：

- 分类指标的每一次计算都要声明 **aggregation**（macro / micro / weighted /
  samples / per_class）；
- 回归指标的每一次计算都要声明 **unit**（每步 mL/h、累计 mL、仅正流 mL/h）；
- 聚合的 **层级**（窗口级 vs 先按患者再平均）必须显式选择。

多标签注意：三类液体相互独立（见 ``utils/label_protocol.py``），
因此 micro / samples 等口径与多类别任务的含义不同，必须写明。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

PROTOCOL_VERSION = "1.0"

#: 分类指标允许的聚合方式。
AGGREGATIONS = ("per_class", "macro", "micro", "weighted", "samples")

#: 回归指标的允许单位。三者不可互换：
#: - ``step_ml_per_h``      逐小时速率误差，单位 mL/h
#: - ``cumulative_ml``      单个患者累计体积误差，单位 mL
#: - ``positive_flow_ml_per_h``  仅在真实正流小时上计算的速率误差，单位 mL/h
MAE_UNITS = ("step_ml_per_h", "cumulative_ml", "positive_flow_ml_per_h")

#: 聚合层级。
AGGREGATION_LEVELS = ("window", "patient")


@dataclass
class MetricRecord:
    """单条指标记录。任何进入论文或图表的数字都应能被这一结构描述。"""

    metric_name: str
    value: float
    aggregation: str          # AGGREGATIONS 之一，回归指标用 "none"
    unit: str                 # MAE_UNITS 之一，分类指标用 "probability"
    level: str                # AGGREGATION_LEVELS 之一
    split: str                # train / val / test
    seed: Optional[int] = None
    n_patients: Optional[int] = None
    n_windows: Optional[int] = None
    mask: str = "all"         # all / positive_flow / zero_flow
    protocol_version: str = PROTOCOL_VERSION
    note: str = ""

    def validate(self) -> None:
        if self.aggregation not in AGGREGATIONS + ("none",):
            raise ValueError(f"未知的 aggregation：{self.aggregation}")
        if self.unit not in MAE_UNITS + ("probability", "count"):
            raise ValueError(f"未知的 unit：{self.unit}")
        if self.level not in AGGREGATION_LEVELS:
            raise ValueError(f"未知的 level：{self.level}")


class MetricsExporter:
    """把指标按统一 schema 落盘，供表格与绘图统一取数。"""

    def __init__(self, split: str, seed: Optional[int] = None):
        self.split = split
        self.seed = seed
        self.records: List[MetricRecord] = []

    def add(self, metric_name: str, value: float, *, aggregation: str = "none",
            unit: str = "probability", level: str = "window",
            n_patients: Optional[int] = None, n_windows: Optional[int] = None,
            mask: str = "all", note: str = "") -> MetricRecord:
        rec = MetricRecord(
            metric_name=metric_name, value=float(value), aggregation=aggregation,
            unit=unit, level=level, split=self.split, seed=self.seed,
            n_patients=n_patients, n_windows=n_windows, mask=mask, note=note,
        )
        rec.validate()
        self.records.append(rec)
        return rec

    def to_json(self) -> str:
        return json.dumps([asdict(r) for r in self.records],
                          ensure_ascii=False, indent=2)

    def write(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def as_dict(self) -> Dict[str, float]:
        """按 ``name|aggregation|unit|level`` 展平，避免同名指标互相覆盖。"""
        return {
            f"{r.metric_name}|{r.aggregation}|{r.unit}|{r.level}": r.value
            for r in self.records
        }


# --------------------------------------------------------------------------
# 分类指标：所有聚合方式一次算全
# --------------------------------------------------------------------------

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_prob: Optional[np.ndarray] = None) -> Dict[str, dict]:
    """返回 ``{aggregation: {precision, recall, f1, auc?}}``，五种口径全部给出。

    多标签任务下必须看全部口径：只报一个数字是本次审稿意见的根源之一。
    """
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score)

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"形状不一致：{y_true.shape} vs {y_pred.shape}")

    out: Dict[str, dict] = {}
    per_class: Dict[str, float] = {}
    for agg in AGGREGATIONS:
        if agg == "per_class":
            continue
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
# 回归指标：单位与掩码必须显式
# --------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray, *,
        unit: str, mask: Optional[np.ndarray] = None) -> float:
    """带单位声明的 MAE。

    ``unit`` 必须取自 :data:`MAE_UNITS`；调用方有责任保证传入的量纲与该单位
    一致（例如累计体积要传 mL，而不是 mL/h）。
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


def positive_flow_mask(y_true_rate: np.ndarray) -> np.ndarray:
    """真实速率为正的掩码（逐元素）。"""
    return np.asarray(y_true_rate, dtype=float) > 0


# --------------------------------------------------------------------------
# 聚合层级：窗口级 vs 先按患者再平均
# --------------------------------------------------------------------------

def window_average(values: np.ndarray,
                   weights: Optional[np.ndarray] = None) -> float:
    """直接对所有窗口求平均（微平均）。

    同一患者的多个窗口会被当作独立样本——在患者间窗口数不等时，
    患者越多的窗口贡献越大，通常**不是**想要的科研口径。
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
