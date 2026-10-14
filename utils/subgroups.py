"""患者级误差分布、失败病例与临床亚组（issue #22，对应 R2-7/9、R4-10/11）。

背景
----
总体均值会掩盖失效：队列男性占 74.7%，烧伤面积与吸入性损伤高度异质，
平均表现可能来自少数容易的患者。审稿人要求给出**逐患者误差分布**、
**失败病例**、以及按临床相关分层的性能，并要求说明代表性病例的**选例规则**
（否则等同于挑样）。

原则
----
- 亚组一律先按患者聚合再统计；患者是独立性单位（见 #6）。
- 亚组只报样本量与性能，样本过小时明确标注，**不作强泛化结论**。
- 选例规则必须**预先定义且确定性**，不允许按结果好坏挑图。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

#: 基线特征的列序，与 ``data_loader`` 及数据字典一致。
BASE_COLUMNS = ("sex", "age", "height_cm", "weight_kg", "bmi",
                "tbsa_percent", "tbsa_iii", "tbsa_deep_ii", "tbsa_superficial_ii",
                "inhalation_injury")

#: 亚组样本量下限。低于此值仍会报告，但标注为不稳健。
MIN_GROUP_N = 10

#: 预设的年龄分层（岁）。
AGE_BINS = (0, 18, 40, 60, 200)
AGE_LABELS = ("<18", "18-39", "40-59", "60+")

#: 预设的 TBSA 分层（百分数）。
TBSA_BINS = (0, 30, 50, 200)
TBSA_LABELS = ("<30", "30-49", "50+")


def subgroup_labels(base_row: Sequence[float]) -> Dict[str, str]:
    """把一位患者的基线特征映射到各分层标签。

    返回 ``{维度: 标签}``。缺失值（NaN）统一归入 ``unknown``，
    不猜测、不填补——否则亚组结论会建立在编造的分组上。
    """
    b = np.asarray(base_row, dtype=float).ravel()
    if b.size < len(BASE_COLUMNS):
        raise ValueError(f"基线特征至少需要 {len(BASE_COLUMNS)} 维，实际 {b.size}")
    sex, age, tbsa, inhale = b[0], b[1], b[5], b[9]

    def _band(v, bins, labels):
        if not np.isfinite(v):
            return "unknown"
        for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
            if lo <= v < hi:
                return lab
        return labels[-1]

    return {
        "sex": ("male" if sex == 1 else "female") if np.isfinite(sex) else "unknown",
        "age": _band(age, AGE_BINS, AGE_LABELS),
        "tbsa": _band(tbsa, TBSA_BINS, TBSA_LABELS),
        "inhalation": (f"injury_{int(inhale)}" if np.isfinite(inhale) else "unknown"),
    }


def group_metrics(values: np.ndarray, patient_ids: Sequence[str],
                  labels: Sequence[str]) -> Dict[str, dict]:
    """按给定标签分组统计，**先按患者聚合**再描述组内分布。

    ``values`` 与 ``patient_ids`` 为一窗一行；``labels`` 为患者级标签（与
    ``patient_ids`` 等长）。返回每组 ``{n_patients, mean, p50, p90, unstable}``。
    """
    values = np.asarray(values, dtype=float)
    ids = [str(p) for p in patient_ids]
    labs = [str(x) for x in labels]
    if not (len(values) == len(ids) == len(labs)):
        raise ValueError("values / patient_ids / labels 长度必须一致")

    per_patient: Dict[Tuple[str, str], list] = {}
    for v, pid, lab in zip(values, ids, labs):
        if not np.isfinite(v):
            continue
        per_patient.setdefault((lab, pid), []).append(float(v))

    grouped: Dict[str, List[float]] = {}
    for (lab, _pid), vs in per_patient.items():
        grouped.setdefault(lab, []).append(float(np.mean(vs)))

    out: Dict[str, dict] = {}
    for lab, vs in sorted(grouped.items()):
        arr = np.asarray(vs, dtype=float)
        out[lab] = {
            "n_patients": int(arr.size),
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(arr.max()),
            "unstable": bool(arr.size < MIN_GROUP_N),
        }
    return out


def select_cases(values: np.ndarray, patient_ids: Sequence[str],
                 n_each: int = 2) -> Dict[str, list]:
    """**预先定义、确定性**的选例规则：中位、最好、最差各取若干。

    规则只依赖数值排序，不依赖人工挑选；并列时按患者编号排序以保证可复现。
    返回 ``{bucket: [(patient_id, value), ...]}``。

    这样图 6 的病例就有据可依，且**必须同时给出最差病例**——
    只展示成功案例属于挑样。
    """
    if n_each < 1:
        raise ValueError(f"n_each 必须 >= 1，实际 {n_each}")
    per_patient: Dict[str, list] = {}
    for v, pid in zip(np.asarray(values, dtype=float), [str(p) for p in patient_ids]):
        if np.isfinite(v):
            per_patient.setdefault(str(pid), []).append(float(v))
    if not per_patient:
        return {"median": [], "best": [], "worst": []}

    scored = sorted(((pid, float(np.mean(vs))) for pid, vs in per_patient.items()),
                    key=lambda kv: (kv[1], kv[0]))
    n = len(scored)
    mid = n // 2
    half = max(1, n_each // 2)
    median_block = scored[max(0, mid - half):mid + half] or scored[mid:mid + 1]
    return {
        "best": scored[:n_each],
        "worst": scored[-n_each:][::-1],
        "median": median_block,
    }


def per_patient_errors(pred_rates: np.ndarray, true_rates: np.ndarray,
                       patient_ids: Sequence[str],
                       mask: np.ndarray = None) -> Dict[str, float]:
    """逐患者误差统计：均值、分位数、最大，以及**失败患者计数**。

    ``mask`` 默认取真实速率 > 0；每位患者的误差先在其自身窗口上平均，
    再进入分布统计（患者等权）。
    """
    pred = np.asarray(pred_rates, dtype=float)
    true = np.asarray(true_rates, dtype=float)
    if pred.shape != true.shape:
        raise ValueError(f"形状不一致：{pred.shape} vs {true.shape}")
    if mask is None:
        mask = true > 0
    mask = np.asarray(mask, dtype=bool)

    per_patient: Dict[str, list] = {}
    for i, pid in enumerate([str(p) for p in patient_ids]):
        m = mask[i]
        if m.sum() == 0:
            continue
        per_patient.setdefault(pid, []).extend(
            np.abs(pred[i][m] - true[i][m]).tolist())
    if not per_patient:
        return {"n_patients": 0}

    means = np.array([float(np.mean(v)) for v in per_patient.values()])
    all_err = np.concatenate([np.asarray(v, dtype=float)
                              for v in per_patient.values()])
    return {
        "n_patients": int(means.size),
        "patient_mean": float(means.mean()),
        "patient_p50": float(np.percentile(means, 50)),
        "patient_p90": float(np.percentile(means, 90)),
        "patient_max": float(means.max()),
        "cell_mae": float(all_err.mean()),
        "cell_p90": float(np.percentile(all_err, 90)),
        "unit": "positive_flow_ml_per_h",
    }


def failure_patients(pred_rates: np.ndarray, true_rates: np.ndarray,
                     patient_ids: Sequence[str], threshold: float,
                     mask: np.ndarray = None) -> List[Dict[str, float]]:
    """逐患者平均误差超过 ``threshold`` 的病例，按误差降序。"""
    pred = np.asarray(pred_rates, dtype=float)
    true = np.asarray(true_rates, dtype=float)
    if mask is None:
        mask = true > 0
    mask = np.asarray(mask, dtype=bool)

    out: Dict[str, list] = {}
    for i, pid in enumerate([str(p) for p in patient_ids]):
        m = mask[i]
        if m.sum() == 0:
            continue
        out.setdefault(pid, []).extend(np.abs(pred[i][m] - true[i][m]).tolist())

    rows = [{"patient_id": pid, "mean_abs_error": float(np.mean(v)),
             "n_cells": len(v)}
            for pid, v in out.items() if float(np.mean(v)) > float(threshold)]
    return sorted(rows, key=lambda r: (-r["mean_abs_error"], r["patient_id"]))


# ==========================================================================
# v2（issue #81）：亚组结果的结构化输出
# ==========================================================================

def per_patient_error_values(pred_rates: np.ndarray, true_rates: np.ndarray,
                             patient_ids: Sequence[str],
                             mask: np.ndarray = None
                             ) -> Tuple[np.ndarray, List[str]]:
    """逐患者一个误差值（患者内先平均），返回 ``(values, patient_ids)``。

    亚组统计需要"一位患者一个值"这一形式；把逐单元误差直接交给分组函数会
    让小时多的患者主导组内均值。``mask`` 默认取真实速率 > 0。
    """
    pred = np.asarray(pred_rates, dtype=float)
    true = np.asarray(true_rates, dtype=float)
    if pred.shape != true.shape:
        raise ValueError(f"形状不一致：{pred.shape} vs {true.shape}")
    ids = [str(p) for p in patient_ids]
    if len(ids) != len(pred):
        raise ValueError(f"patient_ids 长度 {len(ids)} 与数据 {len(pred)} 不一致")
    if mask is None:
        mask = true > 0
    mask = np.asarray(mask, dtype=bool)

    per_patient: Dict[str, List[float]] = {}
    for i, pid in enumerate(ids):
        m = mask[i]
        if not m.any():
            continue
        per_patient.setdefault(pid, []).extend(np.abs(pred[i][m] - true[i][m]).tolist())

    keys = sorted(per_patient)
    values = np.array([float(np.mean(per_patient[k])) for k in keys], dtype=float)
    return values, keys


def subgroup_rows(values: np.ndarray, patient_ids: Sequence[str],
                  labels: Sequence[str], *, metric_name: str,
                  unit: str, mask: str) -> List[Dict[str, object]]:
    """把亚组统计整理成结果行的**取值部分**（不含 schema 字段）。

    每条都带 ``n_patients`` 与 ``unstable``：样本过小的亚组只作描述，
    **不作显著优越主张**。这一点必须随行输出，否则亚组表会被读成结论。
    """
    grouped = group_metrics(values, patient_ids, labels)
    rows: List[Dict[str, object]] = []
    for lab, stat in sorted(grouped.items()):
        rows.append({
            "metric_name": metric_name,
            "subgroup": lab,
            "value": stat["mean"],
            "unit": unit,
            "mask": mask,
            "aggregation": "patient_mean",
            "n_patients": stat["n_patients"],
            "p50": stat["p50"], "p90": stat["p90"], "max": stat["max"],
            "unstable": stat["unstable"],
            "note": ("样本量低于 %d，仅作描述，不作优越性主张" % MIN_GROUP_N
                     if stat["unstable"] else ""),
        })
    return rows
