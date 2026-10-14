"""分类校准与回归不确定性（issue #23，对应 R4-9）。

背景
----
AUC 只衡量排序，不衡量概率是否可信。多标签任务下每个类别都要单独看校准：
模型的"0.7"究竟是不是 70% 的发生率。回归一侧则要说清误差区间是
**抽样不确定性**还是**个体预测不确定性**——两者含义不同，不可混用。

本模块提供 Brier 分解、期望校准误差（ECE）与可靠性分箱，以及
患者级区间与个体预测区间的区分。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def brier_per_class(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """逐类 Brier 分数（均方概率误差）。"""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"形状不一致：{y.shape} vs {p.shape}")
    out = {f"class_{k}": float(np.mean((p[:, k] - y[:, k]) ** 2))
           for k in range(y.shape[1])}
    out["mean"] = float(np.mean(list(out.values())))
    return out


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 10) -> Dict[str, float]:
    """期望校准误差：按置信度分箱后 |平均置信度 − 实际频率| 的加权和。

    空箱不计入（其权重为 0），但会在 ``n_empty_bins`` 中报出，
    避免"分箱很密所以 ECE 很小"的错觉。
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"形状不一致：{y.shape} vs {p.shape}")
    if not 0 < n_bins <= 1000:
        raise ValueError(f"n_bins 应在 (0, 1000]，实际 {n_bins}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    total = len(p)
    ece = 0.0
    empty = 0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            empty += 1
            continue
        conf = float(p[m].mean())
        freq = float(y[m].mean())
        ece += (m.sum() / total) * abs(conf - freq)
    return {"ece": float(ece), "n": int(total), "n_bins": n_bins,
            "n_empty_bins": int(empty)}


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray,
                      n_bins: int = 10) -> Dict[str, list]:
    """可靠性图的原始数据：每箱的平均置信度、实际频率与样本量。"""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    conf, freq, count = [], [], []
    for b in range(n_bins):
        m = idx == b
        count.append(int(m.sum()))
        conf.append(float(p[m].mean()) if m.any() else float("nan"))
        freq.append(float(y[m].mean()) if m.any() else float("nan"))
    return {"mean_confidence": conf, "fraction_positive": freq, "count": count,
            "bin_edges": edges.tolist()}


def overconfidence_gap(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """整体过度自信程度：平均置信度减去实际阳性率。

    正值表示模型**过度自信**。多标签任务下逐类给出，并给出加权平均。
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    out = {}
    gaps = []
    for k in range(y.shape[1]):
        g = float(p[:, k].mean() - y[:, k].mean())
        out[f"class_{k}"] = g
        gaps.append(g)
    out["mean"] = float(np.mean(gaps))
    out["note"] = "正值 = 过度自信"
    return out


def prediction_interval_coverage(y_true: Sequence[float],
                                 lower: Sequence[float],
                                 upper: Sequence[float]) -> Dict[str, float]:
    """区间覆盖率：真值落在 [lower, upper] 内的比例。

    必须与**名义**水平一起报告——没有名义水平的覆盖率无法解读。
    """
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError("y_true / lower / upper 形状必须一致")
    if np.any(hi < lo):
        raise ValueError("存在 upper < lower 的区间")
    inside = (y >= lo) & (y <= hi)
    return {"coverage": float(inside.mean()), "n": int(y.size),
            "mean_width": float((hi - lo).mean())}


def describe_interval_kind(kind: str) -> str:
    """区分两类区间的含义——混用会让"不确定性"这句话失去信息量。"""
    kinds = {
        "sampling": "抽样不确定性：换一批患者/窗口后会怎样，用于比较方法优劣",
        "predictive": "个体预测不确定性：这一位患者这一小时的取值区间，用于临床提示",
    }
    if kind not in kinds:
        raise ValueError(f"kind 必须是 {sorted(kinds)} 之一，实际 {kind}")
    return kinds[kind]


def ece_macro(y_true: np.ndarray, y_prob: np.ndarray,
              n_bins: int = 10) -> Dict[str, float]:
    """逐类 ECE 及其均值（v2，issue #81 的校准终点）。

    多标签任务下"整体 ECE"会被类别不平衡主导，因此逐类给出后再取均值；
    某类若只有一个标签值，ECE 仍可计算（分箱内频率恒为 0 或 1），
    故不设 NA 规则——与 AUC 不同，这一点在协议中已说明。
    """
    y = np.asarray(y_true, dtype=float)
    q = np.asarray(y_prob, dtype=float)
    if y.shape != q.shape:
        raise ValueError(f"形状不一致：{y.shape} vs {q.shape}")
    per_class = [float(expected_calibration_error(y[:, k][:, None],
                                                  q[:, k][:, None],
                                                  n_bins=n_bins)["ece"])
                 for k in range(y.shape[1])]
    return {"per_class": per_class, "ece_macro": float(np.mean(per_class)),
            "n_bins": n_bins, "aggregation": "macro",
            "note": "逐类 ECE 等权平均；类别不平衡时不得只看整体值。"}
