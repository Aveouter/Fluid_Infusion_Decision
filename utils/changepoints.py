"""输液变更点的定义与分层评估（issue #10，对应 R1-1 / R1-3 / R2-2 / R4-11）。

背景
----
总体 MAE 会被大量"稳定小时"稀释：若多数小时速率几乎不变，一个只会延续
上一刻的模型也能取得不错的总体误差，而在临床医生**改变**输注速率时失效。
审稿人 1 从轨迹图看到约一小时滞后，要求单独报告变更时点的性能。

定义（v1，冻结于评估之前）
------------------------
以**真实**速率序列为准判定变更，预测方不得参与：

- ``abs``    ：某类速率相对上一小时的变化量 >= ``CHANGE_ABS_ML``（mL/h）
- ``rel``    ：相对变化 >= ``CHANGE_REL``，且基数足够大（避免小速率下的噪声）
- 两者取**或**，因为临床上"从 0 开启"只有绝对量有意义，而高速率下的
  小幅调整只有相对量有意义。
- 变更细分四类：``off_to_on``（0→正）、``on_to_off``（正→0）、
  ``rate_up``、``rate_down``。开启/停止单独成类，因为它们临床含义不同。

这些阈值必须**在查看测试结果之前冻结**；本模块的默认值是先验设定的，
若要改动需在论文中说明并重跑。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

#: 绝对变化阈值（mL/h）、相对变化阈值与基数下限。
#:
#: v2 起**唯一来源**是 :mod:`utils.eval_protocol`——此处只做 re-export，
#: 使"冻结阈值"与"冻结 hash"必然一致。
from utils.eval_protocol import (  # noqa: E402  (re-export)
    CHANGE_ABS_ML,
    CHANGE_REL,
    REL_BASE_ML,
)

#: 变更类型。
CHANGE_TYPES = ("off_to_on", "on_to_off", "rate_up", "rate_down")


def detect_changes(rates: np.ndarray,
                   abs_thresh: float = CHANGE_ABS_ML,
                   rel_thresh: float = CHANGE_REL,
                   rel_base: float = REL_BASE_ML) -> np.ndarray:
    """返回 (T, C) 的布尔矩阵，标记每个 (时刻, 类别) 是否为变更点。

    ``t=0`` 没有前值，恒为 False。判定只用真实速率序列，与预测无关。
    """
    rates = np.asarray(rates, dtype=float)
    if rates.ndim != 2:
        raise ValueError(f"rates 应为 (T, C)，实际 {rates.shape}")
    T, C = rates.shape
    out = np.zeros((T, C), dtype=bool)
    if T < 2:
        return out

    prev = rates[:-1]
    cur = rates[1:]
    delta = np.abs(cur - prev)
    base = np.maximum(np.abs(prev), np.abs(cur))
    abs_hit = delta >= abs_thresh
    rel_hit = (base >= rel_base) & (delta >= rel_thresh * base)
    out[1:] = abs_hit | rel_hit
    return out


def classify_change(prev: float, cur: float) -> str:
    """把一次变更归类。仅在确实构成变更时调用。"""
    if prev <= 0 and cur > 0:
        return "off_to_on"
    if prev > 0 and cur <= 0:
        return "on_to_off"
    return "rate_up" if cur > prev else "rate_down"


def change_type_matrix(rates: np.ndarray, changes: np.ndarray = None) -> np.ndarray:
    """返回 (T, C) 的字符串矩阵，非变更处为空串。"""
    rates = np.asarray(rates, dtype=float)
    if changes is None:
        changes = detect_changes(rates)
    T, C = rates.shape
    out = np.full((T, C), "", dtype=object)
    for t in range(1, T):
        for c in range(C):
            if changes[t, c]:
                out[t, c] = classify_change(rates[t - 1, c], rates[t, c])
    return out


def neighborhood_mask(changes: np.ndarray, radius: int) -> np.ndarray:
    """变更点的 ±radius 邻域掩码（含变更点本身）。"""
    if radius < 0:
        raise ValueError(f"radius 必须 >= 0，实际 {radius}")
    changes = np.asarray(changes, dtype=bool)
    out = changes.copy()
    for r in range(1, radius + 1):
        out[r:] |= changes[:-r]
        out[:-r] |= changes[r:]
    return out


def stratified_metrics(pred_rates: np.ndarray, true_rates: np.ndarray,
                       changes: np.ndarray, radius: int = 1) -> Dict[str, dict]:
    """按"稳定 / 变更 / 变更邻域"分层统计逐单元误差。

    仅统计真实速率 > 0 的单元（与 #9 的对照口径一致），并单独给出变更点
    自身所在单元的数量，避免小样本被当作总体结论。

    .. warning::
       ``changed`` 是 ``changed_neighborhood`` 的**子集**（邻域含变更点本身），
       两组刻意重叠以便同时给出"仅变更点"与"变更±radius"两种视角。
       **不要把它们相加**来求单元总数，否则会重复计数；``stable`` 与
       ``changed_neighborhood`` 才是互斥的。
    """
    pred = np.asarray(pred_rates, dtype=float)
    true = np.asarray(true_rates, dtype=float)
    if pred.shape != true.shape:
        raise ValueError(f"形状不一致：{pred.shape} vs {true.shape}")
    changes = np.asarray(changes, dtype=bool)
    if changes.shape != true.shape:
        raise ValueError(f"changes 形状 {changes.shape} 与速率 {true.shape} 不一致")

    positive = true > 0
    near = neighborhood_mask(changes, radius)

    groups = {
        "stable": positive & ~near,
        "changed": positive & changes,
        "changed_neighborhood": positive & near,
    }
    out: Dict[str, dict] = {}
    for name, mask in groups.items():
        if mask.sum() == 0:
            out[name] = {"n_cells": 0, "mae": float("nan")}
            continue
        out[name] = {
            "n_cells": int(mask.sum()),
            "mae": float(np.abs(pred[mask] - true[mask]).mean()),
        }
    return out


def per_type_metrics(pred_rates: np.ndarray, true_rates: np.ndarray,
                     changes: np.ndarray) -> Dict[str, dict]:
    """按变更类型分别统计（仅变更点所在单元）。

    .. note::
       这里**不按 ``rates > 0`` 过滤**。``on_to_off`` 的含义是"该小时停止输注"，
       其当前速率为 0；若沿用正流掩码，这一类会恒为空，而"该停没停"恰是
       临床上最关键的失效模式。因此对变更点单元统一按速率（含 0）计算误差，
       并在 ``n_zero_target`` 中单独报出真实速率为 0 的单元数，便于解读。
    """
    rates = np.asarray(true_rates, dtype=float)
    pred = np.asarray(pred_rates, dtype=float)
    types = change_type_matrix(rates, changes)
    out: Dict[str, dict] = {}
    for t in CHANGE_TYPES:
        mask = (types == t)
        if mask.sum() == 0:
            out[t] = {"n_cells": 0, "mae": float("nan"), "n_zero_target": 0}
        else:
            out[t] = {
                "n_cells": int(mask.sum()),
                "mae": float(np.abs(pred[mask] - rates[mask]).mean()),
                "n_zero_target": int((rates[mask] == 0).sum()),
            }
    return out


def change_timing_error(pred_rates: np.ndarray, true_rates: np.ndarray,
                        tolerance: int = 2, **kw) -> Dict[str, float]:
    """变更时点的提前/滞后（小时，正数表示预测**滞后**）。

    对每个真实变更点，在 ±``tolerance`` 小时内寻找最近的预测变更点：
    - 找到：记入有符号的时延
    - 找不到：记入 ``unmatched``，**不按 0 计入均值**——把未命中当作"准时"
      会系统性高估时点精度（审稿人正是担心这一点）。
    """
    true = np.asarray(true_rates, dtype=float)
    pred = np.asarray(pred_rates, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"形状不一致：{true.shape} vs {pred.shape}")

    true_cp = detect_changes(true, **kw)
    pred_cp = detect_changes(pred, **kw)

    lags = []
    unmatched = 0
    total = 0
    T = true.shape[0]
    for c in range(true.shape[1]):
        for t in range(T):
            if not true_cp[t, c]:
                continue
            total += 1
            best = None
            for d in range(-tolerance, tolerance + 1):
                u = t + d
                if 0 <= u < T and pred_cp[u, c]:
                    if best is None or abs(d) < abs(best):
                        best = d
            if best is None:
                unmatched += 1
            else:
                lags.append(best)

    if total == 0:
        return {"n_true_change_points": 0, "n_matched": 0, "n_unmatched": 0,
                "mean_lag_hours": float("nan"), "median_lag_hours": float("nan"),
                "mean_abs_lag_hours": float("nan")}
    arr = np.array(lags, dtype=float) if lags else np.array([], dtype=float)
    return {
        "n_true_change_points": total,
        "n_matched": len(lags),
        "n_unmatched": unmatched,
        "match_rate": len(lags) / total,
        "mean_lag_hours": float(arr.mean()) if len(arr) else float("nan"),
        "median_lag_hours": float(np.median(arr)) if len(arr) else float("nan"),
        "mean_abs_lag_hours": float(np.abs(arr).mean()) if len(arr) else float("nan"),
    }


def patient_level_stratification(pred_rates: np.ndarray, true_rates: np.ndarray,
                                 changes: np.ndarray, patient_ids,
                                 radius: int = 1) -> Dict[str, float]:
    """按患者聚合后再平均（患者等权），与 #6 的独立性口径一致。

    直接对所有单元求平均会让窗口多的患者主导结果。
    """
    from utils.metrics import patient_then_average

    pred = np.asarray(pred_rates, dtype=float)
    true = np.asarray(true_rates, dtype=float)
    changes = np.asarray(changes, dtype=bool)
    ids = list(patient_ids)
    if not (len(ids) == len(pred) == len(true) == len(changes)):
        raise ValueError("patient_ids 与逐窗口数组长度不一致")

    positive = true > 0
    near = neighborhood_mask(changes, radius)
    out: Dict[str, float] = {}
    for name, mask in (("stable", positive & ~near),
                       ("changed", positive & changes)):
        per_row = []
        for i in range(len(pred)):
            if mask[i].sum() == 0:
                per_row.append(np.nan)
            else:
                per_row.append(float(np.abs(pred[i][mask[i]] - true[i][mask[i]]).mean()))
        out[name] = patient_then_average(np.array(per_row, dtype=float), ids)
    return out


# ==========================================================================
# v2（issue #81）：最大基数一对一匹配与变点分类指标
# ==========================================================================
#
# ``change_timing_error``（上方）用的是**最近邻**：同一个预测变点可以同时
# "命中"±tolerance 内的多个真实变点，因此它的 ``match_rate`` 是**上界**，
# 不是精确率/召回率。v2 用一对一匹配取代它。
#
# 为什么不是简单贪心
# ------------------
# "先取距离最小的" 不是最大基数匹配。反例（±1h）：真实变点在 {10, 11}，
# 预测变点在 {9, 10}。贪心先配 10→10（距离 0），剩下的 11 无处可配 → 1 个 TP；
# 最大基数是 2（10→9, 11→10）。
#
# 协议还要求最大基数中的总绝对时间偏差最小。最早匹配贪心只满足基数，
# 例如真实 {5}、预测 {4,5} 会错误选择 4。按有序前缀动态规划，同时优化
# (匹配数, -总偏差)。一维绝对距离存在不交叉最优匹配，因此前缀递推完备。
# tests/test_event_matching_offset.py 以穷举独立验证两个目标。


def match_events(true_cp: np.ndarray, pred_cp: np.ndarray, *,
                 tolerance: int = 1) -> Dict[str, object]:
    """同通道内的一对一变点匹配，返回匹配明细与基数。

    ``true_cp`` / ``pred_cp`` 为 ``(T, C)`` 布尔矩阵（``detect_changes`` 的输出）。
    每个真实变点至多命中一个预测变点，反之亦然——``max_multiplicity``
    恒为 1，用于断言"没有重复命中"。
    """
    true_cp = np.asarray(true_cp, dtype=bool)
    pred_cp = np.asarray(pred_cp, dtype=bool)
    if true_cp.shape != pred_cp.shape:
        raise ValueError(f"形状不一致：{true_cp.shape} vs {pred_cp.shape}")
    if tolerance < 0:
        raise ValueError(f"tolerance 必须 >= 0，实际 {tolerance}")

    T, C = true_cp.shape
    per_channel: Dict[int, dict] = {}
    matches = []
    n_true = n_pred = 0

    for c in range(C):
        true_ts = list(np.where(true_cp[:, c])[0])
        pred_ts = list(np.where(pred_cp[:, c])[0])
        n_true += len(true_ts)
        n_pred += len(pred_ts)

        rows,cols=len(true_ts),len(pred_ts)
        scores=[[(0,0) for _ in range(cols+1)] for _ in range(rows+1)]
        parent=np.zeros((rows+1,cols+1),dtype=np.uint8)
        for i in range(1,rows+1):
            for j in range(1,cols+1):
                options=[(scores[i-1][j],1),(scores[i][j-1],2)]
                offset=abs(int(true_ts[i-1])-int(pred_ts[j-1]))
                if offset<=tolerance:
                    count,cost=scores[i-1][j-1]
                    options.append(((count+1,cost-offset),3))
                scores[i][j],parent[i,j]=max(options,key=lambda item:item[0])
        ch_matches=[];i=rows;j=cols
        while i and j:
            action=parent[i,j]
            if action==3:
                t,u=int(true_ts[i-1]),int(pred_ts[j-1])
                ch_matches.append((t,u,u-t));i-=1;j-=1
            elif action==1:i-=1
            else:j-=1
        ch_matches.reverse()

        per_channel[c] = {
            "n_true": len(true_ts), "n_pred": len(pred_ts),
            "n_matched": len(ch_matches),
            "matches": ch_matches,
        }
        matches.extend((c, a, b, d) for a, b, d in ch_matches)

    # 每个真实/预测变点至多被用到一次
    true_used = [(m[0], m[1]) for m in matches]
    pred_used = [(m[0], m[2]) for m in matches]
    multiplicity = 0
    if true_used:
        multiplicity = max(max(true_used.count(x) for x in set(true_used)),
                           max(pred_used.count(x) for x in set(pred_used)))

    return {
        "n_true": int(n_true), "n_pred": int(n_pred), "n_matched": int(len(matches)),
        "n_unmatched_true": int(n_true - len(matches)),
        "n_unused_pred": int(n_pred - len(matches)),
        "max_multiplicity": int(multiplicity),
        "tolerance": int(tolerance),
        "matches": matches,
        "per_channel": per_channel,
    }


def event_prf(true_cp: np.ndarray, pred_cp: np.ndarray, *,
              tolerance: int = 1) -> Dict[str, object]:
    """变点的 precision / recall / F1（一对一匹配口径）。

    精确率的分母是**预测**变点数，召回率的分母是**真实**变点数；两者都用
    匹配上的数量作分子。逐通道给出，另给 macro（通道等权）与 micro（事件合并）。
    """
    m = match_events(true_cp, pred_cp, tolerance=tolerance)
    C = np.asarray(true_cp).shape[1]

    def _prf(n_true, n_pred, n_match):
        prec = (n_match / n_pred) if n_pred else float("nan")
        rec = (n_match / n_true) if n_true else float("nan")
        f1 = (2 * prec * rec / (prec + rec)
              if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0
              else float("nan"))
        return {"precision": prec, "recall": rec, "f1": f1,
                "n_true": n_true, "n_pred": n_pred, "n_matched": n_match}

    per_channel = {c: _prf(d["n_true"], d["n_pred"], d["n_matched"])
                   for c, d in m["per_channel"].items()}
    micro = _prf(m["n_true"], m["n_pred"], m["n_matched"])
    finite = [per_channel[c]["f1"] for c in per_channel
              if np.isfinite(per_channel[c]["f1"])]
    macro = {
        "precision": float(np.mean([per_channel[c]["precision"] for c in per_channel
                                    if np.isfinite(per_channel[c]["precision"])]))
        if any(np.isfinite(per_channel[c]["precision"]) for c in per_channel) else float("nan"),
        "recall": float(np.mean([per_channel[c]["recall"] for c in per_channel
                                 if np.isfinite(per_channel[c]["recall"])]))
        if any(np.isfinite(per_channel[c]["recall"]) for c in per_channel) else float("nan"),
        "f1": float(np.mean(finite)) if finite else float("nan"),
        "n_channels": len(finite),
    }
    return {"micro": micro, "macro": macro, "per_channel": per_channel,
            "tolerance": int(tolerance), "n_matched": m["n_matched"],
            "max_multiplicity": m["max_multiplicity"], "unit": "ratio"}


def change_timing_v2(true_cp: np.ndarray, pred_cp: np.ndarray, *,
                     tolerance: int = 1) -> Dict[str, float]:
    """匹配上的变点的提前/滞后分布（正数 = 预测**滞后**）。

    只统计**一对一匹配上**的事件。未匹配的真实变点记入 ``n_unmatched_true``，
    **不按 0 计入均值**——把漏检当作"准时"会系统性高估时点精度。
    """
    m = match_events(true_cp, pred_cp, tolerance=tolerance)
    deltas = np.array([d for _, _, _, d in m["matches"]], dtype=float)
    if deltas.size == 0:
        return {"n_matched": 0, "n_unmatched_true": int(m["n_unmatched_true"]),
                "mean_lag_hours": float("nan"), "median_lag_hours": float("nan"),
                "mean_abs_lag_hours": float("nan"), "tolerance": int(tolerance)}
    return {
        "n_matched": int(deltas.size),
        "n_unmatched_true": int(m["n_unmatched_true"]),
        "mean_lag_hours": float(deltas.mean()),
        "median_lag_hours": float(np.median(deltas)),
        "mean_abs_lag_hours": float(np.abs(deltas).mean()),
        "tolerance": int(tolerance),
    }


def false_change_rate(true_cp: np.ndarray, pred_cp: np.ndarray, *,
                      tolerance: int = 1, patient_hours: float = None) -> Dict[str, float]:
    """非变点小时上误报的变化率。

    与 change-point recall **配对报告**：一个"到处都在变"的预测器可以拿到很高
    的召回，只有同时看误报率才能识别它。
    """
    true_cp = np.asarray(true_cp, dtype=bool)
    pred_cp = np.asarray(pred_cp, dtype=bool)
    if true_cp.shape != pred_cp.shape:
        raise ValueError(f"形状不一致：{true_cp.shape} vs {pred_cp.shape}")

    m = match_events(true_cp, pred_cp, tolerance=tolerance)
    n_pred = m["n_pred"]
    n_false = m["n_unused_pred"]

    # 非变点的"机会数"：真值不是变点的单元数（首行恒非变点，一并计入）
    n_opportunity = int((~true_cp).sum())
    return {
        "false_change_rate": (n_false / n_opportunity) if n_opportunity else float("nan"),
        "n_false_changes": int(n_false),
        "n_pred_changes": int(n_pred),
        "n_opportunities": n_opportunity,
        "per_100_patient_hours": (100.0 * n_false / patient_hours
                                  if patient_hours else float("nan")),
        "unit": "events_per_100_patient_hours",
    }
