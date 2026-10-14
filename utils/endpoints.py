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
TOL_ABS_ML = 200.0
TOL_PCT = 0.10

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
