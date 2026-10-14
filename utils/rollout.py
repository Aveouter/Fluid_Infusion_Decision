"""教师强制与开环滚动评估（issue #11，对应 R1-2 / R3-7 / R4-5）。

为什么必须分开
--------------
模型的输入由两部分拼成（见 ``run_epoch.py``）：

    inp_T = cat(ts[:, :-P], lab[:, :-P])

其中 ``ts`` 里含**由标签派生的累计体积列 ``cum``**。因此"模型是否使用了
真实历史"这个问题有两个层面：

- **教师强制（one-step teacher-forced）**：每一步都用真实历史标签、真实 ``cum``。
  报告的所有指标默认都是这一种。它衡量"给定真实上下文，下一步预测得准不准"。
- **开环（autoregressive open loop）**：把自己的预测回填成历史，**并按预测
  重算 ``cum``**。只回填标签而不重算 ``cum`` 会偷偷把真实累计量喂回去，
  那不是真的开环——本模块显式禁止这种做法。

图 5（公式比较）与图 6（轨迹）必须以哪种模式产生，是审稿人 1 的第 2 条意见；
本模块提供两种入口，便于逐图声明。

生理特征 ``f[t]`` 不需要滚动：H=1 时决策只用 t 时刻已知的特征，
t+1 的体征在真实部署中是新观测到的，不属于"未来标签"。
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from utils import label_protocol as lp

#: 评测模式。
MODES = ("teacher_forced", "open_loop")

#: 本模块的模式名 → 评估协议 v2 的 ``eval_mode``（issue #81 §3）。
#:
#: 两种模式**绝不混排为同一个 estimand**：
#:
#: - ``teacher_forced`` → ``one_step_observed_history``：逐小时累计出的总量
#:   只能叫 **conditional cumulative volume**，不得称自主轨迹。
#: - ``open_loop`` → ``action_autoregressive_rollout``：本模块以**真实**初始
#:   标签与真实初始累计量起步（见 :func:`rollout` 的 ``cur_label`` / ``cur_cum``），
#:   因此准确的名字是 **action autoregressive rollout under true initial state**，
#:   不是 fully closed-loop。若同时使用未来真实生理特征，应升格为
#:   ``observed_physiology_conditioned_rollout``（见 eval_protocol.ESTIMANDS）。
EVAL_MODE_OF = {
    "teacher_forced": "one_step_observed_history",
    "open_loop": "action_autoregressive_rollout",
}

#: 本模块的 rollout 起步状态。写进结果与文档，避免被读成闭环仿真。
STARTS_FROM_TRUE_INITIAL_STATE = True

#: 每步体积列的顺序：三类各自体积 + 总量（与 data_loader 一致）。
INST_COLS = (0, 1, 2, 3)


def inst_from_label(label_row: np.ndarray, step_hours: float = 1.0) -> np.ndarray:
    """由一步标签算该步体积 ``(4,)``：三类体积与总量。"""
    row = np.asarray(label_row, dtype=np.float32).reshape(-1)
    if row.shape[0] != 2 * lp.NUM_TYPES:
        raise ValueError(f"标签行长度应为 {2 * lp.NUM_TYPES}，实际 {row.shape[0]}")
    bits, rates = row[lp.BIT_SLICE], row[lp.RATE_SLICE]
    v = bits * rates * float(step_hours)
    return np.concatenate([v, [v.sum()]]).astype(np.float32)


def cum_from_labels(labels: np.ndarray, step_hours: float = 1.0) -> np.ndarray:
    """由整段标签算逐步累计体积 ``(T, 4)``。"""
    labels = np.asarray(labels, dtype=np.float32)
    if len(labels) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    inst = np.stack([inst_from_label(r, step_hours) for r in labels], axis=0)
    return np.cumsum(inst, axis=0).astype(np.float32)


def _model_input(feature_row: np.ndarray, cum_row: np.ndarray,
                 label_row: np.ndarray) -> np.ndarray:
    """拼出单步模型输入，列序与 ``run_epoch`` 一致：``[ts(9+4), lab(6)]``。"""
    return np.concatenate([feature_row, cum_row, label_row]).astype(np.float32)


def rollout(model_predict: Callable[[np.ndarray], np.ndarray],
            features: np.ndarray,
            labels: np.ndarray,
            mode: str = "teacher_forced",
            step_hours: float = 1.0,
            start: int = 0,
            horizon: int = None) -> Dict[str, np.ndarray]:
    """在一条序列上产出一小时一步的预测。

    参数
    ----
    model_predict : 接受 ``(19,)`` 输入、返回 ``(6,)`` 标签形式预测的**纯函数**。
        用回调而非直接传模型，便于用合成函数做单元测试。
    features : ``(T, 9)`` 生理/时间特征，不含标签派生列。
    labels   : ``(T, 6)`` 真实标签。
    mode     : ``teacher_forced`` 或 ``open_loop``。

    返回 ``{"pred": (T-1-start, 6), "target": (T-1-start, 6), "cum_used": ...,
    "mode": ...}``；``cum_used`` 供核对开环时回填的是预测累计量。
    """
    if mode not in MODES:
        raise ValueError(f"mode 必须是 {MODES} 之一，实际 {mode}")
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    T = len(labels)
    if len(features) != T:
        raise ValueError(f"features 与 labels 长度不一致：{len(features)} vs {T}")

    end = T - 1 if horizon is None else min(T - 1, start + horizon)
    if start < 0 or start >= T:
        raise ValueError(f"start {start} 超出 [0, {T - 1}]")

    true_cum = cum_from_labels(labels, step_hours)

    preds, targets, cums = [], [], []
    # 教师强制：全程使用真实历史；开环：历史标签与累计量都来自预测
    cur_label = labels[start].copy()
    cur_cum = true_cum[start].copy()

    for t in range(start, end):
        inp = _model_input(features[t], cur_cum, cur_label)
        p = np.asarray(model_predict(inp), dtype=np.float32).reshape(-1)
        if p.shape[0] != 2 * lp.NUM_TYPES:
            raise ValueError(f"预测长度应为 {2 * lp.NUM_TYPES}，实际 {p.shape[0]}")
        p = lp.normalize(p[None, :])[0]

        preds.append(p)
        targets.append(labels[t + 1])
        cums.append(cur_cum.copy())

        if mode == "teacher_forced":
            cur_label = labels[t + 1].copy()
            cur_cum = true_cum[t + 1].copy()
        else:
            # 开环：回填预测，且累计量按预测步进 —— 不得偷用真实 cum
            cur_label = p
            cur_cum = (cur_cum + inst_from_label(p, step_hours)).astype(np.float32)

    return {
        "pred": np.stack(preds, axis=0) if preds else np.zeros((0, 6), np.float32),
        "target": np.stack(targets, axis=0) if targets else np.zeros((0, 6), np.float32),
        "cum_used": np.stack(cums, axis=0) if cums else np.zeros((0, 4), np.float32),
        "mode": np.array([mode]),
    }


def compare_modes(model_predict, features, labels, step_hours: float = 1.0,
                  start: int = 0, horizon: int = None) -> Dict[str, dict]:
    """同一序列上跑两种模式并汇总误差，便于量化误差累积。"""
    out = {}
    for mode in MODES:
        r = rollout(model_predict, features, labels, mode=mode,
                    step_hours=step_hours, start=start, horizon=horizon)
        tgt, pred = r["target"], r["pred"]
        pos = tgt[:, lp.RATE_SLICE] > 0
        mae = (float(np.abs(pred[:, lp.RATE_SLICE][pos]
                            - tgt[:, lp.RATE_SLICE][pos]).mean())
               if pos.sum() else float("nan"))
        out[mode] = {
            "n_steps": int(len(pred)),
            "mae_positive_flow": mae,
            "cum_final_true": float(cum_from_labels(labels, step_hours)[-1, 3]),
            "cum_final_used": float(r["cum_used"][-1, 3]) if len(r["cum_used"]) else float("nan"),
        }
    return out
