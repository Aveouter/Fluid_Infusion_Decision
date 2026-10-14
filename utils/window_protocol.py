"""预测窗口与时间截断协议（v1）。

把「历史窗口 / 预测目标」的切分从训练循环里的内联切片提升为显式定义，
使其可以被测试钉住。见 issue #5。

约定
----
- 一个样本窗口长度为 ``W = history_length + pred_length``。
- **输入（历史）** 占窗口的前 ``W - pred_length`` 步，即 ``[0, W-P)``；
  决策时刻（cutoff）是历史的最后一步，索引 ``W - P - 1``。
- **目标** 占窗口的后 ``P`` 步，即 ``[W-P, W)``，严格晚于 cutoff。
- 因此历史与目标**不相交**，且目标的第一步恰为 ``cutoff + 1``。

对应到 ``run_epoch.py`` 的既有实现：
``inp_T = cat(ts[:, :-P], lab[:, :-P])``（历史），``y = lab[:, -P:]``（目标）。
两者从同一窗口切出，顺序不能颠倒。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

PROTOCOL_VERSION = "1.0"


@dataclass(frozen=True)
class WindowSpec:
    """一个样本的时间切分。所有索引均为窗口内相对下标。"""

    window_len: int
    history_length: int
    pred_length: int
    stride: int = 1

    @property
    def history_slice(self) -> slice:
        return slice(0, self.history_length)

    @property
    def target_slice(self) -> slice:
        return slice(self.history_length, self.window_len)

    @property
    def cutoff(self) -> int:
        """决策时刻：历史最后一步的下标。"""
        return self.history_length - 1

    @property
    def target_start(self) -> int:
        return self.history_length

    def absolute(self, window_start: int) -> Tuple[int, int]:
        """返回 (cutoff_abs, target_end_abs)，用于换算到序列上的绝对下标。"""
        return window_start + self.cutoff, window_start + self.window_len - 1


def make_window_spec(history_length: int, pred_length: int, stride: int = 1) -> WindowSpec:
    """按协议构造窗口切分，并校验参数合法。"""
    H, P, S = int(history_length), int(pred_length), int(stride)
    if H < 1:
        raise ValueError(f"history_length 必须 >= 1，实际 {H}")
    if P < 1:
        raise ValueError(f"pred_length 必须 >= 1，实际 {P}")
    if S < 1:
        raise ValueError(f"stride 必须 >= 1，实际 {S}")
    return WindowSpec(window_len=H + P, history_length=H, pred_length=P, stride=S)


def window_starts(series_len: int, spec: WindowSpec):
    """产出该序列上所有窗口的起点。

    上界取 ``series_len - window_len + 1``，保证最后一个窗口的右端不越界，
    即目标最后一步的下标 ``start + window_len - 1 <= series_len - 1``。
    """
    if series_len < spec.window_len:
        return []
    return list(range(0, series_len - spec.window_len + 1, spec.stride))


def split_history_target(ts, lab, pred_length: int):
    """按协议把 (ts, lab) 窗口切成模型输入与监督目标。

    行为与 ``run_epoch.py`` 的既有实现完全一致，只是把规则显式化以便测试：

    - 输入 = ``cat(ts[:, :-P], lab[:, :-P])``（窗口前 W-P 步）
    - 目标 = ``lab[:, -P:]``（窗口后 P 步）

    返回 ``(model_input, target_tensor)``。
    """
    import torch

    P = int(pred_length)
    if P < 1:
        raise ValueError(f"pred_length 必须 >= 1，实际 {P}")
    if ts.dim() != 3 or lab.dim() != 3:
        raise ValueError("ts 与 lab 都应为 (B, W, D) 形状")
    if ts.size(1) != lab.size(1):
        raise ValueError(
            f"ts 与 lab 的窗口长度不一致：{ts.size(1)} vs {lab.size(1)}"
        )
    if ts.size(1) <= P:
        raise ValueError(
            f"窗口长度 {ts.size(1)} 必须大于 pred_length {P}，否则历史为空"
        )

    model_input = torch.cat((ts[:, :-P, :], lab[:, :-P, :]), dim=-1)
    target = lab[:, -P:, :]
    return model_input, target


def assert_causal_alignment(ts, lab, pred_length: int) -> None:
    """断言窗口的历史与目标不相交，且目标严格晚于 cutoff。

    这是 #5 的核心不变量：任何通过 ``split_history_target`` 构造的样本，
    其监督目标都不能出现在模型输入里。
    """
    P = int(pred_length)
    W = ts.size(1)
    spec = make_window_spec(W - P, P)
    hist = spec.history_slice
    tgt = spec.target_slice
    if hist.stop > tgt.start:
        raise AssertionError(
            f"历史 {hist} 与目标 {tgt} 相交，目标泄漏进输入"
        )
    if tgt.start != spec.cutoff + 1:
        raise AssertionError(
            f"目标起点 {tgt.start} 不等于 cutoff+1 ({spec.cutoff + 1})"
        )
