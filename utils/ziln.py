"""零膨胀对数正态（ZILN）的规范定义与自查（issue #15，对应 R2-4 / R4-6）。

论文把回归头描述为零膨胀对数正态分布。本模块给出该分布的**正确**数学形式，
供对照实现使用；同时提供自查函数，用来判定代码是否真的在按 ZILN 训练。

ZILN 定义
---------
令 ``p0 = σ(p0_logit)`` 为零值概率，``μ``、``σ`` 为**正值部分**的对数正态参数
（``log y | y>0 ~ N(μ, σ²)``）。则

- 零值概率：``P(Y = 0) = p0``
- 正值密度：``f(y) = (1-p0) · LogNormal(y; μ, σ)``
- 期望：``E[Y] = (1-p0) · (exp(μ + σ²/2) - 1)``   （减 1 对应 log1p 参数化）
- 负对数似然：``-log f(y)``，零值处取 ``-log p0``

**必须用 NLL 训练**，否则 ``p0`` 不会被当作概率来拟合——
这正是本仓库的现状（见下）。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def ziln_mean(p0_logit: torch.Tensor, mu: torch.Tensor,
              log_sigma: torch.Tensor) -> torch.Tensor:
    """``E[Y] = (1-p0) · (exp(μ + σ²/2) - 1)``，单位 mL/h。"""
    p0 = torch.sigmoid(p0_logit)
    sigma2 = torch.exp(2.0 * log_sigma)
    return (1.0 - p0) * (torch.exp(mu + 0.5 * sigma2) - 1.0)


def ziln_nll(p0_logit: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor,
             y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """逐元素负对数似然。

    零值样本：``-log p0``
    正值样本：``-log(1-p0) + log y + log σ + (log y - μ)²/(2σ²) + 0.5·log(2π)``
    """
    # 用 softplus 的稳定形式表示两个对数概率，**不要**先算 p0 再取对数：
    # float32 下 sigmoid(50) 恰好等于 1.0，clamp 到 1-1e-12 仍是 1.0，
    # log1p(-p0) 会得到 log(0) = -inf。
    #   log p0     = -softplus(-z)
    #   log(1-p0)  = -softplus(z)
    log_p0 = -F.softplus(-p0_logit)
    log_1m_p0 = -F.softplus(p0_logit)

    y = y.clamp(min=0.0)
    is_zero = y <= eps

    nll_zero = -log_p0

    log_y = torch.log(y.clamp(min=eps))
    sigma = torch.exp(log_sigma).clamp(min=eps)
    nll_pos = (-log_1m_p0
               + log_y
               + log_sigma
               + (log_y - mu).pow(2) / (2.0 * sigma.pow(2))
               + 0.5 * math.log(2 * math.pi))

    return torch.where(is_zero, nll_zero, nll_pos)


def check_zero_mass_is_trained(model) -> dict:
    """自查：训练路径是否用 NLL 训练了零值概率 ``p0``。

    仅做静态检查（调用点是否存在），不改动任何权重。返回可复核的结论。
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    callers = []
    for path in list(root.glob("*.py")) + list((root / "models").glob("*.py")) \
            + list((root / "utils").glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for name in ("ziln_loss_from_logits", "ziln_nll_stable", "ziln_nll"):
            for m in re.finditer(rf"\b{name}\s*\(", text):
                line_no = text[:m.start()].count("\n") + 1
                line = text.splitlines()[line_no - 1]
                if line.lstrip().startswith("def "):
                    continue
                callers.append(f"{path.name}:{line_no}")
    return {
        "nll_call_sites": sorted(set(callers)),
        "nll_is_used": bool(callers),
        "note": "为空表示 ZILN 似然只是定义、从未参与训练",
    }


def zero_penalty_gradient(pred_value: float, zero_eps: float = 1e-6) -> float:
    """``_zero_penalty`` 在给定预测值处的梯度。

    该损失为 ``mean(clamp(pred, 0, zero_eps*5))``。当 ``pred`` 超过
    ``zero_eps*5`` 时 clamp 饱和，梯度为 **0** —— 即惩罚项不起任何作用。
    """
    p = torch.tensor([float(pred_value)], requires_grad=True)
    loss = torch.clamp(p, 0.0, zero_eps * 5).mean()
    loss.backward()
    return float(p.grad.item())
