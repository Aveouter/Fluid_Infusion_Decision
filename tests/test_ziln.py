"""issue #15 / RED：ZILN 概率模型、损失与推理的一致性。

论文把回归头描述为「零膨胀对数正态」。本票要回答的是：**代码是否真的
按 ZILN 训练**。答案见下方特征化用例。
"""

from __future__ import annotations

import math

import pytest
import torch

from utils import ziln as Z


# ------------------------------------------------------------- 数学正确性

def test_ziln_mean_matches_closed_form():
    """E[Y] = (1-p0)·(exp(μ + σ²/2) - 1)，与手算一致。"""
    p0_logit = torch.tensor([0.0])          # p0 = 0.5
    mu = torch.tensor([math.log(100.0)])
    log_sigma = torch.tensor([0.0])         # σ = 1
    got = Z.ziln_mean(p0_logit, mu, log_sigma)
    want = (1 - 0.5) * (math.exp(math.log(100.0) + 0.5) - 1.0)
    assert got.item() == pytest.approx(want, rel=1e-5)


def test_ziln_mean_is_zero_when_zero_probability_is_one():
    """p0 → 1 时期望必须 → 0。这是零膨胀的意义所在。"""
    got = Z.ziln_mean(torch.tensor([50.0]), torch.tensor([5.0]), torch.tensor([0.0]))
    assert got.item() == pytest.approx(0.0, abs=1e-6)


def test_ziln_mean_is_positive_and_bounded():
    got = Z.ziln_mean(torch.tensor([-50.0]), torch.tensor([5.0]), torch.tensor([0.5]))
    assert got.item() > 0


def test_ziln_nll_zero_sample_uses_minus_log_p0():
    """零值样本的 NLL 应为 -log p0。"""
    p0_logit = torch.tensor([0.0])          # p0 = 0.5
    nll = Z.ziln_nll(p0_logit, torch.tensor([3.0]), torch.tensor([0.0]),
                     torch.tensor([0.0]))
    assert nll.item() == pytest.approx(-math.log(0.5), rel=1e-5)


def test_ziln_nll_positive_sample_is_finite_and_favours_the_truth():
    """正值样本：真实参数下的 NLL 应低于错配参数下的 NLL。"""
    y = torch.tensor([100.0])
    mu_true = torch.tensor([math.log(100.0)])
    good = Z.ziln_nll(torch.tensor([-3.0]), mu_true, torch.tensor([-1.0]), y)
    bad = Z.ziln_nll(torch.tensor([-3.0]), torch.tensor([math.log(500.0)]),
                     torch.tensor([-1.0]), y)
    assert torch.isfinite(good) and torch.isfinite(bad)
    assert good < bad


def test_ziln_nll_is_finite_at_extremes():
    """极端参数不得产生 NaN/Inf —— 这是原实现最容易爆的地方。"""
    for p0_logit in (-50.0, 0.0, 50.0):
        for mu in (-10.0, 0.0, 10.0):
            for ls in (-6.0, 0.0, 6.0):
                for yv in (0.0, 1e-8, 1.0, 5000.0):
                    nll = Z.ziln_nll(torch.tensor([p0_logit]), torch.tensor([mu]),
                                     torch.tensor([ls]), torch.tensor([yv]))
                    assert torch.isfinite(nll), (p0_logit, mu, ls, yv, nll)


def test_zero_mass_receives_gradient_through_nll():
    """NLL 必须把梯度传到 p0 —— 否则零值概率根本没被训练。"""
    p0_logit = torch.tensor([0.0], requires_grad=True)
    nll = Z.ziln_nll(p0_logit, torch.tensor([3.0]), torch.tensor([0.0]),
                     torch.tensor([0.0]))
    nll.backward()
    assert p0_logit.grad is not None
    assert abs(p0_logit.grad.item()) > 1e-6, "零值样本未对 p0 产生梯度"


# ------------------------------------------- 本仓库是否真的按 ZILN 训练

def test_ziln_nll_has_no_call_site():
    """特征化：ZILN 似然在本仓库中**只有定义、没有调用点**。

    这不是期望行为。论文声称回归头是零膨胀对数正态，而
    ``models/Flim.py`` 的 ``ziln_loss_from_logits`` 与 ``utils/util.py`` 的
    ``ziln_nll_stable`` 都没有被任何地方调用，训练用的是对 ``speed`` 的 L1。

    一旦训练改为真正使用 NLL，本用例应当失败并被改写。
    """
    info = Z.check_zero_mass_is_trained(None)
    assert info["nll_call_sites"] == [], (
        f"ZILN 似然出现了调用点 {info['nll_call_sites']}，请更新本用例与 #15 的结论"
    )
    assert info["nll_is_used"] is False


def test_training_uses_l1_on_positive_cells_only():
    """特征化：训练用的回归损失是 L1，且只作用于正流单元。

    ``_reg_l1_pos_only`` 的掩码是 ``target > 0``，因此零流小时**完全不参与
    回归损失**；唯一涉及零流的是 ``_zero_penalty``，而它是死损失（见下一条）。
    两者合起来解释了为什么模型从不输出 0。
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parent.parent.joinpath(
        "run_epoch.py").read_text(encoding="utf-8", errors="replace")
    assert "def _reg_l1_pos_only" in src
    assert "mask = (target > zero_eps)" in src, "正流掩码不见了"
    assert "L_reg = _reg_l1_pos_only(speed, y_flow" in src, "回归损失不再是 speed 的 L1"
    assert "ziln_nll" not in src and "ziln_loss" not in src


def test_zero_penalty_has_no_gradient_at_realistic_predictions():
    """特征化：``_zero_penalty`` 在真实预测范围内梯度恒为 0。

    该损失是 ``mean(clamp(pred, 0, 5e-6))``。实测模型输出 ``speed ≥ 19``，
    远高于 5e-6，clamp 饱和 → 损失恒为常数 → **梯度精确为 0**。
    也就是说"零流惩罚项"对训练没有任何作用，尽管它在总损失里占着一项。
    """
    for pred in (19.0, 100.0, 888.0):
        g = Z.zero_penalty_gradient(pred)
        assert g == 0.0, f"pred={pred} 时零惩罚项竟有梯度 {g}"

    # 反向对照：低于饱和阈值时确实有梯度，确认用例不是恒真
    assert Z.zero_penalty_gradient(1e-9) != 0.0


def test_head_output_range_cannot_reach_zero():
    """特征化：`ziln_mean` 在实测参数范围内不可能接近 0。

    实测 ``flow_pred``（即 ZILN 均值）取值范围 [19.0, 887.8]。
    由于 ``(1-p0) > 0`` 且 ``exp(μ+σ²/2) - 1 > 0``，乘积恒正；
    要让均值 < 1 需要 ``p0 > 0.99``，而这只有 NLL 才会训练出来。
    """
    p0_logit = torch.tensor([0.0])           # 未训练的初始 p0
    mean = Z.ziln_mean(p0_logit, torch.tensor([math.log(150.0)]),
                       torch.tensor([0.0]))
    assert mean.item() > 1.0

    # 只有把 p0 推向 1 才能压到 0 附近——而那需要 NLL 的梯度
    near_one = Z.ziln_mean(torch.tensor([10.0]), torch.tensor([math.log(150.0)]),
                           torch.tensor([0.0]))
    assert near_one.item() < 0.1
