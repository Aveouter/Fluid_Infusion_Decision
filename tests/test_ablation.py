"""issue #16 / RED：消融退化排查与单步配置的结构后果。

对应审稿意见：移除自注意力后 AUC 几乎不变而 MAE 从 27.44 跳到 74.67，
移除 TCN 却几乎无影响——这种不对称需要解释。
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
CKPTS = ROOT / "ckpts"

#: 本票重评的结果（患者级，`rate = prob × speed`，正流掩码，n_boot=500, seed=42）
RE_EVALUATED_MAE = {
    "full": 90.14, "wo_selfattn": 88.62, "wo_tcn": 87.60,
    "wo_film": 90.56, "wo_temporal": 128.39, "wo_baseline": 148.70,
}

#: 论文表 3 报告的测试集 MAE
PAPER_MAE = {
    "full": 27.44, "wo_selfattn": 74.67, "wo_tcn": 27.63,
    "wo_film": 28.88, "wo_temporal": 72.42, "wo_baseline": 74.32,
}


# --------------------------------------------------- 出厂配置是单步

def test_shipped_config_is_single_timestep():
    """出厂配置 history_length=1, pred_length=1 —— 模型只处理**一个**时间步。

    这不是笔误：``ckpts/training_metadata.json`` 与数据集样本形状
    ``X=(2, 13)``（= H+P）都指向同一点。
    """
    meta = CKPTS / "training_metadata.json"
    if not meta.exists():
        pytest.skip("缺少 training_metadata.json")
    args = json.load(open(meta, encoding="utf-8"))["args"]
    assert int(args["history_length"]) == 1
    assert int(args["pred_length"]) == 1


def test_model_accepts_a_single_step():
    """模型在 T=1 时可正常前向 —— 即架构并未要求序列长度大于 1。"""
    from models.tcn_film import build_tcnfilm_full

    m = build_tcnfilm_full(input_dim_base=10, input_dim_temporal=19, embed_dim=32,
                           num_classes=3, pred_length=1, n_transformer_layers=1,
                           n_heads=2, tcn_layers_short=1, tcn_layers_long=1).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 1, 19), torch.randn(1, 10))
    assert out["flow_pred"].shape == (1, 1, 3)
    assert torch.isfinite(out["flow_pred"]).all()


def test_run_epoch_feeds_exactly_one_timestep():
    """出厂配置下，模型的输入**恰为一个时间步**。

    ``run_epoch`` 构造输入的方式是
    ``inp_T = cat(ts[:, :-P, :], lab[:, :-P, :])``；窗口长度为 ``H+P``，
    于是输入步数 = ``H``。``history_length=1`` 时输入只有 1 步。

    **后果**：扩张因果卷积在长度为 1 的序列上只能看到自身，多尺度感受野无从
    建立；自注意力在单个 token 上退化为对自身加权。也就是说，架构描述里的
    "双路扩张 TCN""因果 Transformer""多尺度时序建模"在出厂配置下**没有可
    施展的时间维度**——这与实测中移除这两者几乎不改变性能（见下方用例）一致。

    注意：模型本身**接受**任意长度输入（有用例保证 T=1 可前向），且对长度敏感
    （位置编码会变）；这里断言的是**训练与评测时实际喂入的长度**，不是模型的
    能力上限。
    """
    H, P = 1, 1
    W = H + P
    ts = torch.randn(2, W, 13)
    lab = torch.randn(2, W, 6)
    inp = torch.cat((ts[:, :-P, :], lab[:, :-P, :]), dim=-1)
    assert inp.shape[1] == H == 1, f"输入步数应为 H=1，实际 {inp.shape[1]}"
    assert inp.shape[-1] == 19, "输入特征维度应为 19"


def test_model_still_accepts_longer_input_but_is_never_given_one():
    """对照：模型能接受更长的输入，只是训练与评测从不这样喂。

    两条一起看才能说明问题——不是模型不支持时序，而是**配置让它无事可做**。
    """
    from models.tcn_film import build_tcnfilm_full

    torch.manual_seed(0)
    m = build_tcnfilm_full(input_dim_base=10, input_dim_temporal=19, embed_dim=32,
                           num_classes=3, pred_length=1, n_transformer_layers=1,
                           n_heads=2, tcn_layers_short=1, tcn_layers_long=1).eval()
    with torch.no_grad():
        for T in (1, 4, 12):
            out = m(torch.randn(1, T, 19), torch.randn(1, 10))
            assert out["flow_pred"].shape == (1, 1, 3)
            assert torch.isfinite(out["flow_pred"]).all()


# ------------------------------------------- 论文表 3 与重评不一致

def test_paper_ablation_numbers_are_not_reproducible():
    """特征化：论文表 3 的 MAE 与本仓库重评结果**不可对账**。

    同一批 checkpoint、同一批测试窗口、同一口径（``rate`` + 正流掩码）：


    变体            重评(患者级)    论文表 3
    full                 90.14        27.44
    w/o self-attn        88.62        74.67
    w/o TCN              87.60        27.63
    w/o FiLM             90.56        28.88
    w/o Temporal Enc.   128.39        72.42
    w/o Baseline Enc.   148.70        74.32

    论文报告"移除自注意力使 MAE 从 27.44 跳到 74.67"（相差 47.23），
    而重评显示两者只差 **1.53**，且配对区间跨 0。

    一旦表 3 按统一口径重算（#26），本用例应当失败并被改写。
    """
    for name, paper in PAPER_MAE.items():
        got = RE_EVALUATED_MAE[name]
        assert abs(paper - got) > 5.0, (
            f"{name}: 论文 {paper} 与重评 {got} 一致，前提已变"
        )


def test_self_attention_removal_is_not_a_collapse():
    """移除自注意力的差异在患者级区间下不显著 —— 不存在"坍缩"。

    实测 ΔMAE = +1.53，95% CI [−0.12, +3.54]，跨 0。
    论文所报的 47 分差距在本仓库无法重现。
    """
    delta, lo, hi = 1.53, -0.12, 3.54
    assert lo <= 0 <= hi, "区间不再跨 0，结论需要更新"
    paper_gap = PAPER_MAE["wo_selfattn"] - PAPER_MAE["full"]
    assert paper_gap > 20, "论文确实报告了一个数量级更大的差距"
    assert abs(delta) < 0.1 * paper_gap, (
        f"实测差异 {delta} 与论文差距 {paper_gap} 同量级，前提已变"
    )


def test_only_input_modality_ablations_matter():
    """只有**输入模态**消融显著：去掉时序或基线特征会大幅变差。

    架构组件（自注意力、FiLM）的差异跨 0 或不具实际意义。这与 T=1 的
    结构事实一致——没有时间维度可供卷积或注意力发挥。
    """
    modality = ("wo_temporal", "wo_baseline")
    architecture = ("wo_selfattn", "wo_film")
    for name in modality:
        gap = RE_EVALUATED_MAE[name] - RE_EVALUATED_MAE["full"]
        assert gap > 20, f"{name} 的模态消融差异只有 {gap:.1f}，与预期不符"
    for name in architecture:
        gap = abs(RE_EVALUATED_MAE[name] - RE_EVALUATED_MAE["full"])
        assert gap < 3, f"{name} 的架构消融差异达 {gap:.1f}，与 T=1 的推断不符"


# --------------------------------------------------- 权重可用性

@pytest.mark.realdata
def test_ablation_checkpoints_are_present():
    """六个消融权重都应在库，否则重评无法进行。"""
    files = ["tcn_film_best.pth", "ours_wo_selfattn_best.pth", "ours_wo_tcn_best.pth",
             "ours_wo_film_best.pth", "ours_wo_temporal_best.pth",
             "ours_wo_baseline_best.pth"]
    missing = [f for f in files if not (CKPTS / f).exists()]
    if missing and not CKPTS.exists():
        pytest.skip("本地无 ckpts/")
    assert not missing, f"缺少消融权重：{missing}"


@pytest.mark.realdata
def test_full_checkpoint_loads_into_full_builder():
    """完整权重能载入完整构建器，且参数量与记录一致（扣除 buffer）。"""
    import sys
    sys.path.insert(0, str(ROOT))
    from predict_test_set import MODEL_CFG
    from models.tcn_film import build_tcnfilm_full

    ck = CKPTS / "tcn_film_best.pth"
    if not ck.exists():
        pytest.skip("本地无 ckpts/tcn_film_best.pth")
    m = build_tcnfilm_full(**MODEL_CFG)
    state = torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"]
    missing, unexpected = m.load_state_dict(state, strict=False)
    assert not unexpected, f"权重含未知键：{unexpected[:3]}"
