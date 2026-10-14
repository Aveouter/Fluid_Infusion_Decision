"""issue #3 / RED：CPU 冒烟测试 —— 核心模块可导入，合成输入可跑通 forward。

设计原则
--------
- 只用合成张量，**不读** data/ 下任何真实记录。
- 全部在 CPU 上运行；需要 GPU 的用例标 ``@pytest.mark.gpu``。
- 断言的是接口契约（形状、键、有限性），不是指标数值——
  指标是否达标由 #7/#12/#13 负责，这里只保证"跑得起来且形状对"。
"""

from __future__ import annotations

import importlib

import pytest
import torch

from tests import synthetic

BUILDERS = [
    ("full", "build_tcnfilm_full"),
    ("wo_temporal", "build_tcnfilm_wo_temporal"),
    ("wo_baseline", "build_tcnfilm_wo_baseline"),
    ("wo_selfattn", "build_tcnfilm_wo_selfattn"),
    ("wo_tcn", "build_tcnfilm_wo_tcn"),
    ("wo_film", "build_tcnfilm_wo_film"),
]

# 冒烟用小配置：足够跑通全部算子，又能在 CPU 上秒级完成
SMOKE = dict(
    input_dim_base=10,
    input_dim_temporal=19,
    embed_dim=32,
    num_classes=3,
    pred_length=1,
    n_transformer_layers=2,
    n_heads=4,
    tcn_layers_short=2,
    tcn_layers_long=2,
    dropout=0.1,
)
BATCH, HISTORY = 2, 8


@pytest.mark.parametrize("module_name", ["models.tcn_film", "utils.data_loader", "utils.util"])
def test_core_modules_import(module_name):
    """核心模块必须可导入，且不在导入期加载任何数据文件。"""
    module = importlib.import_module(module_name)
    assert module is not None


def test_main_module_import():
    """main.py 顶层只定义，不执行 —— 导入不应触发 argparse 或读数据。"""
    module = importlib.import_module("main")
    assert hasattr(module, "build_model")


def _build(fn_name: str):
    tcn_film = importlib.import_module("models.tcn_film")
    return getattr(tcn_film, fn_name)(**SMOKE)


@pytest.mark.parametrize("label,fn_name", BUILDERS, ids=[b[0] for b in BUILDERS])
def test_forward_all_builders(label, fn_name):
    """六个构建器（全模型 + 五个消融变体）都要能完成一次 CPU forward。"""
    model = _build(fn_name).eval()
    xT, xB = synthetic.batch_from_arrays(BATCH, HISTORY, SMOKE["input_dim_temporal"],
                                         SMOKE["input_dim_base"], seed=0)
    with torch.no_grad():
        out = model(xT, xB)

    assert isinstance(out, dict), f"{label}: forward 应返回 dict"
    assert "class_logits" in out and "flow_pred" in out, f"{label}: 缺少输出键"

    logits, flow = out["class_logits"], out["flow_pred"]
    assert logits.shape[0] == BATCH, f"{label}: batch 维丢失 {logits.shape}"
    assert logits.shape[-1] == SMOKE["num_classes"], f"{label}: 类别维错误 {logits.shape}"
    assert flow.shape[0] == BATCH and flow.shape[-1] == SMOKE["num_classes"], \
        f"{label}: flow_pred 形状错误 {flow.shape}"


@pytest.mark.parametrize("label,fn_name", BUILDERS, ids=[b[0] for b in BUILDERS])
def test_forward_outputs_finite(label, fn_name):
    """输出不得为 NaN/Inf —— 零膨胀头在 log/exp 处最容易爆掉。"""
    model = _build(fn_name).eval()
    xT, xB = synthetic.batch_from_arrays(BATCH, HISTORY, SMOKE["input_dim_temporal"],
                                         SMOKE["input_dim_base"], seed=1)
    with torch.no_grad():
        out = model(xT, xB)
    for key in ("class_logits", "flow_pred"):
        assert torch.isfinite(out[key]).all(), f"{label}: {key} 出现 NaN/Inf"


def test_ziln_extras_consistent_with_flow_pred():
    """flow_pred 必须等于由 extras 中的 (p0_logit, mu, log_sigma) 重算的期望值。

    这条把 #15「ZILN 概率模型与推理一致性」的核心不变式提前钉住：
    推理返回的速率不能与训练用的似然参数脱节。
    """
    tcn_film = importlib.import_module("models.tcn_film")
    model = _build("build_tcnfilm_full").eval()
    xT, xB = synthetic.batch_from_arrays(BATCH, HISTORY, SMOKE["input_dim_temporal"],
                                         SMOKE["input_dim_base"], seed=2)
    with torch.no_grad():
        out = model(xT, xB)

    extras = out["extras"]
    for key in ("p0_logit", "mu", "log_sigma"):
        assert key in extras, f"extras 缺少 {key}"

    recomputed = tcn_film.ziln_predict_mean(extras["p0_logit"], extras["mu"], extras["log_sigma"])
    assert torch.allclose(recomputed, out["flow_pred"], atol=1e-5), \
        "flow_pred 与 ZILN 参数重算结果不一致"


def test_backward_gradients_finite():
    """反向传播可执行且梯度有限 —— 训练可跑的最低门槛。"""
    model = _build("build_tcnfilm_full").train()
    xT, xB = synthetic.batch_from_arrays(BATCH, HISTORY, SMOKE["input_dim_temporal"],
                                         SMOKE["input_dim_base"], seed=3)
    out = model(xT, xB)
    loss = out["class_logits"].sum() + out["flow_pred"].sum()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "没有任何参数收到梯度"
    for g in grads:
        assert torch.isfinite(g).all(), "梯度中出现 NaN/Inf"


def test_param_count_is_nontrivial():
    """参数量应为正且处于合理量级 —— 防止构建器静默返回空模型。"""
    tcn_film = importlib.import_module("models.tcn_film")
    model = _build("build_tcnfilm_full")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n > 1_000, f"参数量异常偏小: {n}"


@pytest.mark.gpu
def test_forward_on_cuda():
    """GPU 路径可用性：模型与数据都能真正落在 CUDA 上并跑通一步。

    无 CUDA 的环境由 conftest 自动跳过，因此这条在 CI（CPU runner）上不会失败。
    """
    device = torch.device("cuda")
    model = _build("build_tcnfilm_full").to(device).eval()
    xT, xB = synthetic.batch_from_arrays(BATCH, HISTORY, SMOKE["input_dim_temporal"],
                                         SMOKE["input_dim_base"], seed=4)
    xT, xB = xT.to(device), xB.to(device)

    with torch.no_grad():
        out = model(xT, xB)

    assert out["flow_pred"].is_cuda, "输出未落在 GPU 上"
    assert torch.isfinite(out["flow_pred"]).all(), "GPU 前向出现 NaN/Inf"
    assert torch.cuda.get_device_name(0)


@pytest.mark.xfail(
    strict=True,
    reason="val.py 在模块顶层执行 pickle.load 并硬编码原训练机绝对路径 "
           "/baksv/CIGIT/GXN_Liuxy/...，无法被 import。属 #11/#25 的修复范围。",
)
def test_val_module_importable():
    """已知缺陷登记：分析脚本应在 __main__ 保护下才执行 IO。"""
    importlib.import_module("val")
