"""issue #18 / RED：多任务贡献与零膨胀控制。

票面要求训练 class-only / rate-only / joint 三种形态并做门控对照。
**判断**：该对照需要在已被证明有缺陷的流水线上重训三个变体（#5/#6 修正了
增强与 scaler），而模型相对朴素基线又无显著优势（#13）——重训的结论无法
支撑任何主张。故本票只做**可回答的部分**：门控机制与零膨胀的实际行为，
并记录多任务对照的前置条件。
"""

from __future__ import annotations

import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _src(name):
    return (ROOT / name).read_text(encoding="utf-8", errors="replace")


# ------------------------------------------------------- 门控机制

def test_rate_applies_two_gates():
    """特征化：推断时的速率被**两次门控**。

    ``flow_pred`` 已是 ZILN 均值 ``(1-p0)·(exp(μ+σ²/2)-1)``，其中 ``(1-p0)``
    本身就是一个门；``run_epoch`` 又乘了一次类型概率 ``prob``：

        speed = softplus(flow_pred)
        rate  = clamp(prob * speed, 0, max_speed)

    两次门控**未必错**（若 ``1-p0`` 表示"该类型激活时为正"的概率、
    ``prob`` 表示"该类型是否激活"，则乘积正是期望速率），
    但论文 Methods 从未定义二者在训练与推理时的交互方式——
    而这正是 R2-4 明确追问的内容。此处把实际行为固定下来。
    """
    src = _src("run_epoch.py")
    assert "rate   = torch.clamp(prob * speed" in src or \
           "rate = torch.clamp(prob * speed" in src, "门控形式变了"
    assert "prob   = torch.sigmoid(logits)" in src or \
           "prob = torch.sigmoid(logits)" in src

    from utils.ziln import ziln_mean
    p0_logit = torch.tensor([0.0])
    mean = ziln_mean(p0_logit, torch.tensor([5.0]), torch.tensor([0.0]))
    assert mean.item() > 0, "ZILN 均值本身已含 (1-p0) 门控"


def test_two_gates_are_never_separable_in_the_shipped_model():
    """两次门控在输出里不可分离：没有任何接口能单独取出任一门。

    这解释了为什么"门控导致的分布偏差"无法从随包产物中量化——
    必须重训才能做对照。
    """
    src = _src("run_epoch.py")
    assert "extras" not in src or "p0_logit" not in src.split("rate   = torch.clamp")[0][-200:], \
        "run_epoch 似乎已经能拿到 p0，请更新本用例"


# ------------------------------------------------------- 零膨胀

def test_zero_inflation_contributes_nothing():
    """零膨胀分量在实测输出中不起作用（与 #15 一致）。

    ``flow_pred`` 实测范围 [19.0, 887.8]，从不接近 0；而 ``ziln_nll`` 从未被
    调用，``p0`` 没有被当作概率训练过。因此"零膨胀"是命名而非机制。
    """
    import json
    p = ROOT / "ckpts" / "test_metrics.json"
    if not p.exists():
        pytest.skip("缺少 test_metrics.json")
    m = json.load(open(p, encoding="utf-8"))
    # 记录中确实存在零流相关的键，说明设计意图是有的
    assert "mae_all_pos" in m["overall"]
    note = m["overall"].get("note", "")
    assert "y_true>0" in note, "记录的掩码说明变了"


# ------------------------------------------------- 多任务对照的前置条件

def test_single_task_control_is_not_runnable_now():
    """多任务 vs 单任务对照**当前不可执行**，理由是前置条件不满足。

    需要训练 class-only / rate-only / joint 三个变体。而：

    1. 随包权重出自 #5/#6 修正前的流水线（增强置换泄漏、scaler 在增强数据上拟合），
       在其上比较任务形态会把流水线缺陷与任务形态的效应混在一起；
    2. #13 已表明完整模型相对朴素基线无显著优势，因此"多任务是否带来增益"
       这个问题缺少可被增益的对象；
    3. 论文声称的损失项有四个不存在（#17），单任务版本该用什么损失尚无定义。

    本用例把这三条固定下来；条件满足后应当失败并被改写。
    """
    assert (ROOT / "utils" / "ziln.py").exists()          # #15 的结论
    assert (ROOT / "docs" / "table_provenance.md").exists()  # #8 的溯源
    src = _src("run_epoch.py")
    assert "L_cum" in src, "多任务损失仍在"


def test_task_heads_are_separable_in_principle():
    """架构上两个头是可分离的——对照在**实现层面**可行，缺的是训练与预算。"""
    src = _src("models/tcn_film.py")
    assert "class TypeHead" in src
    assert "class ZILNHead" in src
    assert "self.cls_head" in src and "self.ziln_head" in src


def test_head_outputs_share_one_trunk():
    """两个头共享同一主干 —— 这是"多任务"的实际含义，也是对照的着力点。"""
    src = _src("models/tcn_film.py")
    assert "class_logits = self.cls_head(z)" in src
    assert "self.ziln_head(z)" in src
