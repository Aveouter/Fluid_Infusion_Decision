"""issue #17 / RED：论文声称的损失项与调度器是否真实存在。

票面要求"逐项消融 L_cum/L_pos/L_cp/L_tv/L_pre 与 Tversky/Focal/BCE、scheduler"。
**在做消融之前必须先确认这些项存在** —— 本文件正是这个前提检查。

结论：论文 Methods 描述的决策优先目标与实现严重不符。
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCES = [p for p in list(ROOT.glob("*.py")) + list((ROOT / "models").glob("*.py"))
           + list((ROOT / "utils").glob("*.py"))]


def _occurrences(needle: str) -> list:
    """统计标识符在非测试源码中的出现位置（排除定义行本身）。"""
    hits = []
    pat = re.compile(rf"\b{re.escape(needle)}\b")
    for path in SOURCES:
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if pat.search(line):
                hits.append(f"{path.name}:{lineno}")
    return hits


# ------------------------------------ 论文声称的五项损失项是否存在

@pytest.mark.parametrize("term", ["L_cp", "L_tv", "L_pre"])
def test_claimed_regularisation_terms_do_not_exist(term):
    """特征化：``L_cp`` / ``L_tv`` / ``L_pre`` 在代码中**完全不存在**。

    论文 Methods 两次声明目标函数包含 change-point 边界惩罚、total-variation
    平滑与 preconditioning 项，但这三个标识符全库零出现——既没实现，也没有
    对应的 ``lambda_*`` 权重。

    一旦实现，本用例应当失败并被改写；在那之前，票面要求的"逐项消融"无从进行。
    """
    assert _occurrences(term) == [], f"{term} 竟然存在，请更新 #17 的结论"


def test_l_pos_is_not_the_claimed_log_huber():
    """特征化：``L_pos`` 不存在；实际的正流回归损失是 **L1**。

    论文写的是"positive-only instantaneous rate supervision（log 变换的 Huber
    损失）"，而 ``run_epoch._reg_l1_pos_only`` 用的是 ``F.l1_loss``，
    既不是 log 变换也不是 Huber。
    """
    assert _occurrences("L_pos") == []
    src = (ROOT / "run_epoch.py").read_text(encoding="utf-8", errors="replace")
    assert "def _reg_l1_pos_only" in src
    assert "F.l1_loss(pred[mask], target[mask])" in src, "正流损失不再是 L1"
    assert "huber" not in src.split("def _reg_l1_pos_only")[1][:400].lower(), \
        "正流损失改成了 Huber，需更新本用例"


def test_l_cum_exists_and_is_the_only_surviving_term():
    """``L_cum`` 是五项里唯一真实存在的（对累计量做 Huber）。"""
    assert _occurrences("L_cum"), "L_cum 不见了"
    src = (ROOT / "run_epoch.py").read_text(encoding="utf-8", errors="replace")
    assert "L_cum" in src and "huber(cum_pred - cum_true" in src


# ---------------------------------------- 分类损失：三方不一致

def test_tversky_is_not_implemented():
    """特征化：论文声称分类损失用 **Tversky**，但代码里没有任何 Tversky 实现。

    论文 Methods 两处提及 Tversky 并引用了 Tversky 1977。全库检索零命中。
    """
    hits = _occurrences("Tversky") + _occurrences("tversky")
    assert hits == [], f"出现了 Tversky 实现 {hits}，请更新本用例"


def test_actual_classification_loss_is_bce():
    """实际用的是 BCE —— 与论文所述、与配置所记**都不一致**。"""
    src = (ROOT / "run_epoch.py").read_text(encoding="utf-8", errors="replace")
    assert "def _bce_logits" in src
    assert "L_cls = _bce_logits(logits, y_type.float()" in src, \
        "分类损失不再是 BCE，三方不一致的结论需要更新"


def test_use_focal_flag_is_never_read():
    """特征化：``--use_focal`` 默认开启、写进 metadata，却从未被读取。

    与 ``--min_thresholds`` / ``--tol_abs_ml`` / ``--temp_scale`` 属同一类
    死配置：参数存在、被记录、不生效。``utils/Focal_loss.py`` 也从未被 import。
    """
    assert _occurrences("use_focal"), "use_focal 参数声明不见了"
    readers = [h for h in _occurrences("use_focal") if not h.startswith("main.py")]
    assert readers == [], f"use_focal 被读取了（{readers}），请更新本用例"

    importers = [h for h in _occurrences("Focal_loss") if not h.startswith("main.py")]
    assert importers == [], "Focal_loss 被 import 了"


def test_config_records_focal_as_enabled():
    """配置记录 ``use_focal: True`` —— 与实际行为不符，是会误导人的记录。"""
    meta = ROOT / "ckpts" / "training_metadata.json"
    if not meta.exists():
        pytest.skip("缺少 training_metadata.json")
    args = json.load(open(meta, encoding="utf-8"))["args"]
    assert args.get("use_focal") is True or args.get("use_focal") == "True", \
        "配置不再记录 use_focal=True，需更新三方不一致的描述"


# ------------------------------------------------- 调度器

def test_scheduler_never_fired_in_the_shipped_run():
    """特征化：出厂运行用的是 StepLR（lr_step=100），而训练只跑了 16 epoch。

    学习率全程恒为 1e-4 —— 论文提到的"cosine warm restart"稳健性分析
    在随包运行中**从未启用**。
    """
    meta = ROOT / "ckpts" / "training_metadata.json"
    hist = ROOT / "ckpts" / "training_history.json"
    if not (meta.exists() and hist.exists()):
        pytest.skip("缺少训练记录")
    args = json.load(open(meta, encoding="utf-8"))["args"]
    h = json.load(open(hist, encoding="utf-8"))

    assert args["lr_scheduler"] == "step"
    assert int(args["lr_step"]) == 100
    assert len(h["epoch"]) < 100, "训练轮数已达到 lr_step，调度器会触发"
    assert len(set(h["learning_rate"])) == 1, \
        f"学习率发生了变化：{sorted(set(h['learning_rate']))}"


def test_manuscript_claims_tversky_and_temporal_terms():
    """把论文的声明固定下来，作为"实现与描述不符"的证据。"""
    tex = ROOT / "BurnResu .tex"
    if not tex.exists():
        pytest.skip("本地无稿件")
    text = tex.read_text(encoding="utf-8", errors="replace")
    for needle in ("Tversky", "mathcal{L}_{\\mathrm{cum}}",
                   "mathcal{L}_{\\mathrm{pos}}"):
        assert needle in text, f"稿件中未找到 {needle}，本票前提需复核"


# ------------------------------------------------- 消融的可执行性

def test_loss_ablation_is_not_runnable_as_specified():
    """汇总结论：票面要求的逐项消融**无法按原样执行**。

    五个声称的损失项里只有 ``L_cum`` 存在；Tversky 未实现；Focal 未接入。
    要执行消融，必须先实现这些项并重训——那是 Phase 2 的实现工作，
    且前提是确认论文主张仍成立（参见 #13：模型相对朴素基线无显著优势）。
    """
    missing = [t for t in ("L_cp", "L_tv", "L_pre", "L_pos")
               if _occurrences(t) == []]
    assert set(missing) == {"L_cp", "L_tv", "L_pre", "L_pos"}, \
        "缺失项集合发生变化，请更新 #17 的结论"
    assert _occurrences("Tversky") == []
