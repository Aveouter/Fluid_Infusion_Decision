"""issue #14 / RED：基线公平性的结构守卫。

对应审稿意见：LR/SVM 指标近乎相同、GBR 与 RF/XGBoost 差距巨大、LSTM 表现异常，
需排除目标定义、输入、头部与调参口径不一致。

这里钉住的是**结构性事实**（哪条路径用哪个口径），不是数值；一旦有人改动
评测代码使得两族模型再次不可比，用例会失败而不是静默通过。
"""

from __future__ import annotations

import importlib
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _src(name):
    return (ROOT / name).read_text(encoding="utf-8", errors="replace")


# ------------------------------------------------ 两条评测路径的口径

def test_run_epoch_reports_macro_only():
    """深度模型路径只产出 macro F1，没有 micro/weighted/samples 分支。"""
    src = _src("run_epoch.py")
    for agg in ("micro", "weighted", "samples"):
        assert f"average='{agg}'" not in src and f'average="{agg}"' not in src, \
            f"run_epoch.py 出现了 {agg} 口径，表 2 的两半可能已不可比"
    assert "average='macro'" in src


def test_train_for_ml_reports_micro_f1_but_macro_auc():
    """传统 ML 路径：**F1 用 micro，AUC 用 macro**。

    这比"整条路径口径不同"更精确，也解释了表 2 的表现：AUC 列两半同口径
    （都是 macro）因而看起来可比，F1 列则是 micro 对 macro，不可比。
    """
    src = _src("train_for_ml.py")
    assert "average=\"micro\"" in src, "F1 不再是 micro，需更新 A2"
    assert "roc_auc_score(y_cls_val, p_val, average=\"macro\")" in src, \
        "AUC 不再是 macro，需更新 A2"
    assert "f1_score(y_cls_test, y_pred, average=\"micro\")" in src


def test_two_paths_disagree_on_aggregation():
    """特征化：两条路径确实用了不同聚合，这是 A2 的证据。"""
    run_epoch, train_for_ml = _src("run_epoch.py"), _src("train_for_ml.py")
    assert "average='macro'" in run_epoch
    assert 'average="micro"' in train_for_ml
    doc = _src("docs/baseline_fairness.md")
    assert "A2" in doc, "口径混用未在公平性文档中记录"


# ------------------------------------------------ 回归口径与掩码

def test_deep_path_uses_conditional_speed_with_positive_mask():
    """深度路径的 MAE 用 speed（条件均值）+ 正流掩码 —— 与 ML 路径不同。"""
    src = _src("run_epoch.py")
    assert "flow_eval = speed" in src, "回归评估已不再使用 speed，请核对 A1"
    assert "y_true_reg > zero_target_eps" in src, "正流掩码不见了，请核对 A1"


def test_ml_path_does_not_mask():
    """传统 ML 路径对全部窗口计算 MAE，不做正流掩码。"""
    src = _src("train_for_ml.py")
    assert "y_reg.append(y_np[:, -1, num_classes:])" in src, \
        "flatten_loader 的取目标方式变了，请核对 A1"
    assert "eval_regression(y_reg_test, reg_pred)" in src


def test_mask_asymmetry_is_documented():
    doc = _src("docs/baseline_fairness.md")
    for tag in ("A1", "A2", "A3", "A4", "A5"):
        assert tag in doc, f"公平性文档缺少 {tag}"


# ------------------------------------------------ 输入与头部契约

def test_ml_baselines_get_last_timestep_only():
    """非序列模型只拿到最后一步，序列模型拿到整个窗口。"""
    src = _src("train_for_ml.py")
    assert "x_last = x_ts[:, -1, :]" in src, "特征访问方式变了，请核对 A4"


def test_baselines_have_no_ziln_head():
    """传统 ML 基线没有 ZILN 头，无法表达零流量，也不是同一类输出。"""
    src = _src("train_for_ml.py")
    assert "ziln" not in src.lower(), \
        "ML 路径出现 ZILN 相关内容，A3 需要重写"
    deep = _src("models/tcn_film.py")
    assert "ZILNHead" in deep


def test_feature_contract_documented():
    doc = _src("docs/baseline_fairness.md")
    assert "末" in doc or "last" in doc.lower()


# ------------------------------------------------ 超参搜索不得碰测试集

def test_hyperparameter_search_does_not_touch_test():
    """两条路径的超参/阈值搜索都必须在验证集内进行。

    结构检查：ML 路径的阈值与模型选择应引用 val；深度路径的阈值来自
    验证集产出的 best_thresholds.json（train.py 在测试前显式关闭调优）。
    """
    ml = _src("train_for_ml.py")
    assert "y_cls_val" in ml and "y_reg_val" in ml, "未见验证集参与选择"

    train_src = _src("train.py")
    assert "args.tune_thresholds = False" in train_src, \
        "测试前未关闭阈值调优"


def test_cross_validation_module_keeps_fit_train_only():
    """`utils/` 内不应出现直接在测试集上 fit 的调用。"""
    bad = []
    for path in (ROOT / "utils").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if ".fit(" in line and ("test" in line.lower() or "X_test" in line):
                bad.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not bad, f"疑似在测试集上拟合：{bad}"


# ------------------------------------------------ Salinas 任务差异

def test_salinas_task_difference_is_recorded():
    doc = _src("docs/baseline_fairness.md")
    assert "Salinas" in doc
    assert "mL/kg/%TBSA" in doc or "mL/kg/%TBSA" in doc.replace(" ", "")
    ledger = _src("docs/table_provenance.md")
    assert "Salinas" in ledger, "溯源台账未收录 Salinas 行"


# ------------------------------------------------ 公平性清单

def test_fairness_checklist_is_present_and_unchecked():
    """公平性文档必须给出"一张公平表需要什么"的清单，且尚未勾选。"""
    doc = _src("docs/baseline_fairness.md")
    assert "## 3. What a fair table requires" in doc
    assert doc.count("- [ ]") >= 5, "清单条目过少"
    assert "- [x]" not in doc.split("## 3.")[1].split("## 4.")[0], \
        "清单被预先勾选，说明尚未真正完成"
