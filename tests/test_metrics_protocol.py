"""issue #7 / RED：指标口径、单位与聚合层级的不变量。

对应审稿意见：同一个数字在不同表格里口径不一致（表 2 的 F1=0.743 vs
表 4 的 macro F1=0.696），且 MAE 未标明是瞬时 mL/h 还是累计 mL。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from utils import metrics as M


# ------------------------------------------------- 与 sklearn 参考值一致

@pytest.mark.parametrize("agg", ["macro", "micro", "weighted", "samples"])
def test_aggregations_match_sklearn(agg):
    from sklearn.metrics import f1_score, precision_score, recall_score

    rng = np.random.default_rng(0)
    y_true = (rng.random((200, 3)) < 0.4).astype(int)
    y_pred = (rng.random((200, 3)) < 0.4).astype(int)

    got = M.classification_metrics(y_true, y_pred)[agg]
    assert got["f1"] == pytest.approx(f1_score(y_true, y_pred, average=agg, zero_division=0))
    assert got["precision"] == pytest.approx(
        precision_score(y_true, y_pred, average=agg, zero_division=0))
    assert got["recall"] == pytest.approx(
        recall_score(y_true, y_pred, average=agg, zero_division=0))


def test_per_class_and_auc_reported():
    rng = np.random.default_rng(1)
    y_true = (rng.random((150, 3)) < 0.5).astype(int)
    y_prob = rng.random((150, 3))
    y_pred = (y_prob >= 0.5).astype(int)

    out = M.classification_metrics(y_true, y_pred, y_prob)
    assert set(out["per_class"]) == {"class_0", "class_1", "class_2"}
    assert len(out["auc_per_class"]) == 3
    # macro AUC 应等于逐类 AUC 的均值
    assert out["auc_macro"] == pytest.approx(
        float(np.mean(out["auc_per_class"])), nan_ok=True)


# ------------------------------- 不同聚合方式必须给出不同数字（口径不可混用）

def test_aggregations_differ_on_imbalanced_multilabel():
    """类别不平衡时 macro / micro / weighted 会出现可观差异。

    这正是表 2 的数字无法用单一口径解释的原因：口径不写明，数字不可比。
    """
    # 第 0 类占绝大多数阳性，第 2 类很少
    n = 300
    y_true = np.zeros((n, 3), dtype=int)
    y_true[:, 0] = 1
    y_true[:30, 1] = 1
    y_true[:10, 2] = 1

    y_pred = y_true.copy()
    y_pred[:60, 0] = 0          # 主类漏检 60 个
    y_pred[100:130, 1] = 1      # 次类多报 30 个
    y_pred[50:60, 2] = 1        # 稀有类多报 10 个

    out = M.classification_metrics(y_true, y_pred)
    f1 = {a: out[a]["f1"] for a in ("macro", "micro", "weighted")}

    assert len(set(round(v, 3) for v in f1.values())) == 3, \
        f"三种口径应给出不同 F1，实际 {f1}"
    assert f1["weighted"] > f1["macro"], "加权口径应向多数类倾斜"
    assert max(f1.values()) - min(f1.values()) > 0.05, "差异过小，用例失去意义"


# ------------------------ 论文自身数字的口径矛盾（特征化，记录历史不一致）

def test_paper_table2_f1_is_inconsistent_with_its_own_row():
    """用论文自己发表的数字做算术核对，证明表 2 的 F1 与同行其它指标不同口径。

    表 4（Per-class）：
        Crystalloid F1=0.752, Colloid F1=0.727, Glucose F1=0.609
        -> macro F1 = (0.752+0.727+0.609)/3 = 0.696
        Macro Recall = 0.812, Macro AUC = 0.816

    表 2（Comparison）BurnResu (Full) 行：
        Acc=0.742  Recall=0.813  F1=0.743  AUC=0.816

    核对：
        AUC   0.816 == 表 4 macro AUC 0.816      -> 同口径（macro）
        Recall 0.813 ≈ 表 4 macro recall 0.812   -> 同口径（macro）
        F1    0.743 != 表 4 macro F1   0.696     -> **不同口径**

    即同一行的三个分类指标里，有两个是 macro，F1 不是。随包代码
    （run_epoch.py）只计算 macro，因此 0.743 无法由现有代码产出。
    本用例把该历史不一致固定下来；#26 修表时必须给出解释或更正。
    """
    t4_per_class_f1 = [0.752, 0.727, 0.609]
    t4_macro_f1 = float(np.mean(t4_per_class_f1))
    t4_macro_recall = 0.812
    t4_macro_auc = 0.816

    assert t4_macro_f1 == pytest.approx(0.696, abs=0.001), "表 4 自身不自洽"

    t2 = {"acc": 0.742, "recall": 0.813, "f1": 0.743, "auc": 0.816}

    # AUC 与 Recall 与表 4 的 macro 一致
    assert t2["auc"] == pytest.approx(t4_macro_auc, abs=0.001)
    assert t2["recall"] == pytest.approx(t4_macro_recall, abs=0.002)

    # 但 F1 对不上 macro
    assert t2["f1"] != pytest.approx(t4_macro_f1, abs=0.005), (
        "表 2 的 F1 与表 4 macro F1 意外一致，本用例的前提已变"
    )
    # 也不是任何单一类别
    assert t2["f1"] not in [pytest.approx(v, abs=0.005) for v in t4_per_class_f1]


def test_paper_table2_f1_has_a_different_aggregation_signature():
    """表 2 中 BurnResu 的 (Acc - F1) 差值与同行其它模型相差一个量级。

    多标签下 macro F1 通常明显低于逐位准确率（少数类把宏平均拉低）。
    表 2 各行的差值：

        LSTM        Acc 0.736, F1 0.687  -> 差 0.049
        CNN         Acc 0.747, F1 0.688  -> 差 0.059
        Transformer Acc 0.744, F1 0.690  -> 差 0.054
        BurnResu    Acc 0.742, F1 0.743  -> 差 0.001   <-- 相差约 50 倍

    BurnResu 的 F1 与自身准确率几乎相等，而与同类模型及表 4 的 macro F1
    都对不上。这与「表 2 该行的 F1 采用了全局（micro 类）聚合」一致，
    也说明该行与其它行**不可直接比较**。

    随包代码只计算 macro，无法产出 0.743，故该值属历史不一致；
    #26 修表时必须解释或更正，且不得在未说明口径的情况下继续比较。
    """
    dl_rows = {
        "LSTM": (0.736, 0.687),
        "CNN": (0.747, 0.688),
        "Transformer": (0.744, 0.690),
    }
    burnresu = (0.742, 0.743)          # (Acc, F1)

    peer_gaps = [acc - f1 for acc, f1 in dl_rows.values()]
    own_gap = burnresu[0] - burnresu[1]

    assert min(peer_gaps) > 0.04, f"同类模型的 Acc-F1 差值异常：{peer_gaps}"
    assert abs(own_gap) < 0.005, f"BurnResu 的 Acc-F1 差值异常：{own_gap}"
    assert min(peer_gaps) / max(abs(own_gap), 1e-6) > 10, \
        "BurnResu 与同类模型的差值未拉开量级，本用例的前提已变"


def test_shipped_code_produces_macro_only():
    """特征化：随包代码只产出 macro 口径，没有 micro/weighted/samples 路径。

    因此表 2 的 BurnResu F1=0.743 无法由该代码复现（见 #8 与 #26）。
    一旦补上其它口径的导出路径，本用例应当失败并被改写。
    """
    root = Path(__file__).resolve().parent.parent
    src = (root / "run_epoch.py").read_text(encoding="utf-8", errors="replace")
    for agg in ("micro", "weighted", "samples"):
        assert f"average='{agg}'" not in src and f'average="{agg}"' not in src, (
            f"run_epoch.py 出现了 {agg} 口径，请更新本用例"
        )
    assert "average='macro'" in src


# --------------------------------------------------------------- 单位

def test_mae_units_are_distinct():
    """同一批预测，按不同单位报告会得到不同数值，不能混用。"""
    rng = np.random.default_rng(2)
    rate_true = rng.random((10, 3)) * 100.0        # mL/h
    rate_pred = rate_true + rng.normal(0, 10, rate_true.shape)

    step = M.mae(rate_true, rate_pred, unit="step_ml_per_h")
    pos = M.mae(rate_true, rate_pred, unit="positive_flow_ml_per_h",
                mask=M.positive_flow_mask(rate_true))
    assert step == pytest.approx(pos)              # 全为正流时二者相同

    # 引入零流小时：全时段 MAE 会被零流小时拉低
    rate_true2 = rate_true.copy()
    rate_true2[:5] = 0.0
    masked = M.mae(rate_true2, rate_pred, unit="positive_flow_ml_per_h",
                   mask=M.positive_flow_mask(rate_true2))
    allh = M.mae(rate_true2, rate_pred, unit="step_ml_per_h")
    assert masked != pytest.approx(allh), "正流口径与全时段口径未区分开"


def test_mae_rejects_unknown_unit():
    with pytest.raises(ValueError, match="unit 必须是"):
        M.mae(np.zeros(3), np.zeros(3), unit="mL")


def test_cumulative_mae_is_a_different_quantity():
    """累计体积误差（mL）与逐小时速率误差（mL/h）不是同一个量。"""
    rng = np.random.default_rng(3)
    rate = rng.random((24, 3)) * 100.0
    pred = rate + rng.normal(0, 5, rate.shape)

    step_mae = M.mae(rate, pred, unit="step_ml_per_h")
    cum_true = rate.sum(axis=0)
    cum_pred = pred.sum(axis=0)
    cum_mae = M.mae(cum_true, cum_pred, unit="cumulative_ml")

    assert cum_mae > step_mae, "累计口径没有体现 24 小时的累积"
    assert cum_mae == pytest.approx(float(np.abs(cum_true - cum_pred).mean()))


# --------------------------------------------------------- 聚合层级

def test_patient_vs_window_aggregation_differ():
    """先按患者平均再对患者平均，与直接对窗口平均，结果不同。

    真实数据里每位患者的窗口数不等，两种口径不可互换。
    """
    values = np.array([10.0, 10.0, 10.0, 0.0, 0.0])   # 患者 A 有 3 个窗口
    pids = ["A", "A", "A", "B", "B"]                    # 患者 B 有 2 个窗口

    win = M.window_average(values)
    pat = M.patient_then_average(values, pids)

    assert win == pytest.approx(6.0)
    assert pat == pytest.approx((10.0 + 0.0) / 2)      # 每位患者等权
    assert win != pytest.approx(pat), "两种层级未区分开"


def test_patient_average_ignores_nan():
    values = np.array([1.0, np.nan, 3.0])
    assert M.patient_then_average(values, ["A", "A", "B"]) == pytest.approx(2.0)


def test_patient_average_validates_lengths():
    with pytest.raises(ValueError, match="长度不一致"):
        M.patient_then_average(np.zeros(3), ["A", "B"])


# ------------------------------------------- 零流与类别缺失的显式处理

def test_zero_handling_is_reported_not_silently_averaged():
    y_true = np.array([[1, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=int)
    rate = np.array([[10.0, 0, 0], [0, 0, 0], [5.0, 5.0, 5.0]])

    info = M.describe_zero_handling(y_true, rate)
    assert info["n_rows"] == 3
    assert info["zero_flow_rows"] == 1
    assert info["per_class_positive"] == [2, 1, 1]
    assert info["classes_without_positive"] == []
    assert info["positive_flow_cells"] == 4
    assert "不能由占比余数推得" in info["note"]


def test_class_without_positive_is_flagged():
    y_true = np.array([[1, 0], [1, 0]], dtype=int)
    rate = np.array([[1.0, 0.0], [1.0, 0.0]])
    info = M.describe_zero_handling(y_true, rate)
    assert info["classes_without_positive"] == [1]


def test_all_zero_input_does_not_crash_metrics():
    """全零输入不得抛异常，也不得静默给出误导性数字。"""
    y_true = np.zeros((5, 3), dtype=int)
    y_pred = np.zeros((5, 3), dtype=int)
    out = M.classification_metrics(y_true, y_pred)
    assert out["macro"]["f1"] == 0.0
    assert np.isnan(M.window_average(np.array([np.nan, np.nan])))
    assert np.isnan(M.patient_then_average(np.array([np.nan]), ["A"]))


# ----------------------------------------------------------- 导出 schema

def test_metric_record_validates_fields():
    ok = M.MetricRecord(metric_name="f1", value=0.7, aggregation="macro",
                        unit="probability", level="patient", split="test")
    ok.validate()

    with pytest.raises(ValueError, match="未知的 aggregation"):
        M.MetricRecord("f1", 0.7, "median", "probability", "window", "test").validate()
    with pytest.raises(ValueError, match="未知的 unit"):
        M.MetricRecord("mae", 1.0, "none", "mL", "window", "test").validate()
    with pytest.raises(ValueError, match="未知的 level"):
        M.MetricRecord("mae", 1.0, "none", "cumulative_ml", "hour", "test").validate()


def test_exporter_keys_do_not_collide():
    """同名指标的不同口径必须在展平后仍然可区分。"""
    ex = M.MetricsExporter(split="test", seed=42)
    ex.add("f1", 0.69, aggregation="macro", unit="probability", level="patient")
    ex.add("f1", 0.74, aggregation="micro", unit="probability", level="window")
    flat = ex.as_dict()

    assert len(flat) == 2, f"口径不同的同名指标被覆盖：{flat}"
    assert flat["f1|macro|probability|patient"] == pytest.approx(0.69)
    assert flat["f1|micro|probability|window"] == pytest.approx(0.74)


def test_exporter_roundtrip_and_provenance():
    ex = M.MetricsExporter(split="test", seed=7)
    ex.add("mae", 27.5, aggregation="none", unit="step_ml_per_h",
           level="patient", n_patients=72, n_windows=3622, mask="positive_flow")

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "metrics.json"
        ex.write(path)
        rows = json.loads(path.read_text(encoding="utf-8"))

    assert len(rows) == 1
    r = rows[0]
    for key in ("metric_name", "value", "aggregation", "unit", "level", "split",
                "seed", "n_patients", "n_windows", "mask", "protocol_version"):
        assert key in r, f"导出记录缺少字段 {key}"
    assert r["n_patients"] == 72
    assert r["protocol_version"] == M.PROTOCOL_VERSION


def test_run_epoch_exports_metric_schema():
    """``run_epoch`` 的输出必须自带口径声明与样本规模。

    没有这层声明，同一个数字就会像表 2 那样在不同表格里被当成不同口径使用。
    """
    import torch

    from run_epoch import run_epoch
    from tests.test_split_protocol import _tiny_args, _tiny_loader, _tiny_model

    pack = run_epoch(
        _tiny_model(), _tiny_loader(), optimizer=None,
        criterion_classification=None, criterion_regression=None,
        device=torch.device("cpu"),
        args=_tiny_args(tune_thresholds=False, fixed_thresholds=[0.5, 0.5, 0.5]),
        is_train=False, epoch=0, draw_roc=False,
    )
    schema = pack[-1]["overall"]["metric_schema"]

    assert schema["protocol_version"] == M.PROTOCOL_VERSION
    for name in ("precision_macro", "recall_macro", "f1_macro", "auc_macro",
                 "bit_acc", "mae_all_pos", "rmse_all_pos", "r2_pos"):
        entry = schema[name]
        assert entry["aggregation"] in M.AGGREGATIONS + ("none",), name
        assert entry["unit"] in M.MAE_UNITS + ("probability",), name
        assert entry["level"] in M.AGGREGATION_LEVELS, name
    # 回归指标必须标明掩码
    assert schema["mae_all_pos"]["mask"] == "positive_flow"
    # 样本规模必须可报
    assert schema["n_windows"] and schema["n_windows"] > 0
    assert schema["n_patients"] and schema["n_patients"] > 0


def test_two_records_same_metric_different_split_are_distinct():
    a = M.MetricsExporter(split="val", seed=1).add("f1", 0.7, aggregation="macro")
    b = M.MetricsExporter(split="test", seed=1).add("f1", 0.6, aggregation="macro")
    assert a.split != b.split and a.value != b.value
