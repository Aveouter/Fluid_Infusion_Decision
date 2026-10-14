"""issue #6 / RED：患者互斥划分与「仅训练集拟合」不变量。

覆盖四类风险：
1. 同一患者的窗口不得跨划分（患者是独立性单位，窗口不是）；
2. 划分必须由 seed 决定且可复现；
3. scaler / 阈值等只能由训练（或验证）集拟合，测试集只做 transform；
4. 声明的约束（阈值下限、容差终点）必须真正生效。

患者辨识技巧：给每位患者不同的序列长度，则窗口首列的 t_norm 步长
``1/(长度-1)`` 唯一标识该患者，无需改动库代码即可端到端核对划分。
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from tests import synthetic

# 各患者长度互不相同 -> t_norm 步长唯一 -> 可反查窗口属于哪位患者
PATIENT_LENGTHS = [20, 22, 24, 26, 28, 30]


def _make_ts_with_distinct_lengths(seed=11):
    """每位患者一个不同的序列长度，长度即患者标识。"""
    ts, base = {}, {}
    for i, L in enumerate(PATIENT_LENGTHS):
        pid = f"SYNTH_{i:04d}"
        t, b = synthetic.make_dataset(n=1, length=L, dim_base=10,
                                      dim_temporal=19, seed=seed + i)
        rec = next(iter(t.values()))
        rec["patient_id"] = pid          # make_dataset(n=1) 恒用 0000，必须改写
        ts[pid] = rec
        base[pid] = next(iter(b.values()))
    return ts, base


ALL_PIDS = {f"SYNTH_{i:04d}" for i in range(len(PATIENT_LENGTHS))}


def _patient_of(x) -> int:
    """由窗口首列 t_norm 的步长反查患者（返回其序列长度）。

    仅在**未增强**的划分上可用：TSAugmenter 的 magnitude_warp 会扰动该列。
    """
    step = float(x[1, 0] - x[0, 0])
    if step <= 0:
        raise AssertionError("t_norm 步长非正，无法辨识患者")
    return int(round(1.0 / step)) + 1


def _loader_pids(loader):
    """取出该 loader 中每个窗口的 patient_id（自动处理 Subset 包装）。"""
    from torch.utils.data import Subset

    ds = loader.dataset
    if isinstance(ds, Subset):
        pids = ds.dataset.patient_ids
        assert pids is not None, "数据集未携带 patient_ids"
        return [pids[i] for i in ds.indices]
    assert ds.patient_ids is not None, "数据集未携带 patient_ids"
    return list(ds.patient_ids)


def _loaders(ts, base, H=4, P=1, seed=12, use_scaler=False, splits=(0.5, 0.25, 0.25)):
    from utils.data_loader import data_loader

    return data_loader(
        ts, base, history_length=H, pred_length=P,
        batch_size=2, num_workers=0, pin_memory=False,
        oversample_water=False, use_robust_scaler=use_scaler,
        splits=splits, seed=seed,
    )


def _split_patients(ts, base, **kw):
    """返回 (train患者集, val患者集, test患者集)。"""
    loaders = _loaders(ts, base, **kw)
    return tuple(set(_loader_pids(l)) if l is not None else set() for l in loaders)


# ------------------------------------------------------------ 患者互斥

def test_patient_disjoint():
    """训练/验证/测试的患者集合两两无交集，且覆盖全部患者。"""
    ts, base = _make_ts_with_distinct_lengths()
    tr, va, te = _split_patients(ts, base)

    assert tr and va and te, f"存在空划分：{len(tr)}/{len(va)}/{len(te)}"
    assert not (tr & va), f"train 与 val 共享患者：{tr & va}"
    assert not (tr & te), f"train 与 test 共享患者：{tr & te}"
    assert not (va & te), f"val 与 test 共享患者：{va & te}"
    assert tr | va | te == ALL_PIDS, "患者未被完整覆盖"


def test_same_patient_windows_stay_together():
    """同一患者的全部窗口必须落在同一个划分里，不能按窗口随机切。"""
    ts, base = _make_ts_with_distinct_lengths()
    tr, va, te = _split_patients(ts, base)
    # 若按窗口切分，同一患者的长度会同时出现在多个划分中
    for a, b, name in ((tr, va, "train/val"), (tr, te, "train/test"), (va, te, "val/test")):
        assert not (a & b), f"{name} 出现同一患者的窗口"


# ------------------------------------------------------------ 划分可复现

def test_split_seed_reproducibility():
    """同一 seed 重复构造，患者分配必须完全一致。"""
    ts, base = _make_ts_with_distinct_lengths()
    first = _split_patients(ts, base, seed=12)
    second = _split_patients(ts, base, seed=12)
    assert first == second, "相同 seed 下划分不可复现"


def test_different_seeds_can_reshuffle():
    """不同 seed 应能产生不同划分（否则 seed 形同虚设）。"""
    ts, base = _make_ts_with_distinct_lengths()
    base_split = _split_patients(ts, base, seed=12)
    seen_other = any(
        _split_patients(ts, base, seed=s) != base_split for s in (1, 2, 3, 4, 5)
    )
    assert seen_other, "更换 seed 未改变划分"


def test_split_reports_counts():
    """划分结果应公布患者数与窗口数，便于论文报告。"""
    ts, base = _make_ts_with_distinct_lengths()
    loaders = _loaders(ts, base)
    counts = [len(l.dataset) if l is not None else 0 for l in loaders]
    assert sum(counts) > 0
    assert all(c > 0 for c in counts), f"存在空划分：{counts}"


# ------------------------------------------------- 仅训练集拟合（scaler）

def test_scaler_fit_is_deterministic():
    """同一份数据、同一 seed 重复构造，scaler 统计量必须完全一致。

    旧实现用挂着 TSAugmenter 的 train_dataset 取样本拟合，随机增强使统计量
    每次运行都不同——划分与阈值都固定了，scaler 却是随机的。
    """
    ts, base = _make_ts_with_distinct_lengths()
    a = _loaders(ts, base, use_scaler=True)[1].dataset.transform
    b = _loaders(ts, base, use_scaler=True)[1].dataset.transform
    assert np.allclose(a.median, b.median), "scaler 中位数不可复现"
    assert np.allclose(a.iqr, b.iqr), "scaler IQR 不可复现"


def test_scaler_is_fitted_on_train_only():
    """改动**测试**患者的数据不得改变训练集拟合出的 scaler 统计量。"""
    ts, base = _make_ts_with_distinct_lengths()
    tr, va, te = _split_patients(ts, base)

    loaders = _loaders(ts, base, use_scaler=True)
    scaler_before = loaders[1].dataset.transform          # val 上挂的就是 scaler
    assert hasattr(scaler_before, "median"), "未取得 scaler，测试无效"
    med0, iqr0 = scaler_before.median.copy(), scaler_before.iqr.copy()

    # 只把测试患者的极端值改掉
    pid = sorted(te)[0]
    ts2 = {k: {**v, "tdata": v["tdata"].copy(), "label": v["label"].copy()}
           for k, v in ts.items()}
    ts2[pid]["tdata"] = ts2[pid]["tdata"] * 1000.0

    loaders2 = _loaders(ts2, base, use_scaler=True)
    scaler_after = loaders2[1].dataset.transform

    assert np.allclose(med0, scaler_after.median), "测试患者影响了训练集 scaler 的中位数"
    assert np.allclose(iqr0, scaler_after.iqr), "测试患者影响了训练集 scaler 的 IQR"


def test_patient_ids_match_the_underlying_series():
    """交叉核对：数据集携带的 patient_id 必须与其窗口真实所属的患者一致。

    用 t_norm 步长反查序列长度作为独立证据（仅在未增强的划分上可用）。
    """
    ts, base = _make_ts_with_distinct_lengths()
    loaders = _loaders(ts, base, use_scaler=False)
    for split_name, loader in zip(("train", "val", "test"), loaders):
        if loader is None or split_name == "train":
            continue  # train 挂了增强器，t_norm 被扰动，无法交叉核对
        ds = loader.dataset
        pids = _loader_pids(loader)
        for i in range(len(ds)):
            length = _patient_of(ds[i][0])
            expected = f"SYNTH_{PATIENT_LENGTHS.index(length):04d}"
            assert pids[i] == expected, (
                f"{split_name} 第 {i} 个窗口标注为 {pids[i]}，"
                f"但数据对应的患者是 {expected}"
            )


def test_val_and_test_share_train_scaler():
    """val 与 test 必须使用同一份由训练集拟合的 scaler，而不是各自拟合。"""
    ts, base = _make_ts_with_distinct_lengths()
    loaders = _loaders(ts, base, use_scaler=True)
    s_val, s_test = loaders[1].dataset.transform, loaders[2].dataset.transform
    assert np.allclose(s_val.median, s_test.median)
    assert np.allclose(s_val.iqr, s_test.iqr)


# ------------------------------------------------- 阈值：不得在测试集上调优

def _tiny_args(**over):
    a = types.SimpleNamespace(
        model="tcn_film", num_classes=3, pred_length=1,
        max_speed=1200.0, zero_target_eps=1e-6, delta_ml=1.0,
        cum_tol_epsilon_ml=300.0, smooth_win=1, smooth_classes=[2],
        tune_thresholds=False, fixed_thresholds=[0.5, 0.5, 0.5],
        fbeta_per_class=[1.0, 1.0, 1.0],
        weight_col=3, tsba_col=5, use_amp=False,
        lambda_cls=1.0, lambda_reg=1.0, lambda_cum=0.01,
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _tiny_loader(n=24, seed=0):
    from utils.data_loader import data_loader

    ts, base = synthetic.make_dataset(n=3, length=16, dim_base=10, dim_temporal=19, seed=seed)
    loaders = data_loader(
        ts, base, history_length=4, pred_length=1, batch_size=4,
        num_workers=0, pin_memory=False, oversample_water=False,
        use_robust_scaler=False, splits=(0.7, 0.15, 0.15),
    )
    return next(l for l in loaders if l is not None)


def _tiny_model():
    from models.tcn_film import build_tcnfilm_full

    return build_tcnfilm_full(
        input_dim_base=10, input_dim_temporal=19, embed_dim=16,
        num_classes=3, pred_length=1, n_transformer_layers=1,
        n_heads=2, tcn_layers_short=1, tcn_layers_long=1,
    )


def _run(loader, args):
    from run_epoch import run_epoch

    pack = run_epoch(_tiny_model(), loader, optimizer=None,
                     criterion_classification=None, criterion_regression=None,
                     device=torch.device("cpu"), args=args,
                     is_train=False, epoch=0, draw_roc=False)
    stats = pack[-1] if isinstance(pack[-1], dict) else {}
    return stats.get("overall", {}).get("best_thresholds")


def test_test_split_does_not_tune_thresholds():
    """``tune_thresholds=False`` 时必须沿用 fixed 阈值，不得在评估集上重调。

    这是测试集评测的路径（``train.py`` 在调用前把 tune_thresholds 置 False
    并载入验证集产出的 best_thresholds.json）。
    """
    loader = _tiny_loader()
    fixed = [0.41, 0.42, 0.43]
    got = _run(loader, _tiny_args(tune_thresholds=False, fixed_thresholds=fixed))
    assert got is not None, "未取到阈值字段"
    assert np.allclose(got, fixed), (
        f"tune_thresholds=False 却得到 {got}，固定阈值 {fixed} 未被沿用"
    )


def test_tuning_enabled_actually_searches():
    """反向对照：开启调优时阈值应随数据变化，确认上一条不是恒真。"""
    loader = _tiny_loader()
    fixed = [0.41, 0.42, 0.43]
    got = _run(loader, _tiny_args(tune_thresholds=True, fixed_thresholds=fixed))
    assert got is not None
    assert not np.allclose(got, fixed), "开启调优后阈值未变化，该对照失去意义"


# ------------------------------------------- 声明的约束必须真正生效（红灯）

def test_declared_threshold_floor_is_enforced():
    """``--min_thresholds`` / ``--prec_floor`` 声明了却从未被使用。

    ``main.py`` 暴露这两个参数（默认 0.40 / 0.70），但
    ``fbeta_opt_thresh_per_class`` 没有任何下限输入，网格从 0.05 起。
    实测 ``best_thresholds.json`` 中 colloid 阈值为 0.30，低于声明的 0.40。
    声明的约束不生效会让方法学描述与实际行为不符。
    """
    import inspect

    from utils.util import fbeta_opt_thresh_per_class

    sig = inspect.signature(fbeta_opt_thresh_per_class)
    assert "min_thresholds" in sig.parameters or "floor" in sig.parameters, (
        "阈值优化函数不接受下限参数，--min_thresholds/--prec_floor 无法生效"
    )


def test_tolerance_parameters_are_still_unused():
    """特征化测试：``--tol_abs_ml`` / ``--tol_pct`` 至今仍是死配置。

    **这不是期望行为。** 论文方法把「±200 mL / ±10% 容差准确率」列为终点，
    而这两个参数只在 ``main.py`` 声明、从未参与任何计算，说明该终点尚未实现。
    本用例固定当前状态，待 #12 实现容差终点后应当失败并被改写。
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    hits = []
    for path in root.rglob("*.py"):
        if any(part in {".venv", ".audit", "tests"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "tol_abs_ml" in text or "tol_pct" in text:
            hits.append(path.name)
    # 只应出现在 main.py 的参数声明处，不应参与任何计算
    assert set(hits) <= {"main.py"}, (
        f"容差参数出现在 {sorted(set(hits))}；若已实现，请更新本用例"
    )
