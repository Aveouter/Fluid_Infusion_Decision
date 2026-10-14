"""issue #5 / RED：预测窗口与时间截断的不变量。

覆盖三件事：
1. 窗口切分正确 —— 历史与目标不相交，目标严格晚于 cutoff；
2. 未来变异不影响当前预测的输入 —— 沿真实 data_loader 路径验证；
3. 记录预处理阶段的未来信息泄漏（特征化测试，非期望行为）。

全部使用合成序列；涉及本地预处理源码的用例在源码不可用时自动跳过。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests import synthetic
from utils import window_protocol as wp


# ------------------------------------------------------------ 窗口切分协议

def test_protocol_alignment_is_disjoint():
    spec = wp.make_window_spec(history_length=4, pred_length=3)
    assert spec.window_len == 7
    assert spec.cutoff == 3
    assert spec.history_slice == slice(0, 4)
    assert spec.target_slice == slice(4, 7)
    assert spec.history_slice.stop <= spec.target_slice.start, "历史与目标相交"
    assert spec.target_start == spec.cutoff + 1, "目标未严格晚于 cutoff"


@pytest.mark.parametrize("H,P", [(1, 1), (1, 5), (5, 1), (12, 3), (72, 1)])
def test_history_length_matrix(H, P):
    """H=1/P=1 与 H>1/P>1 都要给出合法切分，不越界。"""
    spec = wp.make_window_spec(H, P)
    assert spec.history_length == H and spec.pred_length == P
    assert spec.target_slice.stop == H + P
    assert spec.cutoff == H - 1

    # 序列恰好容纳一个窗口时，目标右端正好落在最后一步
    starts = wp.window_starts(H + P, spec)
    assert starts == [0]
    assert starts[0] + spec.window_len - 1 == H + P - 1


@pytest.mark.parametrize("H,P", [(1, 1), (3, 2), (8, 4)])
def test_window_starts_never_overflow(H, P):
    spec = wp.make_window_spec(H, P)
    series_len = 40
    starts = wp.window_starts(series_len, spec)
    assert starts, "序列足够长却未产生窗口"
    for s in starts:
        assert s + spec.window_len <= series_len, f"窗口起点 {s} 越界"
    # 窗口数应等于可容纳的数量
    assert len(starts) == (series_len - spec.window_len) // spec.stride + 1


def test_window_too_short_yields_nothing():
    spec = wp.make_window_spec(10, 2)
    assert wp.window_starts(11, spec) == []


@pytest.mark.parametrize("H,P", [(0, 1), (1, 0), (-1, 2), (2, -3)])
def test_degenerate_params_rejected(H, P):
    with pytest.raises(ValueError):
        wp.make_window_spec(H, P)


# ------------------------------------------------ split_history_target 契约

@pytest.mark.parametrize("H,P", [(1, 1), (4, 1), (6, 3)])
def test_split_shapes_and_no_overlap(H, P):
    B, Dts, Dlab = 2, 13, 6
    W = H + P
    ts = torch.arange(B * W * Dts, dtype=torch.float32).reshape(B, W, Dts)
    lab = torch.arange(B * W * Dlab, dtype=torch.float32).reshape(B, W, Dlab) + 1000.0

    inp, tgt = wp.split_history_target(ts, lab, P)

    assert inp.shape == (B, H, Dts + Dlab), "输入长度应恰为历史长度"
    assert tgt.shape == (B, P, Dlab), "目标长度应恰为 pred_length"

    # 目标必须取自窗口最后 P 步
    assert torch.equal(tgt, lab[:, -P:, :])
    # 输入里的标签通道必须只含历史步
    assert torch.equal(inp[:, :, Dts:], lab[:, :-P, :])


def test_split_rejects_short_window():
    """窗口长度不大于 pred_length 时历史为空，必须报错而不是静默切片。"""
    ts = torch.zeros(1, 2, 13)
    lab = torch.zeros(1, 2, 6)
    with pytest.raises(ValueError, match="必须大于 pred_length"):
        wp.split_history_target(ts, lab, 2)


def test_split_rejects_mismatched_lengths():
    ts = torch.zeros(1, 5, 13)
    lab = torch.zeros(1, 4, 6)
    with pytest.raises(ValueError, match="窗口长度不一致"):
        wp.split_history_target(ts, lab, 1)


def test_causal_alignment_checker():
    ts = torch.zeros(1, 6, 13)
    lab = torch.zeros(1, 6, 6)
    wp.assert_causal_alignment(ts, lab, pred_length=2)  # 不应抛出


# ------------------------------------------------------- 因果掩码（模型侧）

@pytest.mark.parametrize("T", [1, 2, 5])
def test_causal_mask_blocks_future(T):
    from models.tcn_film import build_causal_mask

    mask = build_causal_mask(T, torch.device("cpu"))
    assert mask.shape == (T, T)
    assert mask.dtype == torch.bool
    # 严格上三角为 True（屏蔽未来），下三角与对角线为 False（可见）
    expected = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert torch.equal(mask, expected)
    assert not mask.diagonal().any(), "自身位置被错误屏蔽"


def test_causal_mask_changes_output_only_via_past():
    """扰动未来 token 不得改变因果注意力在更早位置的输出。"""
    from models.tcn_film import build_causal_mask

    torch.manual_seed(0)
    T, E = 4, 8
    attn = torch.nn.MultiheadAttention(E, 2, batch_first=True).eval()
    x = torch.randn(1, T, E)
    mask = build_causal_mask(T, torch.device("cpu"))

    with torch.no_grad():
        base, _ = attn(x, x, x, attn_mask=mask)
        x2 = x.clone()
        x2[:, -1, :] += 5.0  # 只改最后一个（未来）位置
        perturbed, _ = attn(x2, x2, x2, attn_mask=mask)

    assert torch.allclose(base[:, :-1], perturbed[:, :-1], atol=1e-6), \
        "改动未来 token 影响了更早位置的输出"
    assert not torch.allclose(base[:, -1], perturbed[:, -1]), \
        "扰动未生效，该用例失去意义"


# -------------------------------------- 未来变异不变性（真实 loader 路径）

def _single_patient_series(length=20, dim_temporal=19, dim_base=10, seed=5):
    ts = synthetic.make_timeseries(n=1, length=length,
                                   dim_temporal=dim_temporal, num_classes=3, seed=seed)
    base = synthetic.make_baseline(n=1, dim_base=dim_base, seed=seed)
    return ts, base


def _raw_windows(ts, base, H, P):
    """返回**未增强**的窗口数据集，顺序即窗口起点顺序。

    训练集一律挂 TSAugmenter，而该增强含随机的时间轴变换，会使任何「逐位
    对比」的断言失去意义。这里走 ``split_mode="window"``：该模式下 val/test
    使用的是 ``transform=None`` 的原始数据集，train 才用增强副本。
    单患者时数据集下标即窗口起点。
    """
    from torch.utils.data import Subset

    from utils.data_loader import data_loader

    loaders = data_loader(
        ts, base, history_length=H, pred_length=P,
        batch_size=1, num_workers=0, pin_memory=False,
        oversample_water=False, use_robust_scaler=False, window_stride=1,
        split_mode="window", splits=(0.6, 0.2, 0.2),
    )
    for loader in loaders:
        if loader is None:
            continue
        ds = loader.dataset
        ds = ds.dataset if isinstance(ds, Subset) else ds
        if ds.transform is None:
            return ds
    raise AssertionError("未找到未增强的数据集")


@pytest.mark.parametrize("H,P", [(1, 1), (4, 1), (5, 2)])
def test_future_mutation_invariance(H, P):
    """只改动窗口 cutoff **之后**的序列，历史输入必须逐位不变。

    这条直接回答 R1-2 / R4-5：模型的输入是否只依赖决策时刻之前的信息。
    """
    length = 20
    ts, base = _single_patient_series(length=length)
    ds_before = _raw_windows(ts, base, H, P)
    n_windows = len(ds_before)
    assert n_windows > 0, "未构造出任何窗口"

    pid = next(iter(ts))
    last_hour = length - 1

    # 只改最后一小时：标签、流速、生理特征全部改掉
    ts_after = {pid: {
        "tdata": ts[pid]["tdata"].copy(),
        "label": ts[pid]["label"].copy(),
    }}
    ts_after[pid]["tdata"][last_hour, :] += 1000.0
    ts_after[pid]["label"][last_hour, :] = 1.0

    ds_after = _raw_windows(ts_after, base, H, P)
    assert len(ds_after) == n_windows, "窗口数量不应因尾部变异而改变"

    spec = wp.make_window_spec(H, P)
    changed_somewhere = False
    for start in range(n_windows):
        x0, _, y0 = ds_before[start]
        x1, _, y1 = ds_after[start]
        cutoff_abs = start + spec.cutoff

        if last_hour > cutoff_abs:
            # 历史完全落在 cutoff 之前 -> 模型输入必须逐位不变
            assert torch.equal(x0[:H], x1[:H]), (
                f"窗口起点 {start}: 改动 cutoff 之后的序列改变了历史输入"
            )

        if not torch.equal(y0, y1):
            changed_somewhere = True

    assert changed_somewhere, (
        "尾部变异没有影响任何窗口的目标，该不变性用例会失去意义"
    )


def test_target_index_is_strictly_after_cutoff_end_to_end():
    """沿真实 loader 路径核对：目标步的下标严格大于历史最后一步。"""
    H, P = 5, 2
    ts, base = _single_patient_series(length=20)
    ds = _raw_windows(ts, base, H, P)
    spec = wp.make_window_spec(H, P)

    for start in range(len(ds)):
        x, _, y = ds[start]
        assert x.shape[0] == spec.window_len, "窗口长度与协议不符"
        assert y.shape[0] == spec.window_len, "标签窗口长度应与特征窗口一致"
        # 模型输入取窗口前 H 步；目标取后 P 步，二者不相交
        model_input, target = wp.split_history_target(
            x.unsqueeze(0), y.unsqueeze(0), P)
        assert model_input.shape[1] == H
        assert target.shape[1] == P
        assert not torch.equal(model_input[0, :, -y.shape[-1]:], target[0]), \
            f"窗口起点 {start}: 目标出现在模型输入中"


# ---------------------------------------- 预处理阶段的未来信息泄漏（记录）

def test_preprocessing_scaler_uses_future_hours():
    """特征化测试：记录预处理在**完整 72 小时**上拟合标准化这一事实。

    这不是期望行为。``data2pkl.py`` 逐患者调用 ``impute_and_scale_df``，
    后者对整张表拟合 ZScoreScaler 与 MF 插补模型，因此以 cutoff 之后观测
    决定的统计量（以及缺失格插补值）会进入模型输入。

    影响面：观测格会被还原为原值，只有**缺失格**被插补值替换——但按数据
    审计，脉搏缺失 99.73%、CVP 97.37%、体温 63.25%、心率 48.55%。

    该测试固定当前行为；一旦预处理改为按可用信息拟合，本测试应当失败并被改写。
    """
    mod = pytest.importorskip(
        "data.InsertionAndNornalization",
        reason="预处理源码位于未被跟踪的 data/ 目录，CI 上跳过",
    )
    ZScoreScaler = mod.ZScoreScaler

    rng = np.random.default_rng(0)
    base = rng.normal(size=(12, 3))

    future_changed = base.copy()
    future_changed[-1, 0] += 50.0      # 只改最后一小时（未来）

    s0 = ZScoreScaler.fit(base)
    s1 = ZScoreScaler.fit(future_changed)

    # 统计量被未来改变了
    assert not np.allclose(s0.mean_, s1.mean_), "均值未受未来观测影响"

    # 于是**早期**小时的标准化取值也随之改变 —— 即过去被未来改写
    early0 = s0.transform(base)[:3]
    early1 = s1.transform(future_changed)[:3]
    assert not np.allclose(early0, early1), "早期小时的标准化值未被未来影响"


# --------------------------- 训练期增强的时间轴变换（发现泄漏，红灯）

def _train_transform(ts, base, H=4, P=1):
    from torch.utils.data import Subset

    from utils.data_loader import data_loader

    loaders = data_loader(
        ts, base, history_length=H, pred_length=P,
        batch_size=1, num_workers=0, pin_memory=False,
        oversample_water=False, use_robust_scaler=False,
    )
    ds = loaders[0].dataset
    ds = ds.dataset if isinstance(ds, Subset) else ds
    return ds.transform


def test_train_augmenter_must_not_reorder_time():
    """训练增强不得启用会破坏时间顺序的变换。

    窗口语义是「前 H 步预测后 P 步」，而 ``TSAugmenter`` 只扰动 ``x``、
    原样返回 ``y``。一旦启用 ``permute_blocks`` 或 ``time_warp_uniform``：

    - 分段置换会把含**目标小时**的段搬进输入区；
    - 而 ``x`` 里含 label 派生的累计体积列 ``cum``，``cum[目标小时]``
      恰好包含待预测的体积 —— 输入里直接出现了答案。

    因此这类增强对本任务是无效的，必须关闭（或改造为不跨越 cutoff 的形式）。
    """
    ts, base = _single_patient_series(length=20)
    aug = _train_transform(ts, base)
    assert aug is not None, "训练集应挂增强器（否则本用例失去意义）"

    assert getattr(aug, "p_perm", 0.0) == 0.0, (
        "训练增强启用了 permute_blocks：会把目标段搬进输入区，构成标签泄漏"
    )
    assert getattr(aug, "p_timewarp", 0.0) == 0.0, (
        "训练增强启用了 time_warp_uniform：会破坏 H/P 边界的时间对齐"
    )


def test_permute_blocks_moves_target_segments_into_input():
    """特征化测试：证明 permute_blocks 会把窗口尾部搬到前部。

    固定随机种子并强制置换，观察原本位于目标区的行落到了输入区。
    """
    from utils.ts_augmentation_toolkit import permute_blocks

    H, P, D = 6, 2, 13
    W = H + P
    # 用行号当唯一标记，方便追踪行去了哪里
    x = torch.arange(W, dtype=torch.float32).reshape(W, 1).repeat(1, D)

    moved = False
    for seed in range(50):
        torch.manual_seed(seed)
        out = permute_blocks(x.clone(), n_segs=3, p=1.0)
        rows = out[:, 0].tolist()
        target_rows = set(range(H, W))
        if target_rows & set(rows[:H]):
            moved = True
            break
    assert moved, "未复现置换把目标行搬进输入区；该风险说明需要重新核实"


@pytest.mark.realdata
def test_mf_imputation_is_fitted_on_full_series():
    """同一机制的插补侧：MF 模型在完整矩阵上训练，缺失格预测依赖全部小时。

    这里只核对源码调用形态（是否把整表交给拟合函数），避免依赖随机初始化
    的 MF 训练结果导致用例不稳定。
    """
    src = pytest.importorskip(
        "data.InsertionAndNornalization",
        reason="预处理源码位于未被跟踪的 data/ 目录，CI 上跳过",
    )
    import inspect

    fn = inspect.getsource(src.impute_and_scale_df)
    assert "_fit_mf(data_t, mask_t" in fn, "未找到在整表上拟合 MF 的调用"
    assert "model()" in fn, "未找到一次性重建整表的调用"
