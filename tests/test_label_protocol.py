"""issue #4 / RED：目标标签协议 v1 的不变量测试。

对应审稿关注点：多标签并发、无输注（零流量）的标记、速率与体积累积口径、
重叠医嘱与推注的处理规则。

全部使用合成标签，不读取 data/ 下的真实记录。
"""

from __future__ import annotations

import numpy as np
import pytest

from utils import label_protocol as lp

TYPES = lp.NUM_TYPES


def make_label(rows):
    """把 [(bits...), (rates...)] 形式的行构造成 (T, 6) 标签。"""
    return np.array(rows, dtype=np.float32)


def row(bits, rates):
    return list(bits) + list(rates)


# ---------------------------------------------------------------- 基础不变量

def test_protocol_shape_and_version():
    """协议常量自洽：列顺序、切片、类别数与版本号。"""
    assert len(lp.COLUMN_ORDER) == 2 * TYPES
    assert len(lp.LIQUID_TYPES) == TYPES
    assert lp.PROTOCOL_VERSION
    assert lp.RATE_SLICE.start == lp.BIT_SLICE.stop


def test_validate_accepts_consistent_label():
    label = make_label([
        row([0, 0, 0], [0, 0, 0]),
        row([1, 0, 0], [100, 0, 0]),
        row([1, 1, 0], [80, 50, 0]),
    ])
    assert lp.validate(label) == []


def test_validate_catches_bit_rate_mismatch():
    """bit=0 但 rate>0 必须被判为违规 —— 这是协议的核心不变量。"""
    bad = make_label([row([0, 0, 0], [100, 0, 0])])
    problems = lp.validate(bad)
    assert problems, "bit=0 与 rate>0 并存却未被检出"
    assert any("不一致" in p for p in problems)

    bad2 = make_label([row([1, 0, 0], [0, 0, 0])])
    assert lp.validate(bad2), "bit=1 但 rate=0 未被检出"


def test_validate_catches_negative_rate():
    bad = make_label([row([0, 0, 0], [-5, 0, 0])])
    assert any("负值" in p for p in lp.validate(bad, allow_sentinel=True))


# ------------------------------------------------- 多标签并发（R2-4 / R3-8）

def test_multilabel_coadministration():
    """同一小时多种液体并发是**合法**的 —— 协议不是 softmax 互斥分类。

    真实数据显示三类可同时使用，因此任何把它当多类别处理的口径都是错的。
    """
    label = make_label([row([1, 1, 1], [150, 120, 90])])
    assert lp.validate(label) == [], "三类并发被误判为违规"

    active = lp.active_mask(label)
    assert active.sum() == 3, "三类并发未被全部识别为激活"
    assert bool(lp.coadministration_mask(label)[0]), "并发掩码未置位"

    # 互斥模型下三类的概率之和为 1；多标签下各类独立，占比之和可超过 100%
    summary = lp.summarize(label)
    assert sum(summary["per_type_ratio"].values()) == pytest.approx(3.0)
    assert summary["all_three_hours"] == 1


def test_coadministration_requires_at_least_two():
    label = make_label([
        row([1, 0, 0], [100, 0, 0]),
        row([1, 1, 0], [100, 50, 0]),
    ])
    assert lp.coadministration_mask(label).tolist() == [False, True]


# --------------------------------------------- 零流量与中断（R3-8 / R4-5）

def test_zero_flow_and_interruptions():
    """三类全零即为无输注；中断后重启不应被回填成持续输注。"""
    label = make_label([
        row([1, 0, 0], [120, 0, 0]),   # 输注中
        row([0, 0, 0], [0, 0, 0]),     # 中断：无输注
        row([1, 0, 0], [90, 0, 0]),    # 重启
    ])
    assert lp.validate(label) == []

    zf = lp.zero_flow_mask(label)
    assert zf.tolist() == [False, True, False], "中断小时未被识别为零流量"

    summary = lp.summarize(label)
    assert summary["zero_flow_hours"] == 1

    # 中断小时的体积必须为 0；不能被前值回填
    vol = lp.hourly_volume(label)
    assert vol[1].tolist() == [0.0, 0.0, 0.0], "中断小时被回填出了体积"


def test_zero_flow_is_defined_by_all_zero_bits_not_by_ratio_remainder():
    """无输注由「三类全零」定义，不能由类别占比之和的余数反推。

    真实数据里三类占比之和远大于 100%，余数没有「零输液比例」的含义。
    """
    label = make_label([
        row([1, 1, 1], [10, 10, 10]),
        row([0, 0, 0], [0, 0, 0]),
        row([1, 0, 1], [10, 0, 10]),
    ])
    summary = lp.summarize(label)
    ratio_sum = sum(summary["per_type_ratio"].values())
    assert ratio_sum > 1.0, "多标签下占比之和本就可超过 100%"
    # 1 - ratio_sum 是负数，显然不能当作零输液比例
    assert (1.0 - ratio_sum) < 0
    assert summary["zero_flow_ratio"] == pytest.approx(1 / 3)


# ------------------------------------------------- 速率/体量口径（R1-3 相关）

def test_rate_volume_identity():
    """累计体积 = Σ(active × rate[mL/h] × Δt[h])，单位一致。"""
    label = make_label([
        row([1, 1, 0], [100, 60, 0]),
        row([0, 0, 0], [0, 0, 0]),
        row([1, 0, 1], [40, 0, 200]),
    ])
    vol = lp.cumulative_volume(label, dt_hours=1.0)
    assert vol.tolist() == pytest.approx([140.0, 60.0, 200.0])

    # 逐小时体积之和必须等于累计体积
    hourly = lp.hourly_volume(label, dt_hours=1.0)
    assert hourly.sum(axis=0).tolist() == pytest.approx(vol.tolist())


def test_dt_scaling_is_linear():
    """Δt 翻倍体积翻倍 —— 防止把 mL/h 当作 mL 使用。"""
    label = make_label([row([1, 0, 0], [100, 0, 0])])
    v1 = lp.cumulative_volume(label, dt_hours=1.0)
    v2 = lp.cumulative_volume(label, dt_hours=2.0)
    assert v2.tolist() == pytest.approx((v1 * 2).tolist())


def test_inactive_hours_contribute_no_volume():
    """bit=0 的小时即便速率列残留非零值也不得计入体积（防回填污染）。"""
    label = make_label([row([0, 0, 0], [999, 0, 0])])
    vol = lp.cumulative_volume(label)
    assert vol.tolist() == [0.0, 0.0, 0.0]


# --------------------------------- 重叠医嘱 / 推注规则（R4-5，需明确策略）

def test_overlap_policy_sums_same_type_records():
    """同一小时同类多条记录按**求和**聚合，不取首条、不丢弃。"""
    label = lp.aggregate_infusion_records(
        {0: [(0, 50.0), (0, 70.0)]}, num_hours=2)
    assert label[0, lp.RATE_SLICE][0] == pytest.approx(120.0)
    assert lp.validate(label) == []


def test_overlap_policy_keeps_types_independent():
    """同小时并存的不同类各自独立成条，互不覆盖。"""
    label = lp.aggregate_infusion_records(
        {0: [(0, 100.0), (2, 200.0)]}, num_hours=1)
    assert label[0, 0] == 1 and label[0, 2] == 1
    assert label[0, 1] == 0
    assert label[0, lp.RATE_SLICE].tolist() == pytest.approx([100.0, 0.0, 200.0])


def test_overlap_policy_bolus_is_not_distinguished():
    """推注与持续输注都折算为等效小时速率 —— 记录该已知限制。"""
    bolus = lp.aggregate_infusion_records({0: [(0, 500.0)]}, num_hours=1)
    drip = lp.aggregate_infusion_records({0: [(0, 500.0)]}, num_hours=1)
    assert bolus.tolist() == drip.tolist()


def test_overlap_policy_rejects_negative_volume():
    with pytest.raises(ValueError, match="体积不得为负"):
        lp.aggregate_infusion_records({0: [(0, -10.0)]}, num_hours=1)


def test_overlap_policy_rejects_out_of_range():
    with pytest.raises(ValueError, match="越界"):
        lp.aggregate_infusion_records({5: [(0, 10.0)]}, num_hours=3)
    with pytest.raises(ValueError, match="液体序号"):
        lp.aggregate_infusion_records({0: [(9, 10.0)]}, num_hours=1)


def test_missing_hours_are_zero_flow():
    """未出现在记录里的小时按无输注处理，而不是插值或回填。"""
    label = lp.aggregate_infusion_records({0: [(0, 60.0)]}, num_hours=3)
    assert lp.zero_flow_mask(label).tolist() == [False, True, True]


# ------------------------------------------------------- 哨兵与规范化

def test_sentinel_is_normalized_to_zero():
    raw = make_label([row([-1, 0, 1], [0, 0, 50])])
    assert lp.validate(raw, allow_sentinel=True) == []
    norm = lp.normalize(raw)
    assert norm[0, 0] == 0.0
    assert lp.validate(norm) == []


def test_normalize_derives_bits_from_rate():
    """速率是权威字段：bit 由速率重写，与 data2pkl 的口径一致。"""
    raw = make_label([row([0, 1, 0], [30, 0, 0])])
    norm = lp.normalize(raw)
    assert norm[0, lp.BIT_SLICE].tolist() == [1.0, 0.0, 0.0]


def test_normalize_does_not_mutate_input():
    raw = make_label([row([0, 0, 0], [10, 0, 0])])
    before = raw.copy()
    lp.normalize(raw)
    assert np.array_equal(raw, before), "normalize 修改了输入数组"


# --------------------------------------- 与加载器清洗的一致性（当前会失败）

def _tiny_ts_data():
    """构造一条含「中断」的合成记录，用于检验 lint_data 是否遵守协议。"""
    label = make_label([
        row([1, 0, 0], [120, 0, 0]),
        row([0, 0, 0], [0, 0, 0]),      # 中断
        row([1, 0, 0], [90, 0, 0]),
    ])
    return {"SYNTH_0001": {"tdata": np.zeros((3, 19), dtype=np.float32), "label": label}}


def test_lint_data_output_satisfies_protocol():
    """lint_data 的输出必须满足协议不变量。

    当前实现只按速率做前值回填、完全忽略状态位，会把中断小时的速率填成前值，
    造成 bit=0 而 rate>0 —— 正是本测试要拦住的。
    """
    from utils.data_loader import lint_data
    cleaned = lint_data(_tiny_ts_data())
    label = np.asarray(cleaned["SYNTH_0001"]["label"], dtype=np.float32)
    problems = lp.validate(label)
    assert not problems, (
        "lint_data 输出违反协议（中断被回填成持续输注）：\n- " + "\n- ".join(problems)
    )


def test_lint_data_does_not_mutate_input():
    """lint_data 声称「不在原 dict 上原地修改」，该保证必须成立。"""
    from utils.data_loader import lint_data
    data = _tiny_ts_data()
    before = np.asarray(data["SYNTH_0001"]["label"], dtype=np.float32).copy()
    lint_data(data)
    after = np.asarray(data["SYNTH_0001"]["label"], dtype=np.float32)
    assert np.array_equal(after, before), "lint_data 原地修改了输入标签"


def test_loader_lint_path_is_protocol_valid():
    """经 data_loader(run_lint=True) 构造的标签同样满足协议。"""
    from utils.data_loader import data_loader
    from tests import synthetic

    Ts, Base = synthetic.make_dataset(n=6, length=24, dim_base=10, dim_temporal=19, seed=3)
    train, val, test = data_loader(
        Ts, Base, history_length=4, pred_length=1,
        batch_size=4, num_workers=0, pin_memory=False,
        run_lint=True, oversample_water=False, use_robust_scaler=False,
    )
    ds = train.dataset
    stacked = np.asarray(ds.labels, dtype=np.float32)
    assert stacked.ndim == 3, "标签未被堆叠成 (N, T, 6)"
    flat = stacked.reshape(-1, stacked.shape[-1])
    assert not lp.validate(flat), "加载器产出的标签违反协议"


# ------------------------------------------------------------- 汇总口径

def test_summarize_reports_multi_label_ratios():
    label = make_label([
        row([1, 1, 1], [10, 10, 10]),
        row([1, 0, 0], [10, 0, 0]),
        row([1, 0, 0], [10, 0, 0]),
        row([0, 0, 0], [0, 0, 0]),
    ])
    s = lp.summarize(label)
    assert s["hours"] == 4
    assert s["per_type_hours"] == {"crystal": 3, "colloid": 1, "water": 1}
    assert s["zero_flow_hours"] == 1
    assert s["coadministration_hours"] == 1
    assert s["all_three_hours"] == 1


def test_stack_records_concatenates_patients():
    a = make_label([row([1, 0, 0], [10, 0, 0])])
    b = make_label([row([0, 1, 0], [0, 20, 0]), row([0, 0, 0], [0, 0, 0])])
    stacked = lp.stack_records([a, b])
    assert stacked.shape == (3, 6)
    assert lp.summarize(stacked)["hours"] == 3
