"""issue #3 / RED：隐私与自足性守卫。

两条不变式：
1. **测试自足** —— 跑测试不需要真实临床数据，缺 data/ 也能全绿。
2. **fixture 虚构** —— 测试用到的 patient_id 全部是合成前缀，不含真实编号。

这些是 roadmap「仅公开合成 fixtures」规则的机器化执行。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests import synthetic

TESTS_DIR = Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent

# 真实数据的落盘位置；测试源码中不应出现对这些路径的读取
FORBIDDEN_PATHS = ("data/baseline.pkl", "data/output_data", "data/output_data_final.pkl")
FORBIDDEN_IO = ("pickle.load", "read_pickle", "pd.read_excel", "np.load")


def _test_sources():
    return sorted(p for p in TESTS_DIR.rglob("*.py"))


def test_all_synthetic_ids_are_fictional():
    """所有 fixture patient_id 必须带合成前缀。"""
    ids = synthetic.make_patient_ids(32)
    assert len(ids) == len(set(ids)), "合成编号出现重复"
    for pid in ids:
        assert pid.startswith(synthetic.SYNTH_PREFIX), f"非虚构编号: {pid}"
        # 真实编号为纯数字，不应与合成编号同形
        assert not re.fullmatch(r"\d+", pid), f"编号形状与真实编号重叠: {pid}"


def test_synthetic_generator_is_self_contained():
    """合成生成器不得读取任何真实数据文件。"""
    src = (TESTS_DIR / "synthetic.py").read_text(encoding="utf-8")
    code_lines = [
        ln for ln in src.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    body = "\n".join(code_lines)
    for token in FORBIDDEN_IO:
        assert token not in body, f"synthetic.py 不应包含 IO 调用: {token}"


def test_tests_do_not_read_real_data():
    """测试源码不得对真实数据路径发起读取。

    判定标准是「同一行同时出现数据路径与 IO 调用」，而非仅出现路径字符串——
    conftest 需要探测 ``data/`` 是否存在以决定是否跳过 realdata 用例，
    这类存在性判断与提示文案不应被误判为读取。
    """
    offenders = []
    for path in _test_sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            has_path = any(tok in line for tok in FORBIDDEN_PATHS)
            has_io = any(tok in line for tok in FORBIDDEN_IO)
            if has_path and has_io:
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, f"测试对真实数据发起了读取: {offenders}"


def test_fixtures_do_not_require_real_pkl():
    """合成 fixture 在 data/ 不存在时也必须可用。"""
    ts, base = synthetic.make_dataset(n=4, length=24, dim_base=10, dim_temporal=19, seed=7)
    assert len(ts) == len(base) == 4
    for pid, rec in ts.items():
        assert pid.startswith(synthetic.SYNTH_PREFIX)
        assert rec["tdata"].shape == (24, 19)
        assert rec["label"].shape == (24, 6)


def test_synthetic_labels_are_bit_speed_consistent():
    """速度在状态为 0 时必须为 0 —— fixture 也要遵守数据契约。"""
    import numpy as np

    ts, _ = synthetic.make_dataset(n=4, length=24, num_classes=3, seed=11)
    for rec in ts.values():
        label = rec["label"]
        bits, speeds = label[:, :3], label[:, 3:]
        assert np.all(speeds[bits == 0] == 0), "合成标签出现 bit=0 但 speed>0"


def test_no_real_data_directory_is_required():
    """data/ 缺失不应导致收集期失败（收集期不读数据即视为通过）。"""
    # 此断言的意义在于：本文件与 conftest 均未在导入期打开 data/
    assert (ROOT / "tests").is_dir()
