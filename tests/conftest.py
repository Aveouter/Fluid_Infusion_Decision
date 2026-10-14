"""pytest 共享配置：路径注入、marker 注册、合成 fixture。

所有 fixture 均为合成数据，不触碰 data/ 下的真实记录。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 让 tests/ 可以直接 import 项目根的模块（models.*、utils.*、main 等）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests import synthetic  # noqa: E402  （需在 sys.path 注入之后）


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: 需要 CUDA 的测试，CPU 环境自动跳过")
    config.addinivalue_line("markers", "realdata: 需要真实数据集的测试，CI 中跳过")
    config.addinivalue_line("markers", "slow: 耗时较长的测试")


def pytest_collection_modifyitems(config, items):
    """无 CUDA 时自动跳过 gpu 测试；无 data/ 时自动跳过 realdata 测试。"""
    import torch

    has_cuda = torch.cuda.is_available()
    has_data = (ROOT / "data" / "baseline.pkl").exists()

    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(pytest.mark.skip(reason="无 CUDA 设备"))
        if "realdata" in item.keywords and not has_data:
            item.add_marker(pytest.mark.skip(reason="无真实数据 data/baseline.pkl"))


@pytest.fixture(scope="session")
def synth_ids():
    return synthetic.make_patient_ids(4)


@pytest.fixture(scope="session")
def synth_ts_data():
    return synthetic.make_timeseries(n=8, length=72, dim_temporal=19, num_classes=3, seed=1)


@pytest.fixture(scope="session")
def synth_base_data():
    return synthetic.make_baseline(n=8, dim_base=10, seed=1)


@pytest.fixture(scope="session")
def synth_dataset():
    return synthetic.make_dataset(n=8, length=72, dim_base=10, dim_temporal=19, seed=1)
