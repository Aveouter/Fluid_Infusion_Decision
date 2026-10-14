"""合成数据生成器：结构与真实数据同形，内容全部虚构。

用途
----
让测试与 CI 无需真实临床数据即可运行（见 issue #3）。

安全约束（roadmap 隐私规则 + issue #3 的 test_data_privacy）
----------------------------------------------------------
- patient_id 一律使用 ``SYNTH_`` 前缀，与真实编号空间不重叠。
- 本模块 **不读取** ``data/`` 下任何文件，也不 import 任何加载真实数据的模块。
- 生成的值仅用于形状/接口校验，不具备任何临床含义。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

#: 虚构编号前缀。真实数据使用纯数字编号，二者不会碰撞。
SYNTH_PREFIX = "SYNTH_"

#: 标签列顺序（与 utils/data_loader 的约定一致）：3 类 bit + 3 类速度
LIQUID_TYPES: Tuple[str, ...] = ("crystal", "colloid", "water")


def make_patient_ids(n: int) -> List[str]:
    """生成 n 个虚构 patient_id。"""
    if n < 0:
        raise ValueError("n 必须非负")
    return [f"{SYNTH_PREFIX}{i:04d}" for i in range(n)]


def make_baseline(n: int, dim_base: int = 10, seed: int = 0) -> Dict[str, np.ndarray]:
    """静态基线特征：{patient_id: (dim_base,)}。"""
    rng = np.random.default_rng(seed)
    ids = make_patient_ids(n)
    return {pid: rng.normal(size=dim_base).astype(np.float32) for pid in ids}


def make_timeseries(
    n: int,
    length: int = 72,
    dim_temporal: int = 19,
    num_classes: int = 3,
    seed: int = 0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """时序记录：{patient_id: {"tdata": (length, dim_temporal), "label": (length, 2*num_classes)}}。

    label 前 num_classes 列为 0/1 状态位，后 num_classes 列为对应速度。
    速度在状态为 0 时恒为 0，保持 bit/speed 自洽。
    """
    rng = np.random.default_rng(seed)
    ids = make_patient_ids(n)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for pid in ids:
        tdata = rng.normal(size=(length, dim_temporal)).astype(np.float32)
        bits = (rng.random(size=(length, num_classes)) < 0.4).astype(np.float32)
        speeds = (rng.random(size=(length, num_classes)) * 200.0).astype(np.float32)
        speeds = speeds * bits
        label = np.concatenate([bits, speeds], axis=1).astype(np.float32)
        out[pid] = {"tdata": tdata, "label": label}
    return out


def make_dataset(
    n: int = 8,
    length: int = 72,
    dim_base: int = 10,
    dim_temporal: int = 19,
    num_classes: int = 3,
    seed: int = 0,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """一次生成 (Ts_data, Base_data)，供 data_loader 直接消费。"""
    return (
        make_timeseries(n, length, dim_temporal, num_classes, seed),
        make_baseline(n, dim_base, seed),
    )


def batch_from_arrays(batch: int, history: int, dim_temporal: int, dim_base: int, seed: int = 0):
    """生成一对 (xT, xB) 张量，形状与模型 forward 的期望一致。"""
    import torch

    g = torch.Generator().manual_seed(seed)
    xT = torch.randn(batch, history, dim_temporal, generator=g)
    xB = torch.randn(batch, dim_base, generator=g)
    return xT, xB
