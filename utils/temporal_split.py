"""时间留出与重复分组划分（issue #21，对应 R2-6 / R3-2 / R4-3）。

背景
----
364 人随机一次患者层面划分（实际 363）只留下约 55 位测试患者，方差未知；
随机划分也**不能**建立时间泛化性——审稿人 4 明确要求补时间留出，否则必须
明确声明缺乏时间验证并收缩泛化主张。

本模块只负责**划分本身**，不训练：划分逻辑可以离线验证，而"划分好不好"
只有在模型上重训才能评价。把两者分开，可以先把可验证的部分钉死。

隐私
----
原始护理记录表头含 ``PATIENT_NAME`` 等身份字段。本模块**只读取**
``PATIENT_ID`` 与 ``FORM_TIME`` 两列，绝不加载姓名、住院号等字段，
也不把它们写入任何返回值。
"""

from __future__ import annotations

import csv
import glob
import os
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

#: 允许读入的列。白名单而非黑名单——新增身份字段时不会被误读。
ALLOWED_COLUMNS = ("PATIENT_ID", "FORM_TIME")


def _parse_time(s: str) -> datetime:
    """解析护理记录里的 ``FORM_TIME``（ISO 风格，可能带 Z）。"""
    s = str(s).strip().replace("Z", "").replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析时间：{s!r}")


def patient_time_ranges(csv_glob: str, encoding: str = "utf-8",
                        errors: str = "replace") -> Dict[str, Tuple[datetime, datetime]]:
    """扫描原始护理记录，返回 ``{patient_id: (最早, 最晚)}``。

    只读取 :data:`ALLOWED_COLUMNS` 中的列；其余列（含身份字段）**不进入内存**。
    """
    out: Dict[str, Tuple[datetime, datetime]] = {}
    for path in sorted(glob.glob(csv_glob)):
        with open(path, "r", encoding=encoding, errors=errors, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            missing = [c for c in ALLOWED_COLUMNS if c not in reader.fieldnames]
            if missing:
                continue
            for row in reader:
                pid = (row.get("PATIENT_ID") or "").strip()
                raw = (row.get("FORM_TIME") or "").strip()
                if not pid or not raw:
                    continue
                try:
                    t = _parse_time(raw)
                except ValueError:
                    continue
                lo, hi = out.get(pid, (t, t))
                out[pid] = (min(lo, t), max(hi, t))
    return out


def temporal_holdout(ranges: Dict[str, Tuple[datetime, datetime]],
                     fractions: Sequence[float] = (0.7, 0.15, 0.15),
                     key: str = "first") -> Dict[str, List[str]]:
    """按**患者首次（或末次）记录时间**排序后切分，保证严格时间顺序。

    ``key="first"`` 用最早记录时间排序（按入组先后），``key="last"`` 用最晚。

    返回 ``{"train": [...], "val": [...], "test": [...]}``，三者互斥且并集为全集；
    训练集的所有患者都**早于**验证集，验证集都早于测试集。
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions 必须和为 1.0，实际 {sum(fractions)}")
    if key not in ("first", "last"):
        raise ValueError(f"key 必须是 'first' 或 'last'，实际 {key}")

    idx = 0 if key == "first" else 1
    ordered = sorted(ranges.items(), key=lambda kv: (kv[1][idx], kv[0]))
    pids = [pid for pid, _ in ordered]

    n = len(pids)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    train = pids[:n_train]
    val = pids[n_train:n_train + n_val]
    test = pids[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def assert_strict_order(ranges: Dict[str, Tuple[datetime, datetime]],
                        split: Dict[str, List[str]], key: str = "first") -> None:
    """断言划分满足严格时间顺序：max(train) <= min(val) <= min(test)。"""
    idx = 0 if key == "first" else 1

    def _bounds(group):
        times = [ranges[p][idx] for p in group]
        return (min(times), max(times)) if times else (None, None)

    _, hi_tr = _bounds(split["train"])
    lo_va, hi_va = _bounds(split["val"])
    lo_te, _ = _bounds(split["test"])
    if hi_tr and lo_va and hi_tr > lo_va:
        raise AssertionError(f"训练集最晚 {hi_tr} 晚于验证集最早 {lo_va}")
    if hi_va and lo_te and hi_va > lo_te:
        raise AssertionError(f"验证集最晚 {hi_va} 晚于测试集最早 {lo_te}")


def assert_disjoint(split: Dict[str, List[str]]) -> None:
    """断言三个划分两两无交集，且并集为全集（患者不重复、不遗漏）。"""
    tr, va, te = (set(split[k]) for k in ("train", "val", "test"))
    if tr & va or tr & te or va & te:
        raise AssertionError(f"划分存在交集：train∩val={tr & va}, "
                             f"train∩test={tr & te}, val∩test={va & te}")
    total = len(tr) + len(va) + len(te)
    if total != len(tr | va | te):
        raise AssertionError("患者被重复计入多个划分")


def grouped_folds(patient_ids: Sequence[str], patient_groups: Dict[str, str],
                  n_folds: int = 5, seed: int = 42) -> List[List[str]]:
    """把**患者**分成 ``n_folds`` 折，同一位患者整体进入同一折。

    ``patient_groups`` 可把同一真实患者的不同编号归并（数据审计提示
    363 个编号中可能存在同一患者的多条记录）。未在映射中的患者按自身成组。

    返回每折的患者编号列表，折间互斥且并集为全集。
    """
    if n_folds < 2:
        raise ValueError(f"n_folds 必须 >= 2，实际 {n_folds}")
    pids = [str(p) for p in patient_ids]
    if len(set(pids)) != len(pids):
        raise ValueError("patient_ids 存在重复")

    import random
    by_group: Dict[str, List[str]] = {}
    for p in pids:
        by_group.setdefault(patient_groups.get(p, p), []).append(p)

    groups = sorted(by_group)
    rng = random.Random(seed)
    rng.shuffle(groups)

    folds: List[List[str]] = [[] for _ in range(n_folds)]
    for i, g in enumerate(groups):
        folds[i % n_folds].extend(sorted(by_group[g]))
    return folds


def assert_grouped_folds_disjoint(folds: List[List[str]]) -> None:
    seen = set()
    for f in folds:
        for p in f:
            if p in seen:
                raise AssertionError(f"患者 {p} 出现在多个折中")
            seen.add(p)


def repeated_grouped_folds(patient_ids: Sequence[str], patient_groups: Dict[str, str],
                           n_folds: int = 5, n_repeats: int = 3,
                           seed: int = 42) -> List[List[List[str]]]:
    """重复分组划分：每次用不同 seed 重新分折，用于估计划分方差。

    同一次重复内折间互斥；不同重复之间**允许**患者落在不同折——
    这正是"重复"的意义，也是估计划分方差所必需的。
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats 必须 >= 1，实际 {n_repeats}")
    return [grouped_folds(patient_ids, patient_groups, n_folds=n_folds,
                          seed=seed + r)
            for r in range(n_repeats)]
