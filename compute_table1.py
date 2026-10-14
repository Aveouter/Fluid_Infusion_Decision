#!/usr/bin/env python
"""从源数据重算表 1 的统计量（issue #26）。

论文表 1 存在多处不经重训即可修正的错误：呼吸频率与心率数值完全相同、
年龄 IQR 格式为 "(28,6)"、缺失率列标 % 却填 0.55–0.97、液体类别计数与
源数据对不上。本脚本回源计算正确值，**只输出脱敏聚合统计**。

用法：
    python compute_table1.py
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Dict, List

import numpy as np

VITAL_COLUMNS = ("体温", "心率", "脉搏", "呼吸", "收缩压", "舒张压",
                 "CVP", "氧饱和度", "尿量", "口服")


def _read_csv_robust(path, encoding="utf-8"):
    """读取可能带 BOM 或 GBK 的 CSV，返回表头与行。"""
    for enc in (encoding, "utf-8-sig", "gbk"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                rows = list(csv.reader(f))
            if rows:
                return rows[0], rows[1:]
        except (UnicodeDecodeError, LookupError):
            continue
    return [], []


def vital_stats(directory: str) -> Dict[str, dict]:
    """把全部患者的逐小时生命体征汇总，给出中位数、IQR 与缺失率。

    分母 = 全部规划小时位置（患者数 × 72），与论文表 1 的口径一致。
    """
    pooled: Dict[str, List[float]] = {c: [] for c in VITAL_COLUMNS}
    total_positions = 0
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    for path in files:
        header, rows = _read_csv_robust(path)
        if not header:
            continue
        idx = {name: i for i, name in enumerate(header)}
        for row in rows:
            total_positions += 1
            for col in VITAL_COLUMNS:
                if col not in idx or idx[col] >= len(row):
                    continue
                raw = (row[idx[col]] or "").strip()
                if raw == "":
                    continue
                try:
                    pooled[col].append(float(raw))
                except ValueError:
                    continue

    out = {"n_positions": total_positions, "n_files": len(files), "columns": {}}
    for col, vals in pooled.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            out["columns"][col] = {"n_observed": 0, "missing_rate": 1.0}
            continue
        out["columns"][col] = {
            "n_observed": int(arr.size),
            "missing_rate": 1.0 - arr.size / total_positions if total_positions else float("nan"),
            "median": float(np.median(arr)),
            "q1": float(np.percentile(arr, 25)),
            "q3": float(np.percentile(arr, 75)),
        }
    return out


def main():
    base = "data/baseline_processed.csv"
    print("== 基线静态特征 ==")
    header, rows = _read_csv_robust(base)
    if header:
        idx = {n: i for i, n in enumerate(header)}
        for name in ("年龄", "age", "烧伤总面积", "体重"):
            if name not in idx:
                continue
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[idx[name]]))
                except (ValueError, IndexError):
                    continue
            if not vals:
                continue
            a = np.asarray(vals)
            print(f"  {name}: n={a.size}  中位={np.median(a):.1f}  "
                  f"IQR=({np.percentile(a,25):.1f}, {np.percentile(a,75):.1f})  "
                  f"范围=({a.min():.0f}, {a.max():.0f})")
    else:
        print("  （无法读取）")

    print()
    print("== 生命体征（分母 = 全部规划小时位置）==")
    st = vital_stats("data/data_speed/vital_signs")
    print(f"  文件 {st['n_files']} 个，规划小时位置 {st['n_positions']} 个")
    print(f"  {'变量':<10}{'缺失率':>10}{'中位':>10}{'IQR':>20}")
    for col, v in st["columns"].items():
        if v.get("n_observed", 0) == 0:
            print(f"  {col:<10}{'全部缺失':>10}")
            continue
        print(f"  {col:<10}{v['missing_rate']:>10.2%}{v['median']:>10.1f}"
              f"{'(' + format(v['q1'], '.1f') + ', ' + format(v['q3'], '.1f') + ')':>20}")

    # 液体类别：从标签统计（与 #4 的协议口径一致）
    print()
    print("== 液体类别（多标签，分母 = 有标签的小时数）==")
    import pickle
    d = pickle.load(open("data/output_data.pkl", "rb"))
    bits = []
    for rec in d.values():
        lab = np.asarray(rec["label"], dtype=float)
        if lab.size:
            bits.append(lab[:, :3])
    if bits:
        B = np.concatenate(bits, axis=0)
        n = len(B)
        for i, name in enumerate(("晶体 crystalloid", "胶体 colloid", "水 glucose/water")):
            k = int(B[:, i].sum())
            print(f"  {name:<22} {k:>7} 小时  ({k / n:.1%})")
        allzero = int((~np.any(B > 0, axis=1)).sum())
        print(f"  {'三类均无（零流）':<22} {allzero:>7} 小时  ({allzero / n:.1%})")
        print(f"  合计 {n} 小时（多标签，各类占比之和 > 100%）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
