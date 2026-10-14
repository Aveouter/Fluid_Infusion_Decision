"""患者级 bootstrap 与配对比较（issue #13，对应 R2-1 / R3-1 / R4-2）。

为什么重采样单位必须是患者
--------------------------
同一患者的多个小时窗口高度相关，不是独立样本。若按窗口重采样，置信区间会
被人为压窄，把噪声当成显著差异——审稿人 1、2、3 都指出了这一点。

本模块的全部接口都以 ``patient_ids`` 为单位重采样；**同一位患者的窗口永远
一起进出**。配对比较中两个方法共享同一组重采样索引（配对 bootstrap），
否则差异的方差会被高估。

刻意不做的事
------------
- 不报告 p 值来"证明"优越性；只给差异的区间。区间跨 0 就是跨 0。
- 不因结果不利而调整抽样次数、种子或百分位。
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np

#: 默认抽样次数。固定以便复算。
N_BOOTSTRAP = 2000

#: 默认置信水平。
CI_LEVEL = 0.95


def _group(values: np.ndarray, patient_ids: Sequence[str]) -> Dict[str, np.ndarray]:
    values = np.asarray(values, dtype=float)
    ids = np.asarray([str(p) for p in patient_ids])
    if len(values) != len(ids):
        raise ValueError(f"values 与 patient_ids 长度不一致：{len(values)} vs {len(ids)}")
    out: Dict[str, list] = {}
    for v, p in zip(values, ids):
        out.setdefault(p, []).append(v)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def _stat_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else float("nan")


def patient_clustered_bootstrap(values: np.ndarray, patient_ids: Sequence[str],
                                n_boot: int = N_BOOTSTRAP, seed: int = 42,
                                ci_level: float = CI_LEVEL,
                                stat: Callable = _stat_mean) -> Dict[str, float]:
    """对逐窗口指标做患者级 bootstrap，返回点估计与百分位区间。

    每次重采样抽**患者**（有放回），把抽中患者的全部窗口并进来再算统计量。
    """
    if n_boot < 1:
        raise ValueError(f"n_boot 必须 >= 1，实际 {n_boot}")
    groups = _group(values, patient_ids)
    keys = sorted(groups)
    if not keys:
        raise ValueError("没有可用的患者分组")

    point = stat(np.concatenate([groups[k] for k in keys]))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        draws[b] = stat(np.concatenate([groups[keys[i]] for i in pick]))

    lo = float(np.percentile(draws, 100 * (1 - ci_level) / 2))
    hi = float(np.percentile(draws, 100 * (1 + ci_level) / 2))
    return {"point": float(point), "ci_low": lo, "ci_high": hi,
            "n_patients": len(keys), "n_observations": int(len(np.asarray(values))),
            "n_boot": n_boot, "seed": seed, "ci_level": ci_level,
            "stat": getattr(stat, "__name__", "custom")}


def paired_bootstrap_diff(values_a: np.ndarray, values_b: np.ndarray,
                          patient_ids: Sequence[str],
                          n_boot: int = N_BOOTSTRAP, seed: int = 42,
                          ci_level: float = CI_LEVEL,
                          stat: Callable = _stat_mean) -> Dict[str, float]:
    """配对 bootstrap：``A - B`` 的差异与其区间。

    两个方法**共享同一组重采样患者索引**——这是配对的意义所在。
    区间跨 0 表示在给定置信水平下无法把差异与抽样噪声区分开。

    注意符号：正值表示 **A 更差**（若指标为误差），
    因此 ``ci_high < 0`` 才意味着 A 显著更优。
    """
    if len(values_a) != len(values_b):
        raise ValueError(f"两组长度不一致：{len(values_a)} vs {len(values_b)}")
    ga = _group(values_a, patient_ids)
    gb = _group(values_b, patient_ids)
    if set(ga) != set(gb):
        raise ValueError("两组覆盖的患者集合不同，无法配对")

    keys = sorted(ga)
    point = stat(np.concatenate([ga[k] for k in keys])) - \
        stat(np.concatenate([gb[k] for k in keys]))

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        a_cat = np.concatenate([ga[keys[i]] for i in pick])
        b_cat = np.concatenate([gb[keys[i]] for i in pick])
        draws[b] = stat(a_cat) - stat(b_cat)

    lo = float(np.percentile(draws, 100 * (1 - ci_level) / 2))
    hi = float(np.percentile(draws, 100 * (1 + ci_level) / 2))
    # 双侧 bootstrap 的"跨越 0 的比例"作为粗略证据强度，不作为 p 值宣称
    frac_cross = float(np.mean(draws >= 0)) if point > 0 else float(np.mean(draws <= 0))
    return {"delta": float(point), "ci_low": lo, "ci_high": hi,
            "crosses_zero": bool(lo <= 0.0 <= hi),
            "frac_opposite_sign": frac_cross,
            "n_patients": len(keys), "n_boot": n_boot, "seed": seed,
            "ci_level": ci_level, "note": "正值表示 A 的指标更大（误差指标下即更差）"}


def summarize_seeds(per_seed_values: Dict[int, float]) -> Dict[str, float]:
    """把"多个训练随机种子"与"患者级 bootstrap"分开报告。

    两者含义不同，混在一起就没有意义：前者是训练不确定性，后者是抽样不确定性。
    """
    vals = np.asarray([v for v in per_seed_values.values() if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return {"n_seeds": 0, "mean": float("nan"), "sd": float("nan")}
    return {
        "n_seeds": int(vals.size),
        "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "seeds": sorted(int(s) for s in per_seed_values),
    }


def assert_paired_indexing_shared(values_a, values_b, patient_ids,
                                  n_boot: int = 50, seed: int = 0) -> None:
    """自检：配对实现确实对两个方法使用了同一组患者索引。

    做法是把 B 换成 A 的完全复制；此时差异应恒为 0，区间也应为 [0, 0]。
    若实现里两者各自独立重采样，区间不会塌缩到 0。
    """
    r = paired_bootstrap_diff(values_a, np.asarray(values_a, dtype=float),
                              patient_ids, n_boot=n_boot, seed=seed)
    if abs(r["delta"]) > 1e-12 or not (r["ci_low"] == 0.0 and r["ci_high"] == 0.0):
        raise AssertionError(
            f"配对索引未共享：delta={r['delta']}, CI=[{r['ci_low']}, {r['ci_high']}]"
        )
