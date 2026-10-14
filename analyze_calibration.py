#!/usr/bin/env python
"""分类校准与不确定性（issue #23）。只输出脱敏聚合指标。"""
from __future__ import annotations
import argparse
import numpy as np
from utils import calibration as CA

NAMES = ("crystal", "colloid", "water")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="preds/bundle.npz")
    a = ap.parse_args(argv)

    d = np.load(a.bundle, allow_pickle=True)
    y, p = d["target_bits"], d["prob"]

    print(f"{'类别':<10}{'阳性率':>9}{'平均置信':>10}{'过度自信':>10}{'Brier':>9}{'ECE':>9}")
    print("-" * 60)
    for k, n in enumerate(NAMES):
        g = CA.overconfidence_gap(y[:, k:k + 1], p[:, k:k + 1])["class_0"]
        b = CA.brier_per_class(y[:, k:k + 1], p[:, k:k + 1])["mean"]
        e = CA.expected_calibration_error(y[:, k], p[:, k])["ece"]
        print(f"{n:<10}{y[:, k].mean():>9.3f}{p[:, k].mean():>10.3f}"
              f"{g:>10.3f}{b:>9.3f}{e:>9.3f}")

    b = CA.brier_per_class(y, p)
    e = CA.expected_calibration_error(y.ravel(), p.ravel())
    print(f"{'整体':<10}{y.mean():>9.3f}{p.mean():>10.3f}"
          f"{p.mean() - y.mean():>10.3f}{b['mean']:>9.3f}{e['ece']:>9.3f}")
    print()
    print("注：ECE 正且大 = 过度自信。区间类型区分见")
    print("   sampling    :", CA.describe_interval_kind("sampling"))
    print("   predictive  :", CA.describe_interval_kind("predictive"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
