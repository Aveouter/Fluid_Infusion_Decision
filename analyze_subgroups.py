#!/usr/bin/env python
"""逐患者误差、失败病例与临床亚组（issue #22）。

.. warning::
   本脚本的输出**包含真实患者编号**（用于本地定位失败病例与绘图选例）。
   按 roadmap 的隐私规则，**不得**把含编号的输出粘贴进 issue、commit 或
   任何会推送的内容；对外只发布**脱敏聚合指标**（分组样本量与误差统计）。

   本地运行示例：
       python analyze_subgroups.py > .audit/subgroups.txt
"""
from __future__ import annotations
import argparse, pickle
from pathlib import Path
import numpy as np
from utils import subgroups as SG

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="preds/bundle.npz")
    ap.add_argument("--base_pkl", default="data/baseline.pkl")
    ap.add_argument("--threshold", type=float, default=100.0)
    a = ap.parse_args(argv)

    d = np.load(a.bundle, allow_pickle=True)
    tr, rate = d["target_rates"], d["rate"]
    pids = [str(x) for x in d["patient_id"]]
    base = pickle.load(open(a.base_pkl, "rb"))

    s = SG.per_patient_errors(rate, tr, pids)
    print("== 逐患者误差分布（正流，mL/h）==")
    print(f"  患者数 {s['n_patients']}  均值 {s['patient_mean']:.1f}  "
          f"中位 {s['patient_p50']:.1f}  P90 {s['patient_p90']:.1f}  最差 {s['patient_max']:.1f}")
    print(f"  单元级 MAE {s['cell_mae']:.1f}  P90 {s['cell_p90']:.1f}")

    fails = SG.failure_patients(rate, tr, pids, threshold=a.threshold)
    print(f"\n== 失败病例（逐患者平均误差 > {a.threshold:.0f} mL/h）==")
    print(f"  {len(fails)} / {s['n_patients']} 位患者  "
          f"({len(fails)/s['n_patients']:.0%})")
    for r in fails[:5]:
        print(f"    {r['patient_id']:<12} 平均误差 {r['mean_abs_error']:7.1f} mL/h  "
              f"({r['n_cells']} 个正流单元)")

    print("\n== 亚组（先按患者聚合）==")
    labels_by_pid = {}
    for pid, row in base.items():
        try:
            labels_by_pid[str(pid)] = SG.subgroup_labels(row)
        except Exception:
            continue
    per_pat = {}
    for i, pid in enumerate(pids):
        m = tr[i] > 0
        if m.sum():
            per_pat.setdefault(pid, []).extend(np.abs(rate[i][m] - tr[i][m]).tolist())
    for dim in ("sex", "age", "tbsa", "inhalation"):
        vals = np.array([np.mean(v) for v in per_pat.values()])
        keys = list(per_pat.keys())
        labs = [labels_by_pid.get(p, {}).get(dim, "unknown") for p in keys]
        g = SG.group_metrics(vals, keys, labs)
        print(f"\n  [{dim}]")
        for lab, v in g.items():
            flag = "  <-- 样本小" if v["unstable"] else ""
            print(f"    {lab:<12} n={v['n_patients']:>3}  "
                  f"均值 {v['mean']:7.1f}  中位 {v['p50']:7.1f}  P90 {v['p90']:7.1f}{flag}")

    c = SG.select_cases(np.array([np.mean(v) for v in per_pat.values()]),
                        list(per_pat.keys()), n_each=2)
    print("\n== 图 6 选例（确定性规则：最好 / 中位 / 最差各 2 例）==")
    for bucket in ("best", "median", "worst"):
        print(f"  {bucket:<8} " + ", ".join(f"{p}({v:.1f})" for p, v in c[bucket]))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
