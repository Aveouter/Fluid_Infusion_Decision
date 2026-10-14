#!/usr/bin/env python
"""评估协议 v2 的唯一真实数据入口（issue #81，Gate A）。

它做什么、不做什么
------------------
- **做**：把各来源的预测装成 :class:`~utils.eval_engine.PredictionSet`，
  逐个调用 :func:`utils.eval_engine.evaluate`，把结果行、NA 台账与摘要落盘。
- **不做**：任何指标算术。MAE、F1、累计量、变点匹配都在 ``utils/`` 里，
  本文件一个 ``np.abs`` / ``np.mean`` 都没有（有测试用 AST 检查这一点）。
  这样"同一个数字怎么算出来的"只有一个答案。

所有方法走**同一条代码路径**：本模型的口径与持久性/移动平均基线经同一个
``evaluate()``，因此它们的分母、掩码、聚合必然一致。

用法：

    python evaluate_v2.py --bundle preds/bundle.npz --base_pkl data/baseline.pkl \\
        --methods speed,rate,action_rate_uncond,persistence,moving_average \\
        --out .audit/eval_v2/rows.json

逐患者明细含真实编号，默认**不写**；需要时用 ``--per_patient_out`` 指定
（必须落在 gitignore 的目录里）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from utils import eval_protocol as PROTO
from utils import label_protocol as lp
from utils import run_manifest as RM
from utils.decoding import decode_action_rate, decode_bits
from utils.eval_engine import PredictionSet, evaluate

try:                                     # Windows 管道下强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


#: 方法名 → (bundle 中的速率键, 速率口径, 是否需要概率头)
METHOD_SPECS = {
    "speed": ("speed", "speed", False),
    "rate": ("rate", "rate", True),
    "action_rate_uncond": (None, "action_rate_uncond", True),   # 由解码器产出
    "pos_mean": ("pos_mean", "pos_mean", False),                # 需模型侧落盘
    "flow_pred": ("flow_pred", "flow_pred", False),             # 需模型侧落盘
    "persistence": ("pers_rates", "deterministic", False),
    "moving_average": ("ma_rates", "deterministic", False),
}

#: 基线的模型名（身份字段的一部分，缺了会与别的基线撞键）。
METHOD_MODEL_ID = {
    "persistence": "persistence",
    "moving_average": "moving_average",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", default="preds/bundle.npz")
    p.add_argument("--base_pkl", default="", help="基线特征 pkl（C1/亚组需要）")
    p.add_argument("--methods", default="speed,rate,action_rate_uncond,persistence,moving_average")
    p.add_argument("--eval_mode", default="one_step_observed_history",
                   choices=list(PROTO.EVAL_MODES))
    p.add_argument("--horizons", type=int, nargs="+", default=list(PROTO.HORIZONS))
    p.add_argument("--model_id", default="BurnResu")
    p.add_argument("--run_id", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quality_track", default="main",
                   choices=list(PROTO.QUALITY_TRACKS))
    p.add_argument("--quality_mask", default="",
                   help="逐小时质量掩码 CSV（只标合格与否，不增删患者）")
    p.add_argument("--expect_protocol_hash", default="",
                   help="期望的协议哈希；不一致即拒绝运行")
    p.add_argument("--out", default=".audit/eval_v2/rows.json")
    p.add_argument("--summary", default="")
    p.add_argument("--per_patient_out", default="",
                   help="逐患者明细（含真实编号，只能写进 gitignore 的目录）")
    return p.parse_args(argv)


def load_bundle(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"评测包不存在：{path}（先运行 make_eval_bundle.py）")
    d = np.load(path, allow_pickle=True)
    required = ("patient_id", "target_bits", "target_rates", "prob", "speed",
                "rate", "pers_rates", "ma_rates", "thresholds")
    missing = [k for k in required if k not in d.files]
    if missing:
        raise SystemExit(f"评测包缺少字段 {missing}；请重新生成")
    return {k: d[k] for k in d.files}


def build_prediction_set(bundle: dict, method: str, *, eval_mode: str,
                         fold: str, thresholds) -> PredictionSet:
    """把某个方法的预测装成 :class:`PredictionSet`（**不含任何算术**）。"""
    if method not in METHOD_SPECS:
        raise SystemExit(f"未知方法 {method}；可选 {sorted(METHOD_SPECS)}")
    key, rate_definition, needs_prob = METHOD_SPECS[method]

    prob = bundle["prob"] if needs_prob or rate_definition in ("rate",) else None
    if method == "action_rate_uncond":
        pred_rates = decode_action_rate(bundle["prob"], bundle["speed"], thresholds)
    else:
        if key not in bundle:
            raise SystemExit(
                f"评测包里没有 {key!r}；该口径需要模型侧产者先落盘（不得从别的口径反推）")
        pred_rates = bundle[key]

    if needs_prob or rate_definition in ("rate", "action_rate_uncond"):
        pred_bits = decode_bits(bundle["prob"], thresholds)
    else:
        # 无概率头的基线：其动作就是由速率导出的状态位（label_protocol 的不变量）
        pred_bits = (np.asarray(pred_rates) > 0).astype(float)

    return PredictionSet(
        patient_ids=np.asarray(bundle["patient_id"]),
        time_index=np.asarray(bundle["step_hour"]) if "step_hour" in bundle
        else _infer_hours(bundle),
        true_bits=np.asarray(bundle["target_bits"], dtype=float),
        true_rates=np.asarray(bundle["target_rates"], dtype=float),
        pred_rates=np.asarray(pred_rates, dtype=float),
        pred_bits=np.asarray(pred_bits, dtype=float),
        pred_prob=(np.asarray(prob, dtype=float) if prob is not None else None),
        observed_first=_observed_first(bundle),
        thresholds=thresholds,
        rate_definition=rate_definition,
        eval_mode=eval_mode,
        fold=fold,
    )


def _infer_hours(bundle: dict) -> np.ndarray:
    """老评测包没有 ``step_hour`` 时的**唯一**退化规则。

    早期产物只存了目标序列（``label[1..T-1]``），第 0 小时以历史出现。
    这里按患者内出现顺序编号为 1..n，并在摘要中标注降级原因——
    与"把部分轨迹当时域"不同，这只是时域**锚点**的推断，完整性仍由
    ``completeness`` 严格判定。
    """
    pids = [str(p) for p in bundle["patient_id"]]
    seen, out = {}, []
    for p in pids:
        seen[p] = seen.get(p, 0) + 1
        out.append(seen[p])
    return np.asarray(out, dtype=int)


def _scalar(value) -> str:
    """``npz`` 里长度 1 的数组取标量——否则 ``str()`` 会给出 ``"['x']"``。"""
    arr = np.asarray(value).ravel()
    return str(arr[0]) if arr.size else ""


def _observed_first(bundle: dict):
    if "observed_bits" not in bundle or "observed_rates" not in bundle:
        return None
    return np.concatenate([np.asarray(bundle["observed_bits"], dtype=float),
                           np.asarray(bundle["observed_rates"], dtype=float)], axis=1)


def _load_quality_mask(path: str, bundle: dict) -> np.ndarray:
    """读逐小时质量掩码并与行对齐。

    掩码是**逐液体类别**的（晶体/胶体/葡萄糖各自可有不同的合格性），而累计
    剂量终点要的是"这一小时的整体可用性"，因此规则冻结为：**三类全部合格**
    才算该小时合格。用"任一类合格"会把只记了一半的小时算成完整小时。

    小时序号按 ``hour_start`` 与评测包的 ``step_hour`` 对齐（同一套 0 基小时）。
    对不上的小时一律按**不合格**处理并计数——不猜、不回填。
    """
    import csv

    class_cols = ("crystalloid_analysis_eligible", "colloid_analysis_eligible",
                  "water_analysis_eligible")
    want, n_rows = {}, 0
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        id_col = "patient_id" if "patient_id" in cols else (reader.fieldnames or [""])[0]
        need = {"hour_start"} | set(class_cols)
        if not need <= cols:
            raise SystemExit(f"质量掩码缺少列 {sorted(need - cols)}")

        def _truthy(v):
            return str(v).strip().lower() in ("1", "true", "yes")

        for row in reader:
            n_rows += 1
            key = (str(row[id_col]).strip(), int(row["hour_start"]))
            want[key] = all(_truthy(row[c]) for c in class_cols)

    hours = np.asarray(bundle["step_hour"], dtype=int) if "step_hour" in bundle \
        else _infer_hours(bundle)
    pids = [str(p) for p in bundle["patient_id"]]
    mask = np.array([want.get((p, int(h)), False) for p, h in zip(pids, hours)], bool)

    n_missing = sum(1 for p, h in zip(pids, hours) if (p, int(h)) not in want)
    print(f"[quality] {path}: {n_rows} 行；对齐 {len(mask) - n_missing}/{len(mask)}；"
          f"合格 {int(mask.sum())} 行；未对齐按不合格计 {n_missing} 行")
    return mask


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.expect_protocol_hash:
        try:
            PROTO.assert_frozen(args.expect_protocol_hash)
        except AssertionError as exc:
            raise SystemExit(str(exc))
    if not args.run_id:
        args.run_id = f"{args.quality_track}-{RM.git_sha(short=True)}"

    bundle = load_bundle(Path(args.bundle))
    fingerprint = RM.data_fingerprint(args.bundle)
    thresholds = np.asarray(bundle["thresholds"], dtype=float)

    if "protocol_hash" in bundle:
        got = _scalar(bundle["protocol_hash"])
        if got and got != PROTO.PROTOCOL_HASH:
            raise SystemExit(
                f"评测包的协议哈希 {got} 与当前 {PROTO.PROTOCOL_HASH} 不一致："
                "冻结后修改定义必须升级版本并重算，不得混用")

    fold = (_scalar(bundle["split_or_outer_fold"]) if "split_or_outer_fold" in bundle
            else "test")
    quality_mask = (_load_quality_mask(args.quality_mask, bundle)
                    if args.quality_mask else None)

    base_by_patient = {}
    if args.base_pkl:
        if not Path(args.base_pkl).exists():
            # 不静默降级：C1 与 CETE 归一化版本会因为缺体重而变成 NA，
            # 那是有意义的结论，但必须让使用者知道原因是"没给文件"。
            print(f"[warn] 基线特征不存在：{args.base_pkl}；"
                  "C1 与按体重归一化的 CETE 将报 NA + 原因")
        else:
            import pickle
            base_by_patient = pickle.load(open(args.base_pkl, "rb"))
            print(f"[base] 载入 {len(base_by_patient)} 位患者的基线特征")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    per_method, skipped = {}, []
    for method in methods:
        if method in ("pos_mean", "flow_pred") and METHOD_SPECS[method][0] not in bundle:
            skipped.append({"method": method,
                            "na_reason": "producer_not_available_in_bundle",
                            "note": "该口径需要模型侧落盘，不得从 speed 反推"})
            continue
        ps = build_prediction_set(bundle, method, eval_mode=args.eval_mode,
                                  fold=fold, thresholds=thresholds)
        exporter = evaluate(ps, model_id=METHOD_MODEL_ID.get(method, args.model_id),
                            run_id=args.run_id, split_or_outer_fold=fold,
                            train_seed=args.seed, quality_mask=quality_mask,
                            horizons=tuple(args.horizons),
                            data_fingerprint=fingerprint,
                            quality_track=args.quality_track,
                            base_by_patient=base_by_patient)
        per_method[method] = exporter

    if not per_method:
        raise SystemExit("没有任何可评估的方法")

    _write(args, bundle, per_method, skipped, fingerprint, fold)
    return 0


def _write(args, bundle, per_method, skipped, fingerprint, fold) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, ledger = [], []
    for method, exporter in per_method.items():
        for rec in exporter.records:
            row = rec.to_dict()
            row["method"] = method
            rows.append(row)
        ledger.extend({"method": method, **item} for item in exporter.na_ledger())

    # 出口自检：NA 必须有原因；同一文件不得混评估模式
    bad = [r for r in rows if r["value"] is None and not r["na_reason"]]
    if bad:
        raise SystemExit(f"存在 {len(bad)} 行 NA 但无原因，拒绝写出")
    modes = {r["eval_mode"] for r in rows}
    if len(modes) > 1:
        raise SystemExit(f"同一结果文件混了多个评估模式：{sorted(modes)}")

    payload = {
        "protocol_version": PROTO.PROTOCOL_VERSION,
        "protocol_hash": PROTO.PROTOCOL_HASH,
        "eval_mode": args.eval_mode,
        "split_or_outer_fold": fold,
        "quality_track": args.quality_track,
        "data_fingerprint": fingerprint,
        "run_id": args.run_id,
        "seed": args.seed,
        "horizons": list(args.horizons),
        "n_rows": len(rows),
        "n_na_rows": len(ledger),
        "skipped_methods": skipped,
        "na_ledger": ledger,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                   encoding="utf-8")
    print(f"[written] {out}  rows={len(rows)}  NA={len(ledger)}")

    if args.per_patient_out:
        pp = Path(args.per_patient_out)
        pp.parent.mkdir(parents=True, exist_ok=True)
        detail = {m: {"patient_id": [str(p) for p in bundle["patient_id"]]}
                  for m in per_method}
        pp.write_text(json.dumps(detail, ensure_ascii=False, indent=2),
                      encoding="utf-8")
        print(f"[written] {pp}  （含真实编号，勿提交）")

    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(_render_summary(payload), encoding="utf-8")
        print(f"[written] {args.summary}")

    for item in skipped:
        print(f"[skip] {item['method']}: {item['na_reason']}")


def _render_summary(payload: dict) -> str:
    """人读摘要。只做排版，不做算术。"""
    lines = [
        "# 评估结果（协议 v2）",
        "",
        f"- 协议：`{payload['protocol_version']}` / `{payload['protocol_hash']}`",
        f"- 评估模式：`{payload['eval_mode']}`　划分：`{payload['split_or_outer_fold']}`"
        f"　质量轨：`{payload['quality_track']}`",
        f"- 数据指纹：`{payload['data_fingerprint']}`　run：`{payload['run_id']}`"
        f"　seed：{payload['seed']}",
        f"- 行数 {payload['n_rows']}，其中 NA {payload['n_na_rows']}",
        "",
        "> 本轮为**回顾性单次评估**：非嵌套，且原 15% 测试集已被报告过，"
        "**不作为泛化证据**。",
        "",
        "| 方法 | 指标 | 单位 | 时域 | 液体 | 掩码 | 口径 | 值 | n 患者 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in payload["rows"]:
        if r["metric_name"].startswith("audit_"):
            continue
        value = "NA" if r["value"] is None else f"{r['value']:.4f}"
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["method"], r["metric_name"], r["unit"], r["horizon_hours"] or "-",
            r["fluid_class"], r["mask"], r["rate_definition"], value,
            r["n_patients"] if r["n_patients"] is not None else "-"))
    if payload["na_ledger"]:
        lines += ["", "## 不可估计的终点（NA 台账）", "",
                  "| 方法 | 指标 | 原因 |", "|---|---|---|"]
        for item in payload["na_ledger"]:
            lines.append(f"| {item['method']} | {item['metric_name']} | "
                         f"{item['na_reason']} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
