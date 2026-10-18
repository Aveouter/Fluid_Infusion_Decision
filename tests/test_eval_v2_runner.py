"""issue #81 / RED：统一评估入口的行为。

对应审稿意见：所有模型与基线必须走**同一个**评估函数、同一套分母；不可
估计的终点必须带原因出现在 NA 台账里；协议哈希不一致必须拒绝运行。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from tests import eval_fixtures as F
from utils import eval_protocol as P

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "evaluate_v2.py"


def _synthetic_bundle(path: Path, *, n_patients: int = 4, hours: int = 24) -> Path:
    """造一份结构与真实产物同形、内容全部虚构的评测包。"""
    rng = np.random.default_rng(0)
    pids, step, obs_b, obs_r = [], [], [], []
    tb, tr, prob, speed = [], [], [], []
    for i in range(n_patients):
        pid = f"{F.SYNTH}{i:02d}"
        base = rng.random(3) * 50.0
        for h in range(1, hours + 1):        # 目标是 label[1..T-1]
            pids.append(pid)
            step.append(h)
            row = (rng.random(3) < 0.4) * base
            tb.append((row > 0).astype(float))
            tr.append(row)
            obs_b.append(np.zeros(3))
            obs_r.append(np.full(3, 10.0))
            pr = np.clip(rng.random(3), 0.05, 0.95)
            prob.append(pr)
            speed.append(rng.random(3) * 60.0)
    share = {"patient_id": np.array(pids), "step_hour": np.array(step),
             "observed_bits": np.stack(obs_b), "observed_rates": np.stack(obs_r),
             "target_bits": np.stack(tb), "target_rates": np.stack(tr),
             "prob": np.stack(prob), "speed": np.stack(speed)}
    share["rate"] = share["prob"] * share["speed"]
    pers = np.stack([np.concatenate([b[:3], r]) for b, r in
                     zip(share["target_bits"], share["target_rates"])])
    share["pers_bits"], share["pers_rates"] = pers[:, :3], pers[:, 3:]
    share["ma_bits"], share["ma_rates"] = pers[:, :3], pers[:, 3:]
    share["thresholds"] = np.array([0.45, 0.30, 0.55])
    share["ma_k"] = np.array([3])
    share["split_or_outer_fold"] = np.array(["test"])
    share["split_seed"] = np.array([42])
    share["protocol_hash"] = np.array([P.PROTOCOL_HASH])
    np.savez_compressed(path, **share)
    return path


@pytest.fixture()
def bundle(tmp_path):
    return _synthetic_bundle(tmp_path / "bundle.npz")


def _run(bundle, out, *extra):
    import evaluate_v2
    return evaluate_v2.main([
        "--bundle", str(bundle), "--out", str(out),
        "--methods", "speed,rate,action_rate_uncond,persistence,moving_average",
        "--horizons", "24", *extra])


# --------------------------------------------------------- 基本行为

def test_runner_produces_rows_for_every_method(bundle, tmp_path):
    out = tmp_path / "rows.json"
    assert _run(bundle, out) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    methods = {r["method"] for r in payload["rows"]}
    assert methods == {"speed", "rate", "action_rate_uncond",
                       "persistence", "moving_average"}
    assert payload["protocol_hash"] == P.PROTOCOL_HASH


def test_every_method_goes_through_the_same_metric_names(bundle, tmp_path):
    """同一批终点必须对每个方法都出现——否则分母与口径会各说各话。"""
    out = tmp_path / "rows.json"
    _run(bundle, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    by_method = {}
    for r in payload["rows"]:
        by_method.setdefault(r["method"], set()).add(r["metric_name"])
    base = by_method["rate"]
    for method, names in by_method.items():
        assert names == base, f"{method} 的终点集合与 rate 不一致"


def test_na_ledger_covers_every_na_row(bundle, tmp_path):
    """NA 台账必须与行一一对应，且每行都有原因。"""
    out = tmp_path / "rows.json"
    _run(bundle, out)
    payload = json.loads(out.read_text(encoding="utf-8"))

    na_rows = [(r["method"], r["metric_name"]) for r in payload["rows"]
               if r["value"] is None]
    ledger = [(i["method"], i["metric_name"]) for i in payload["na_ledger"]]
    assert sorted(na_rows) == sorted(ledger)
    assert all(i["na_reason"] for i in payload["na_ledger"]), "NA 必须带原因"
    assert payload["n_na_rows"] == len(na_rows)


def test_a_single_result_file_never_mixes_eval_modes(bundle, tmp_path):
    out = tmp_path / "rows.json"
    _run(bundle, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert {r["eval_mode"] for r in payload["rows"]} == {
        payload["eval_mode"]}


def test_rows_carry_the_full_identity(bundle, tmp_path):
    out = tmp_path / "rows.json"
    _run(bundle, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    for r in payload["rows"]:
        for key in ("model_id", "run_id", "quality_track", "rate_definition",
                    "estimand", "horizon_origin", "split_or_outer_fold",
                    "train_seed", "protocol_hash", "data_fingerprint",
                    "identity_key"):
            assert key in r, f"结果行缺少身份字段 {key}"
        assert r["data_fingerprint"].startswith("bundle.npz:")


def test_identity_keys_are_unique_within_a_file(bundle, tmp_path):
    """同一份结果里不得出现两个身份相同却数值不同的行。"""
    out = tmp_path / "rows.json"
    _run(bundle, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    seen = {}
    for r in payload["rows"]:
        key = (r["method"], r["identity_key"])
        assert key not in seen, f"身份重复：{key}"
        seen[key] = r["value"]


# --------------------------------------------------------- 门禁与拒绝

def test_protocol_hash_mismatch_refuses_to_run(bundle, tmp_path):
    """冻结后换了定义就不能再拿旧产物算——必须硬失败。"""
    import evaluate_v2
    with pytest.raises(SystemExit, match="协议 hash 不匹配"):
        evaluate_v2.main([
            "--bundle", str(bundle), "--out", str(tmp_path / "o.json"),
            "--expect_protocol_hash", "0000000000000000"])


def test_bundle_with_a_different_protocol_hash_is_rejected(tmp_path):
    """评测包自带的哈希与当前不一致时也要拒绝。"""
    path = _synthetic_bundle(tmp_path / "b.npz")
    d = dict(np.load(path, allow_pickle=True))
    d["protocol_hash"] = np.array(["deadbeefdeadbeef"])
    np.savez_compressed(path, **d)
    import evaluate_v2
    with pytest.raises(SystemExit, match="协议哈希"):
        evaluate_v2.main(["--bundle", str(path),
                          "--out", str(tmp_path / "o.json")])


def test_missing_producer_is_skipped_with_a_reason_not_faked(tmp_path, bundle):
    """``pos_mean`` 需要模型侧产者；不能从 speed 反推，只能跳过并说明。"""
    import evaluate_v2
    out = tmp_path / "o.json"
    evaluate_v2.main(["--bundle", str(bundle), "--out", str(out),
                      "--methods", "pos_mean,rate", "--horizons", "24"])
    payload = json.loads(out.read_text(encoding="utf-8"))
    skipped = {i["method"]: i["na_reason"] for i in payload["skipped_methods"]}
    assert skipped == {"pos_mean": "producer_not_available_in_bundle"}
    assert {r["method"] for r in payload["rows"]} == {"rate"}


def test_unknown_method_is_rejected(bundle, tmp_path):
    import evaluate_v2
    with pytest.raises(SystemExit, match="未知方法"):
        evaluate_v2.main(["--bundle", str(bundle), "--out", str(tmp_path / "o.json"),
                          "--methods", "definitely_not_a_method"])


# --------------------------------------------- 入口不含指标算术（AST 检查）

def test_runner_contains_no_metric_arithmetic():
    """入口只做装配与排版。

    一旦有人在 ``evaluate_v2.py`` 里直接算 MAE/F1，就会出现第二条计算路径，
    而那正是"同名指标多种算法"重新长出来的方式。
    """
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    banned = {"abs", "mean", "median", "sqrt", "sum", "cumsum", "percentile"}
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in banned:
            if isinstance(node.value, ast.Name) and node.value.id == "np":
                offenders.append(f"{node.value.id}.{node.attr} (line {node.lineno})")
    assert not offenders, f"评估入口里出现了指标算术：{offenders}"


def test_engine_contains_no_metric_arithmetic():
    """引擎同样只做命名与门禁——算术全部在 utils/endpoints.py 等处。"""
    engine = ROOT / "utils" / "eval_engine.py"
    tree = ast.parse(engine.read_text(encoding="utf-8"))
    banned = {"abs", "mean", "median", "sqrt", "cumsum"}
    offenders = [f"np.{n.attr} (line {n.lineno})" for n in ast.walk(tree)
                 if isinstance(n, ast.Attribute) and n.attr in banned
                 and isinstance(n.value, ast.Name) and n.value.id == "np"]
    assert not offenders, f"评估引擎里出现了指标算术：{offenders}"


# ------------------------------------------------------------------ 隐私

def test_runner_does_not_read_real_data_paths():
    """CI 上不能碰真实临床数据；评测包的路径必须由参数给出。"""
    text = RUNNER.read_text(encoding="utf-8")
    assert '"data/' not in text.replace('--base_pkl', ''), \
        "评估入口不得硬编码 data/ 下的路径"
