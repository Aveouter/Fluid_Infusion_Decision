"""issue #8 / RED：运行清单、checkpoint 溯源与跨运行汇总守卫。

对应审稿意见：表 2–4 的每个数字都必须能定位到产生它的运行、权重与 split，
无法复现的必须标为"未复现"，不得虚构。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils import run_manifest as RM

ROOT = Path(__file__).resolve().parent.parent
CKPTS = ROOT / "ckpts"

# 随包记录的两个"全模型"权重
CKPT_A = CKPTS / "tcn_film_best.pth"
CKPT_B = CKPTS / "tcn_film_best_f1_0.7203_mae_21.3136.pth"
TEST_METRICS = CKPTS / "test_metrics.json"


# ------------------------------------------------------------ 清单 schema

def _complete_manifest() -> RM.RunManifest:
    return RM.RunManifest(
        run_id="run-0001", purpose="train", model_name="tcn_film",
        seed=42, split_seed=42, splits=[0.7, 0.15, 0.15],
        checkpoint={"name": "ck.pth", "bytes": 1, "sha256_16": "deadbeef"},
        data=[{"name": "output_data.pkl", "bytes": 1, "sha256_16": "cafebabe"}],
    )


def test_result_manifest_schema():
    """完整清单必须通过校验，并带齐复现所需的字段。"""
    m = _complete_manifest()
    m.validate()

    d = json.loads(m.to_json())
    for key in ("run_id", "purpose", "model_name", "git_sha", "created_at",
                "seed", "split_seed", "split_mode", "splits", "hyperparams",
                "early_stopping", "checkpoint", "data", "env"):
        assert key in d, f"清单缺少字段 {key}"
    assert d["env"]["label_protocol_version"]
    assert d["env"]["metrics_protocol_version"]


@pytest.mark.parametrize("drop,expect", [
    ("run_id", "run_id"),
    ("model_name", "model_name"),
    ("seed", "seed"),
    ("split_seed", "划分 seed"),
    ("splits", "split 比例"),
    ("checkpoint", "checkpoint"),
    ("data", "数据指纹"),
])
def test_manifest_rejects_missing_provenance(drop, expect):
    """缺任何一项关键溯源信息都必须报错，而不是静默通过。"""
    m = _complete_manifest()
    if drop in ("splits", "data"):
        setattr(m, drop, None if drop == "splits" else [])
    else:
        setattr(m, drop, None)
    with pytest.raises(ValueError, match=expect):
        m.validate()


def test_manifest_records_environment():
    env = RM.environment()
    assert env["python"]
    assert env["manifest_version"] == RM.MANIFEST_VERSION
    assert "torch" in env


def test_manifest_write_and_read(tmp_path):
    m = _complete_manifest()
    path = tmp_path / "runs" / "run-0001.json"
    m.write(path)
    back = RM.load_manifest(path)
    assert back["run_id"] == "run-0001"
    assert back["splits"] == [0.7, 0.15, 0.15]


# --------------------------------------------------------- 指纹不含内容

def test_fingerprint_records_no_content(tmp_path):
    """指纹只记名字、字节数与哈希，不得写入文件内容。"""
    p = tmp_path / "output_data.pkl"
    p.write_bytes(b"SYNTHETIC-SECRET-CONTENT")

    fp = RM.fingerprint(p)
    assert fp["name"] == "output_data.pkl"
    assert fp["bytes"] == len(b"SYNTHETIC-SECRET-CONTENT")
    assert len(fp["sha256_16"]) == 16
    assert "SECRET" not in json.dumps(fp)


def test_fingerprint_missing_file_returns_none(tmp_path):
    assert RM.fingerprint(tmp_path / "nope.pkl") is None


# ------------------------------------------------ 跨运行汇总守卫

def test_no_cross_run_join():
    """指标行必须带可解析的 run_id，否则拒绝汇总。"""
    manifests = {"run-1": {"model_name": "tcn_film", "git_sha": "abc",
                           "seed": 42, "split": "test"}}

    ok = RM.join_metrics(manifests, {"f1": {"value": 0.7, "run_id": "run-1"}})
    assert ok[0]["run_id"] == "run-1"

    with pytest.raises(ValueError, match="缺少 run_id"):
        RM.join_metrics(manifests, {"f1": {"value": 0.7}})

    with pytest.raises(ValueError, match="未知的 run_id"):
        RM.join_metrics(manifests, {"f1": {"value": 0.7, "run_id": "run-9"}})

    with pytest.raises(ValueError, match="不一致"):
        RM.join_metrics(manifests,
                        {"f1": {"value": 0.7, "run_id": "run-1", "seed": 7}})


# ------------------------------------ 随包权重与记录的溯源（需要真实 ckpts）

def _require_ckpts():
    if not (CKPT_A.exists() and TEST_METRICS.exists()):
        pytest.skip("缺少 ckpts/tcn_film_best.pth 或 test_metrics.json")


# ------------------------------------------------------ 溯源台账完整性

LEDGER = ROOT / "docs" / "table_provenance.md"

#: 表 2–4 中出现的每一行（按论文中的名字）
TABLE_ROWS = [
    # 表 2
    "Regression", "Random Forest", "SVM", "XG-Boost", "GBR", "Salinas",
    "LSTM", "CNN", "Transformer",
    "BurnResu w/o Baseline Enc.", "BurnResu w/o Temporal Enc.", "BurnResu (Full)",
    # 表 3
    "w/o Temporal Enc.", "w/o Baseline Enc.", "w/o self-attention",
    "w/o TCN", "w/o FiLM", "Full",
    # 表 4
    "Crystalloid", "Colloid", "Glucose", "Macro Average",
]

#: 台账允许的状态取值
LEDGER_STATUS = ("reconciled", "val-only", "inconsistent", "no-artefact")


def test_provenance_ledger_covers_every_table_row():
    """表 2–4 的每一行都必须在台账里有条目，否则无法逐项回复审稿人。"""
    assert LEDGER.exists(), "缺少 docs/table_provenance.md"
    text = LEDGER.read_text(encoding="utf-8")

    missing = [r for r in TABLE_ROWS if r not in text]
    assert not missing, f"台账缺少以下行：{missing}"


def test_provenance_ledger_uses_defined_status_vocabulary():
    """台账中的状态必须取自定义好的词表，避免自造模糊说法。"""
    text = LEDGER.read_text(encoding="utf-8")
    for status in LEDGER_STATUS:
        assert status in text, f"台账未使用状态 `{status}`"

    # 每一行都应在表格里出现，且同一行内标注了状态
    for row in TABLE_ROWS:
        assert f"`{row}`" in text or f"| {row} " in text, f"{row} 未出现在表格行中"


def test_ledger_records_the_inconsistent_rows():
    """已知无法对账的行必须显式标为 inconsistent，不能含糊带过。"""
    text = LEDGER.read_text(encoding="utf-8")
    for row in ("BurnResu (Full)", "BurnResu w/o Baseline Enc."):
        assert row in text
    assert text.count("`inconsistent`") >= 3, "inconsistent 标记数量少于预期"


@pytest.mark.realdata
def test_parameter_count_reconciles_with_recorded_metrics():
    """state_dict 含 buffer，参数量口径必须扣除后才与记录一致。

    ``test_metrics.json`` 记录 ``model_params.total = 5,173,646``；
    ``tcn_film_best.pth`` 的 state_dict 直接求和是 5,697,934。
    差值 524,288 恰好是 ``posenc.pe`` (2048, 256) —— 一个注册 buffer，
    不计入 ``model.parameters()``。扣除后两者完全相等，
    说明该 checkpoint 与这份指标记录确实同源。
    """
    _require_ckpts()
    import torch

    ck = torch.load(CKPT_A, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    recorded = json.load(open(TEST_METRICS, encoding="utf-8"))["model_params"]["total"]

    total_with_buffers = RM.state_dict_tensor_count(sd)
    buffer_numel = sum(v.numel() for k, v in sd.items()
                       if k.endswith("posenc.pe"))
    params_only = total_with_buffers - buffer_numel

    assert params_only == recorded, (
        f"扣除 buffer 后参数为 {params_only}，记录为 {recorded}"
    )
    assert buffer_numel == 2048 * 256


@pytest.mark.realdata
def test_trainable_parameter_counter_excludes_buffers():
    """``count_parameters`` 对 nn.Module 只数可训练参数，不含 buffer。"""
    _require_ckpts()
    import torch

    from models.tcn_film import build_tcnfilm_full

    model = build_tcnfilm_full(
        input_dim_base=10, input_dim_temporal=19, embed_dim=256,
        num_classes=3, pred_length=1, n_transformer_layers=4,
        n_heads=4, tcn_layers_short=3, tcn_layers_long=5,
    )
    n = RM.count_parameters(model)
    # 与 record 同一数量级且不含 buffer
    assert n > 0
    assert n < RM.state_dict_tensor_count(model.state_dict())


@pytest.mark.realdata
def test_two_full_model_checkpoints_are_different_runs():
    """特征化：随包的两个"全模型"权重来自不同 epoch，内容不同。

    这不是期望状态，而是 #8 记录在案的歧义：

        tcn_film_best.pth                      epoch 11, best_f1 0.718953
        tcn_film_best_f1_0.7203_mae_21.3136.pth epoch 15, best_f1 0.720300

    而 ``training_metadata.json`` 与 ``test_metrics.json`` 指向前者，
    ``inference.py`` 的默认 ``--ckpt`` 指向后者。文件名里的指标与**后者**
    自洽，因此"论文的结果由哪一个产生"无法从产物本身判定，
    必须在 #26 修表时明确。

    一旦二者被统一（或其中一个被移除），本用例应当失败并被改写。
    """
    _require_ckpts()
    if not CKPT_B.exists():
        pytest.skip("第二个全模型权重不存在")
    import torch

    a = torch.load(CKPT_A, map_location="cpu", weights_only=False)
    b = torch.load(CKPT_B, map_location="cpu", weights_only=False)

    assert a["epoch"] != b["epoch"], "两个权重的 epoch 相同，前提已变"
    assert abs(a["best_f1"] - b["best_f1"]) > 1e-6, "best_f1 相同，前提已变"

    # 文件名与 B 自洽
    assert b["best_f1"] == pytest.approx(0.7203, abs=1e-4)
    assert b["best_mae"] == pytest.approx(21.3136, abs=1e-4)

    # 权重内容确实不同
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert not all(torch.equal(sa[k], sb[k]) for k in sa)


@pytest.mark.realdata
def test_recorded_metadata_matches_checkpoint_a():
    """``training_metadata.json`` 的 best_f1/best_mae 与 checkpoint A 一致。"""
    _require_ckpts()
    import torch

    meta_path = CKPTS / "training_metadata.json"
    if not meta_path.exists():
        pytest.skip("缺少 training_metadata.json")

    meta = json.load(open(meta_path, encoding="utf-8"))
    a = torch.load(CKPT_A, map_location="cpu", weights_only=False)

    assert a["best_f1"] == pytest.approx(meta["best_f1"], abs=1e-9)
    assert a["best_mae"] == pytest.approx(meta["best_mae"], abs=1e-6)
    assert meta["best_ckpt_path"].endswith("tcn_film_best.pth")
