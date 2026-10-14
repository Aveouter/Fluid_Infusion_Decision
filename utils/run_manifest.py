"""运行清单（run manifest）：让每个结果都能回溯到产生它的那次运行。

背景（issue #8）
---------------
随包产物里同一个"全模型"存在多套互不相同的数字：

- ``tcn_film_best.pth``                -> epoch 11, best_f1 0.718953, best_mae 22.0863
- ``tcn_film_best_f1_0.7203_mae_21.3136.pth`` -> epoch 15, best_f1 0.720300, best_mae 21.3136
- ``training_metadata.json``           -> 记录的是前者
- ``test_metrics.json``                -> ``checkpoint`` 字段指向前者
- ``inference.py`` 的默认 ``--ckpt``    -> 指向后者

两个文件是不同 epoch 的不同权重，因此"论文用的是哪一个"无法从产物本身判定。
本模块提供机器可读的清单，把「哪次运行、哪份权重、哪个 split、哪个指标口径」
固定下来，避免同类歧义。

隐私：数据只记录**指纹**（文件名 + 大小 + sha256 前 16 位），不记录内容，
也不写入任何患者标识。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from utils import eval_protocol, label_protocol, metrics as metrics_protocol

MANIFEST_VERSION = "1.0"


def git_sha(short: bool = False) -> str:
    """当前仓库的 git SHA；不可用时返回 "unknown"。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short" if short else "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def is_dirty() -> Optional[bool]:
    """工作区是否有未提交改动；无法判断时返回 None。"""
    try:
        out = subprocess.run(["git", "status", "--porcelain"],
                             capture_output=True, text=True, timeout=10)
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        return None


def fingerprint(path) -> Optional[dict]:
    """文件指纹：名字、字节数、sha256 前 16 位。**不读取内容进清单。**"""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"name": p.name, "bytes": p.stat().st_size, "sha256_16": h.hexdigest()[:16]}


def data_fingerprint(path) -> str:
    """数据指纹的**短串**形式：``"bundle.npz:1a2b3c4d5e6f7788"``。

    结果行的 ``data_fingerprint`` 字段用它——逐行塞一整个字典会让导出难以
    阅读，而这个短串足以回答"这一行的数字是读哪份文件算出来的"。
    文件不存在时返回空串（调用方据此写 NA + 原因，而不是编一个指纹）。
    """
    fp = fingerprint(path)
    if fp is None:
        return ""
    return f"{fp['name']}:{fp['sha256_16']}"


def count_parameters(model) -> int:
    """模型的**可训练参数**数（与 ``run_epoch`` 报告的口径一致）。

    注意：``state_dict`` 里还可能含 buffer（如位置编码 ``posenc.pe``），
    把 state_dict 直接求和会得到偏大的数字——``tcn_film_best.pth`` 即如此：
    state_dict 合计 5,697,934，其中 524,288 来自 ``posenc.pe`` (2048, 256)，
    扣除后正是 ``test_metrics.json`` 记录的 5,173,646。
    """
    import torch

    if isinstance(model, torch.nn.Module):
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if isinstance(model, dict):
        return int(sum(v.numel() for v in model.values()
                       if hasattr(v, "numel") and getattr(v, "requires_grad", False)))
    raise TypeError("model 应为 nn.Module 或 state_dict")


def state_dict_tensor_count(state_dict: Dict) -> int:
    """state_dict 中所有张量的元素总数（含 buffer）。"""
    return int(sum(v.numel() for v in state_dict.values() if hasattr(v, "numel")))


def environment() -> dict:
    """记录可复现所需的环境信息。"""
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "manifest_version": MANIFEST_VERSION,
        "label_protocol_version": label_protocol.PROTOCOL_VERSION,
        "metrics_protocol_version": metrics_protocol.PROTOCOL_VERSION,
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            env["device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return env


@dataclass
class RunManifest:
    """一次训练/评估运行的机器可读清单。"""

    run_id: str
    purpose: str                       # train / eval / ablations ...
    model_name: str
    git_sha: str = field(default_factory=git_sha)
    git_dirty: Optional[bool] = field(default_factory=is_dirty)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    seed: Optional[int] = None
    split_seed: Optional[int] = None
    split_mode: str = "patient"
    splits: Optional[list] = None
    hyperparams: Dict = field(default_factory=dict)
    early_stopping: Dict = field(default_factory=dict)
    n_params: Optional[int] = None
    n_params_incl_buffers: Optional[int] = None
    #: 评估协议 v2 的冻结哈希。定义一旦改动，hash 随之改变，
    #: 于是"这份结果用的是哪版协议"是可核对的，而不是靠记忆。
    protocol_hash: str = field(default_factory=lambda: eval_protocol.PROTOCOL_HASH)
    checkpoint: Optional[dict] = None       # fingerprint(...)
    data: List[dict] = field(default_factory=list)   # [fingerprint(...), ...]
    metrics_file: Optional[str] = None
    env: Dict = field(default_factory=environment)
    notes: str = ""

    def validate(self) -> None:
        """清单必须足以回答「这个数字是哪次运行、用哪份权重、哪个 split 产生的」。"""
        problems = []
        if not self.run_id:
            problems.append("缺少 run_id")
        if not self.model_name:
            problems.append("缺少 model_name")
        if self.git_sha in (None, "", "unknown"):
            problems.append("缺少 git SHA")
        if self.seed is None:
            problems.append("缺少训练 seed")
        if self.split_seed is None:
            problems.append("缺少划分 seed")
        if not self.splits:
            problems.append("缺少 split 比例")
        if self.checkpoint is None:
            problems.append("缺少 checkpoint 指纹")
        if not self.data:
            problems.append("缺少数据指纹")
        if problems:
            raise ValueError("运行清单不完整：\n- " + "\n- ".join(problems))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)

    def write(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


def load_manifest(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def join_metrics(manifests: Dict[str, dict], metrics: Dict[str, dict]) -> List[dict]:
    """把指标行与运行清单按 ``run_id`` 关联。

    跨运行汇总时最常见的错误是拿到别的 run / seed / split 的数字。
    这里对每个指标行强制要求 ``run_id`` 存在且能在清单里找到，
    否则直接抛错而不是静默拼接。
    """
    rows = []
    for metric_key, row in metrics.items():
        rid = row.get("run_id")
        if rid is None:
            raise ValueError(f"指标 {metric_key} 缺少 run_id，拒绝跨运行汇总")
        if rid not in manifests:
            raise ValueError(f"指标 {metric_key} 引用了未知的 run_id：{rid}")
        m = manifests[rid]
        for field_name in ("seed", "split"):
            if field_name in row and field_name in m and row[field_name] != m[field_name]:
                raise ValueError(
                    f"指标 {metric_key} 的 {field_name}={row[field_name]} "
                    f"与清单 {rid} 的 {m[field_name]} 不一致"
                )
        rows.append({**row, "run_id": rid, "model_name": m.get("model_name"),
                     "git_sha": m.get("git_sha")})
    return rows
