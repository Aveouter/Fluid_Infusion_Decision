"""issue #3 / RED：CLI 契约 —— --help 可用，非法参数被拒绝。

用子进程调用，确保测的是真实的命令行入口而非被 import 的函数。
全程不读取 data/ 下任何文件。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main.py"

# 入口脚本以 UTF-8 输出（帮助文本含中文）；子进程解码必须同样指定，
# 否则 Windows 上会以本地代码页（cp1252）解码而抛 UnicodeDecodeError。
_OUT = dict(capture_output=True, text=True, encoding="utf-8", errors="replace")


def _run(args, timeout=180):
    return subprocess.run(
        [sys.executable, str(MAIN), *args],
        cwd=str(ROOT),
        timeout=timeout,
        **_OUT,
    )


@pytest.mark.parametrize("entry", ["main.py", "inference.py"])
def test_help_exits_zero(entry):
    """--help 必须成功退出并打印 usage。"""
    proc = subprocess.run(
        [sys.executable, str(ROOT / entry), "--help"],
        cwd=str(ROOT), timeout=180, **_OUT,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, f"{entry} --help 退出码 {proc.returncode}\n{out[-2000:]}"
    assert "usage" in out.lower(), f"{entry} --help 未输出 usage"


def test_invalid_model_is_rejected():
    """非法 --model 必须被拒绝（argparse choices），不能静默兜底。"""
    proc = _run(["--model", "definitely_not_a_real_model", "--help"])
    # --help 会先于校验打印，因此这里用一次真实的最小调用
    proc = _run(["--model", "definitely_not_a_real_model"])
    assert proc.returncode != 0, "非法模型名竟然被接受"
    combined = ((proc.stdout or "") + (proc.stderr or "")).lower()
    assert "invalid" in combined or "choose from" in combined, \
        f"未见预期的参数校验信息:\n{combined[-2000:]}"


def test_default_model_name_is_known():
    """默认模型名必须出现在 choices 中 —— 防止默认值与可用模型漂移。"""
    proc = _run(["--help"])
    out = (proc.stdout or "") + (proc.stderr or "")
    assert "tcn_film" in out, "帮助信息中未列出默认模型 tcn_film"
