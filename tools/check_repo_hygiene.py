#!/usr/bin/env python
"""Fail CI when tracked files look like clinical data, credentials, or artifacts.

本脚本只扫描 ``git ls-files`` 的当前树，不扫描历史。历史清理是另一件事。
退出码非零表示发现至少一个问题。
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_SUFFIXES = (
    ".pkl", ".joblib", ".csv", ".xlsx", ".xls", ".npy", ".npz",
    ".parquet", ".onnx", ".pt", ".pth", ".h5", ".ckpt",
    ".zip", ".tar", ".gz", ".pem", ".key",
)
FORBIDDEN_PREFIXES = ("data/", "ckpts/", "reports/", "preds/")
FORBIDDEN_GLOBS = (
    "ml_output/*", "ml_output_refined/*",
    "per_patient_outputs/*", "per_patient_outputs_pre/*",
)
FORBIDDEN_NAMES = {".env"}
ALLOWED_ENV_EXAMPLE = ".env.example"

# 真实患者编号是纯数字；这里只拦“字段名 + 6 位以上数字”的同现。
PATIENT_ID = re.compile(
    r"(?i)\b(?:patient[_\s-]?id|subject[_\s-]?id|mrn|pid)\b"
    r"[^\n]{0,40}?(?<![0-9A-Za-z])\d{6,}(?![0-9A-Za-z])"
)

SECRET_PATTERNS = {
    "aws-access-key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github-token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "github-pat": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    "slack-token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "google-api-key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "openai-key": re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
}


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [p for p in result.stdout.decode("utf-8", "replace").split("\0") if p]


def forbidden_reason(rel: str) -> str | None:
    name = Path(rel).name
    lower = rel.lower()
    if name in FORBIDDEN_NAMES:
        return "environment file"
    if name.startswith(".env.") and name != ALLOWED_ENV_EXAMPLE:
        return "environment file"
    if lower.endswith(FORBIDDEN_SUFFIXES):
        return "data/artifact suffix"
    if lower.startswith(FORBIDDEN_PREFIXES):
        return "private data directory"
    if any(fnmatch.fnmatch(lower, pattern) for pattern in FORBIDDEN_GLOBS):
        return "generated output directory"
    return None


def is_text(path: Path) -> bool:
    try:
        return b"\0" not in path.read_bytes()[:8192]
    except OSError:
        return False


def scan_file(rel: str) -> list[tuple[int, str]]:
    path = ROOT / rel
    if not is_text(path):
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    findings: list[tuple[int, str]] = []
    for lineno, line in enumerate(lines, 1):
        if PATIENT_ID.search(line):
            findings.append((lineno, "patient-id-like value in a tracked file"))
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(line):
                findings.append((lineno, f"possible secret: {label}"))
    return findings


def main() -> int:
    try:
        files = tracked_files()
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"::error::cannot list tracked files: {exc}")
        return 1

    failures = 0
    for rel in files:
        reason = forbidden_reason(rel)
        if reason:
            print(f"::error file={rel}::tracked {reason}; remove it or keep it outside git")
            failures += 1
            continue
        for lineno, message in scan_file(rel):
            print(f"::error file={rel},line={lineno}::{message}")
            failures += 1

    if failures:
        print(
            f"\n[FAIL] {failures} hygiene finding(s) in {len(files)} tracked files.",
            file=sys.stderr,
        )
        return 1
    print(f"[ok] repo hygiene: {len(files)} tracked files clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
