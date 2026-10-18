#!/usr/bin/env python
"""Build a reproducibility manifest for a release.

只读取 git 元数据、已安装依赖版本和显式传入的 artifact；不读临床数据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import eval_protocol as P  # noqa: E402

PACKAGES = (
    "torch", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib",
    "openpyxl", "tqdm", "requests", "pytest", "ruff",
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def file_record(rel: str) -> dict[str, object]:
    path = ROOT / rel
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"path": rel, "sha256": digest.hexdigest(), "bytes": size}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--ref", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("artifacts", nargs="*")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        print(
            "::error::tracked files are modified; refusing to build a release manifest",
            file=sys.stderr,
        )
        print(dirty, file=sys.stderr)
        return 1

    manifest = {
        "schema_version": 1,
        "tag": args.tag or None,
        "ref": args.ref or None,
        "dry_run": args.dry_run,
        "publish": not args.dry_run and bool(args.tag),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "commit_timestamp": git("show", "-s", "--format=%cI", "HEAD"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "dependencies": package_versions(),
        "protocol": {"version": P.PROTOCOL_VERSION, "hash": P.PROTOCOL_HASH},
        "worktree_clean": not dirty,
        "artifacts": [file_record(rel) for rel in args.artifacts],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
