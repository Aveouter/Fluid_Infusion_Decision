#!/usr/bin/env python
"""冻结/校验评估协议 v2（issue #81，Gate A）。

文档与代码不能漂移
------------------
``docs/eval_protocol.md`` 的叙事部分是手写的，但**常量区块由代码生成**：
定界标记之间的内容全部来自 ``utils/eval_protocol.py``。任何阈值或词表改动
都会同时改变生成区块与哈希，于是"文档写了什么"与"代码算什么"无法分叉。

用法：

    python freeze_eval_protocol.py            # 校验（CI 与测试用；不一致即非零退出）
    python freeze_eval_protocol.py --write    # 重新生成区块与 docs/eval_protocol.json

本脚本不读任何临床数据，产物可安全提交到公开仓库。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from utils import eval_protocol as P

# Windows 下 stdout 被重定向（管道/日志）时默认用 cp1252，中文输出会抛
# UnicodeEncodeError。本脚本的产物是给人读的，故强制 UTF-8；无法重配时静默跳过。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # 非标准流或已关闭
    pass

DOC = Path("docs/eval_protocol.md")
RECORD = Path("docs/eval_protocol.json")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--doc", default=str(DOC))
    p.add_argument("--record", default=str(RECORD))
    p.add_argument("--write", action="store_true",
                   help="重新生成区块与协议记录；缺省只做校验")
    return p.parse_args(argv)


def splice(text: str, block: str) -> str:
    """把 ``block`` 放进定界标记之间。标记缺失即报错，不做猜测。"""
    b, e = P.FROZEN_BLOCK_BEGIN, P.FROZEN_BLOCK_END
    if b not in text or e not in text:
        raise SystemExit(
            f"文档缺少定界标记：请在「冻结常量」一节放置\n  {b}\n  {e}"
        )
    head, rest = text.split(b, 1)
    _, tail = rest.split(e, 1)
    return f"{head}{b}\n\n{block}\n{e}{tail}"


def expected_record() -> dict:
    return {
        "protocol_version": P.PROTOCOL_VERSION,
        "protocol_hash": P.PROTOCOL_HASH,
        "frozen": P.freeze_payload(),
        "note": "本文件由 freeze_eval_protocol.py 生成；不含任何患者数据。"
                "冻结后修改任何定义必须递增 PROTOCOL_VERSION 并保留变更记录。",
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    doc = Path(args.doc)
    record = Path(args.record)

    if not doc.exists():
        raise SystemExit(f"协议文档不存在：{doc}（叙事部分需手写，本脚本只生成常量区块）")

    text = doc.read_text(encoding="utf-8")
    block = P.render_frozen_block()

    if args.write:
        doc.write_text(splice(text, block), encoding="utf-8")
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(expected_record(), ensure_ascii=False, indent=2),
                          encoding="utf-8")
        print(f"[write] {doc}  区块已更新")
        print(f"[write] {record}  protocol_hash={P.PROTOCOL_HASH}")
        return 0

    problems = []
    if splice(text, block) != text:
        problems.append(f"{doc} 的常量区块与 utils/eval_protocol.py 不一致")
    if not record.exists():
        problems.append(f"缺少协议记录 {record}（用 --write 生成）")
    else:
        got = json.loads(record.read_text(encoding="utf-8"))
        if got.get("protocol_hash") != P.PROTOCOL_HASH:
            problems.append(
                f"{record} 的哈希 {got.get('protocol_hash')} 与当前 "
                f"{P.PROTOCOL_HASH} 不一致")
        if got.get("protocol_version") != P.PROTOCOL_VERSION:
            problems.append(
                f"{record} 的版本 {got.get('protocol_version')} 与当前 "
                f"{P.PROTOCOL_VERSION} 不一致")

    if problems:
        print("[FAIL] 协议未冻结一致：", file=sys.stderr)
        for item in problems:
            print(f"  - {item}", file=sys.stderr)
        print("\n如确认是有意的定义变更：递增 PROTOCOL_VERSION，"
              "然后 python freeze_eval_protocol.py --write", file=sys.stderr)
        return 1

    print(f"[ok] 协议一致  version={P.PROTOCOL_VERSION}  hash={P.PROTOCOL_HASH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
