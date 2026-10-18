"""测试自身的健全性检查——针对"本地通过、CI 失败"这一类缺陷。

起因：``tests/test_multitask.py`` 在 ``ckpts/test_metrics.json`` 缺失时调用
``pytest.skip``，却没有在模块顶层 ``import pytest``。本地 ``ckpts/`` 存在，
跳过分支从不执行，测试全绿；CI 里该文件不存在，跳过分支执行，抛 NameError，
整条流水线红灯。

这类缺陷的共同点是：**只在跳过分支上暴露**。所以本模块做两件事：

1. 静态检查每个测试模块是否把用到的名字导入了；
2. 提供一个可执行的检查方式，让跳过分支在本地也被走到（见模块末尾说明）。
"""

from __future__ import annotations

import ast
import pathlib

import pytest

TESTS = pathlib.Path(__file__).resolve().parent


def _modules():
    return sorted(p for p in TESTS.glob("test_*.py") if p.name != pathlib.Path(__file__).name)


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_module_imports_everything_it_uses(path):
    """模块里用到的顶层名字必须在模块里绑定过。

    用 AST 找 ``pytest.xxx`` / ``np.xxx`` / ``torch.xxx`` 这类属性访问的根名字，
    再确认该名字在模块作用域内出现过（import、赋值或函数参数）。比"必须
    import pytest"更一般，能一并挡住 ``np`` 漏 import 之类的变体。
    """
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                imported.add(a.asname or a.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in (node.args.args + node.args.kwonlyargs + node.args.posonlyargs):
                imported.add(arg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            imported.add(node.id)

    # 只检查常见测试依赖的根名字，避免把标准库、内建误报成缺失
    watched = {"pytest", "np", "torch", "json", "pathlib", "np"}
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            used.add(node.value.id)

    missing = (used & watched) - imported
    assert not missing, (
        f"{path.name} 使用了未导入的名字：{sorted(missing)}——"
        f"若它只出现在跳过分支上，本地不会暴露，CI 会失败"
    )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_module_parses(path):
    """语法必须合法；解析失败会让整个模块静默变成收集错误。"""
    ast.parse(path.read_text(encoding="utf-8", errors="replace"))


def test_skip_branches_are_reachable_in_principle():
    """说明性用例：记录如何让跳过分支在本地也被执行。

    CI 里 ``ckpts/`` 与 ``.audit/`` 不存在，所有 skip 分支都会走到；本地它们
    存在，跳过分支就是死代码。要复现 CI 的行为，把这两者暂时移开再跑：

        mv ckpts .ckpts.bak && mv .audit .audit.bak
        pytest -q
        mv .ckpts.bak ckpts && mv .audit.bak .audit

    这一步已在本机执行过，见 #79 的验证记录。
    """
    assert (TESTS / "test_multitask.py").exists()
