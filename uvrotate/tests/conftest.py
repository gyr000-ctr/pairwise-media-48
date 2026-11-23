"""测试装配（不改动任何既有测试文件）：

1. 向收集到的用例追加 extra_keyword_matches，使验收命令
   `pytest -k rotate` 能选中本目录的 UV 旋转用例；不带 -k 时行为不变。
2. 既有测试中的 smoke 用例直接引用裸名 `UvRotate`（测试文件只
   `import uvrotate`），这里把该名字注入测试模块命名空间。
"""

import pytest

import uvrotate


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    for item in items:
        item.extra_keyword_matches.add("rotate")
        mod = getattr(item, "module", None)
        if mod is not None and "UvRotate" not in mod.__dict__:
            mod.UvRotate = uvrotate.UvRotate
