"""测试辅助：读取"只有单个子架构"的报告分节。

报告里的子架构键历史上是 ``"LT"``（SimPhony 的静态层映射名），现在是实际拓扑名
（``core_kind``，例如 ``"dptc_crossbar"`` / ``"mzi_mesh_clements"``）。测试不该
写死任何一个名字，否则改一次命名就要改遍全部断言。
"""

from __future__ import annotations

from typing import Any, Dict


def sub(mapping: Any) -> str:
    """返回该分节里唯一的子架构键。"""
    if not isinstance(mapping, dict) or not mapping:
        raise AssertionError(f"expected a non-empty sub-architecture mapping, got {mapping!r}")
    return next(iter(mapping))


def sub_arch(mapping: Any) -> Dict[str, Any]:
    """返回该分节里唯一子架构的内容。"""
    return mapping[sub(mapping)]
