"""工具相关的数据类，与业务 schema 分离。

``ToolSpec`` 向 LLM 描述一个工具（名称 + 描述 + 输入 schema）；
``ToolResult`` 是工具执行后的返回值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ToolResult:
    name: str
    payload: Dict[str, Any]
    generated_files: Dict[str, str] = field(default_factory=dict)


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
