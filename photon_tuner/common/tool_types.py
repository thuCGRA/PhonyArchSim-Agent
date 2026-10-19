"""Tool-related dataclasses, separated from the business schemas.

``ToolSpec`` describes a tool to the LLM (name + description + input schema);
``ToolResult`` is what a tool returns after execution.
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
