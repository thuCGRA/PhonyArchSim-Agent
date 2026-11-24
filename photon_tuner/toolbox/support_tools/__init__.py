"""跨领域支撑工具箱：知识、记忆、搜索状态、调度与仿真。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from photon_tuner.common.tool_types import ToolResult, ToolSpec
from photon_tuner.toolbox.common_tool.context import ToolContext

from . import knowledge, state_memory, workflow


_REGISTRY: Dict[str, Tuple[ToolSpec, Any]] = {}
for _module in (knowledge, state_memory, workflow):
    _REGISTRY.update(_module.TOOLS)


class SupportToolbox:
    """不复制领域实现，只把已存在的项目能力接入统一工具协议。"""

    def __init__(self, devicelib_root: str) -> None:
        self.devicelib_root = str(devicelib_root)

    def specs(self) -> List[ToolSpec]:
        return [spec for spec, _ in _REGISTRY.values()]

    def run(
        self,
        name: str,
        context: ToolContext,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        entry = _REGISTRY.get(name)
        if entry is None:
            raise ValueError(f"Unknown support tool: {name}")
        return entry[1](self, context, dict(arguments or {}))


__all__ = ["SupportToolbox"]
