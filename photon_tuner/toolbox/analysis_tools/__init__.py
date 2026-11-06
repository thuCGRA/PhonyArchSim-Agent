"""光子 DSE agent 的架构 / 层级别工具箱。

工具箱是 agent 读取仿真器证据或编辑设计的*唯一*途径。每个 agent 都收到
一个显式的 ``allowed_tool_names`` 白名单（见 ``photon_tuner.agents``），因此
model-level、layer-level 和 solution-generation 角色只会看到与其相关的工具。

工具按类别一个文件一组（硬件指标 / 层分析 / 搜索状态 / 编辑）；
:class:`AnalysisToolbox` 通过 ``name -> (spec, handler)`` 注册表将它们组装起来。

使用者：model_level、layer_level、solution。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from photon_tuner.common.tool_types import ToolResult, ToolSpec
from photon_tuner.toolbox.common_tool.context import ToolContext
from photon_tuner.toolbox.tool_cache import ToolResultCache

from . import edits, hardware_metrics, layer_analysis, search_state

_REGISTRY: Dict[str, Tuple[ToolSpec, Any]] = {}
for _mod in (hardware_metrics, layer_analysis, search_state, edits):
    _REGISTRY.update(_mod.TOOLS)

# 有副作用的工具（设计编辑 + 基准状态选择）绝不被缓存。
_MUTATING_TOOLS = {
    "modify_hardware_parameter",
    "modify_topology_template",
    "step_hardware_parameter",
    "select_analysis_base",
}


class AnalysisToolbox:
    """架构 / 层级别的读写工具箱。"""

    def __init__(self, search_space: Optional[Dict[str, Any]] = None) -> None:
        self.search_space = search_space or {}
        self._cache = ToolResultCache()

    def specs(self) -> List[ToolSpec]:
        return [spec for spec, _ in _REGISTRY.values()]

    def run(
        self,
        name: str,
        context: ToolContext,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        arguments = dict(arguments or {})
        entry = _REGISTRY.get(name)
        if entry is None:
            raise ValueError(f"Unknown analysis tool: {name}")
        if name not in _MUTATING_TOOLS:
            key = self._cache.key(name, context.current_hardware, arguments)
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            result = entry[1](self, context, arguments)
            self._cache.put(key, result)
            return result
        return entry[1](self, context, arguments)
