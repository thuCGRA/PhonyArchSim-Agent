"""光子 DSE 框架的器件 / 电路级工具箱。

本工具箱支撑*光学层*代理，该代理诊断层级别以下的光链路：光子器件变体、其
可配置的 ``cfgs`` 参数，以及光学结构（交叉开关几何、波长数、精度、数据流）。
它与 :class:`~photon_tuner.toolbox.analysis_tools.AnalysisToolbox` 明确分离，
使每个代理的工具面清晰可见。

工具按类别一文件一组（catalog / metrics / edits）；:class:`DeviceToolbox` 通过
``name -> (spec, handler)`` 注册表将它们组装起来。

使用者：optical_level、solution。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from photon_tuner.common.tool_types import ToolResult, ToolSpec
from photon_tuner.core.search_space import (
    discover_device_catalog,
    discover_device_variants,
)
from photon_tuner.toolbox.common_tool.context import ToolContext
from photon_tuner.toolbox.tool_cache import ToolResultCache

from . import catalog, edits, frontier, metrics

_REGISTRY: Dict[str, Tuple[ToolSpec, Any]] = {}
for _mod in (catalog, metrics, edits, frontier):
    _REGISTRY.update(_mod.TOOLS)

# 只有链路编辑工具会修改状态；其余都是纯报告。
_MUTATING_TOOLS = {"modify_device_parameter"}


class DeviceToolbox:
    """器件级读写工具箱。"""

    # 实例级器件类型切换（非库类型）。选择其一会重写 ``build_sub_arch`` 中
    # 相应的网表实例。
    INSTANCE_CHOICES: Dict[str, List[str]] = {
        "laser_source": ["on_chip", "off_chip"],
        "splitter_kind": ["laser_splitter", "mmi"],
    }

    def __init__(self, devicelib_root: str, search_space: Optional[Dict[str, Any]] = None) -> None:
        self.devicelib_root = str(devicelib_root)
        self.search_space = search_space or {}
        self.catalog = discover_device_catalog(self.devicelib_root)
        self.variants = discover_device_variants(self.devicelib_root)
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
            raise ValueError(f"Unknown device tool: {name}")
        if name not in _MUTATING_TOOLS:
            key = self._cache.key(name, context.current_hardware, arguments)
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            result = entry[1](self, context, arguments)
            self._cache.put(key, result)
            return result
        return entry[1](self, context, arguments)
