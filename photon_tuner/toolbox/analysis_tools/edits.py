"""分析工具箱的编辑工具（仅 solution-generation 使用）。

使用者：solution。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from photon_tuner.common.tool_types import ToolResult, ToolSpec
from photon_tuner.core.search_space import (
    hardware_summary,
    modify_arch_parameter,
    step_arch_parameter,
    switch_topology_template,
)
from photon_tuner.toolbox.common_tool.context import ToolContext


def modify_hardware_parameter(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    updates = arguments.get("updates")
    if not isinstance(updates, list):
        raise ValueError("modify_hardware_parameter requires an 'updates' list")
    new_hardware = modify_arch_parameter(
        context.current_hardware, updates, context.search_space
    )
    context.current_hardware = new_hardware
    return ToolResult("modify_hardware_parameter", {"hardware": hardware_summary(new_hardware)})


def step_hardware_parameter(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    parameter = str(arguments.get("parameter", ""))
    direction = int(arguments.get("direction", -1))
    new_hardware = step_arch_parameter(
        context.current_hardware, parameter, direction, context.search_space
    )
    context.current_hardware = new_hardware
    return ToolResult("step_hardware_parameter", {"hardware": hardware_summary(new_hardware)})


def modify_topology_template(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    core_kind = str(arguments.get("core_kind", "")).strip()
    if not core_kind:
        raise ValueError("modify_topology_template requires a 'core_kind'")
    new_hardware = switch_topology_template(context.current_hardware, core_kind)
    context.current_hardware = new_hardware
    return ToolResult("modify_topology_template", {"hardware": hardware_summary(new_hardware)})


TOOLS: Dict[str, Tuple[ToolSpec, Callable]] = {
    "modify_hardware_parameter": (
        ToolSpec("modify_hardware_parameter", "Apply explicit optical-structure parameter updates.", {"type": "object", "properties": {"updates": {"type": "array"}}}),
        modify_hardware_parameter,
    ),
    "step_hardware_parameter": (
        ToolSpec("step_hardware_parameter", "Step one optical-structure parameter to the next candidate.", {"type": "object", "properties": {"parameter": {"type": "string"}, "direction": {"type": "integer"}}}),
        step_hardware_parameter,
    ),
    "modify_topology_template": (
        ToolSpec("modify_topology_template", "Switch the optical core to another topology template (core_kind), auto-aligning node_kind.", {"type": "object", "properties": {"core_kind": {"type": "string"}}}),
        modify_topology_template,
    ),
}
