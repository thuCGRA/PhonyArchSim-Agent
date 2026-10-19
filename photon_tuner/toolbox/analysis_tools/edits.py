"""Edit tools (solution-generation only) for the analysis toolbox.

Used by: solution.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from photon_tuner.common.tool_types import ToolResult, ToolSpec
from photon_tuner.core.search_space import (
    hardware_summary,
    modify_arch_parameter,
    step_arch_parameter,
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


TOOLS: Dict[str, Tuple[ToolSpec, Callable]] = {
    "modify_hardware_parameter": (
        ToolSpec("modify_hardware_parameter", "Apply explicit optical-structure parameter updates.", {"type": "object", "properties": {"updates": {"type": "array"}}}),
        modify_hardware_parameter,
    ),
    "step_hardware_parameter": (
        ToolSpec("step_hardware_parameter", "Step one optical-structure parameter to the next candidate.", {"type": "object", "properties": {"parameter": {"type": "string"}, "direction": {"type": "integer"}}}),
        step_hardware_parameter,
    ),
}
