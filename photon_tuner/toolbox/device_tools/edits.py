"""Edit tools for the device toolbox.

Used by: solution.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from photon_tuner.common.tool_types import ToolResult, ToolSpec
from photon_tuner.core.search_space import (
    hardware_summary,
    materialize_hardware,
    set_device_choice,
)
from photon_tuner.toolbox.common_tool.context import ToolContext


def modify_device_parameter(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    level = arguments.get("level")
    device_type = str(arguments.get("device_type", ""))
    device_name = str(arguments.get("device_name", ""))
    if arguments.get("parameter") is not None:
        raise ValueError(
            "device cfgs overrides are removed; modify_device_parameter only "
            "selects a library variant via (level, device_type, device_name)"
        )
    if level not in {"core", "node"} or not device_type or not device_name:
        raise ValueError("modify_device_parameter requires (level, device_type, device_name)")
    choices = toolbox.INSTANCE_CHOICES.get(device_type)
    if choices is not None and str(device_name) not in choices:
        raise ValueError(
            f"instance choice {device_type} must be one of {choices}, got {device_name!r}"
        )

    hardware = materialize_hardware(context.current_hardware)
    new_hardware = set_device_choice(hardware, level, device_type, device_name)
    context.current_hardware = new_hardware
    return ToolResult("modify_device_parameter", {"hardware": hardware_summary(new_hardware)})


TOOLS: Dict[str, Tuple[ToolSpec, Callable]] = {
    "modify_device_parameter": (
        ToolSpec("modify_device_parameter", "Select a library variant for a device type at the core or node level.", {"type": "object", "properties": {"level": {"type": "string"}, "device_type": {"type": "string"}, "device_name": {"type": "string"}}}),
        modify_device_parameter,
    ),
}
