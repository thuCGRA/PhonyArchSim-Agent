"""Device-catalog read tools for the device toolbox.

Used by: optical_level.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from photon_tuner.common.tool_types import ToolResult, ToolSpec
from photon_tuner.toolbox.common_tool.context import ToolContext


def list_device_candidates(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    device_type = arguments.get("device_type")
    if device_type:
        choices = toolbox.INSTANCE_CHOICES.get(device_type)
        if choices is not None:
            return ToolResult("list_device_candidates", {device_type: choices})
        return ToolResult("list_device_candidates", {device_type: toolbox.variants.get(device_type, [])})
    return ToolResult("list_device_candidates", {
        "variants": toolbox.variants,
        "instance_choices": dict(toolbox.INSTANCE_CHOICES),
    })


def inspect_device_config(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    device_type, device_name = _resolve_device_ref(toolbox, arguments)
    body = toolbox.catalog.get(device_type, {}).get(device_name)
    if body is None:
        raise ValueError(f"Device {device_name!r} not found for type {device_type!r}")
    return ToolResult("inspect_device_config", {
        "device_type": device_type,
        "device_name": device_name,
        "cfgs": body.get("cfgs", {}),
    })


def _resolve_device_ref(toolbox: Any, arguments: Dict[str, Any]) -> tuple:
    """Tolerate an LLM that passes a device *name* where the *type* belongs."""
    device_type = str(arguments.get("device_type", "") or "")
    device_name = str(arguments.get("device_name", "") or "")
    if device_type in toolbox.catalog:
        return device_type, device_name
    # ``device_type`` is not a category; if it is actually a known variant,
    # re-map it to its category and treat it as the name.
    try:
        resolved_type = _device_type_of(toolbox, device_type)
    except ValueError:
        resolved_type = ""
    if resolved_type:
        return resolved_type, device_type
    if device_name:
        try:
            return _device_type_of(toolbox, device_name), device_name
        except ValueError:
            pass
    return device_type, device_name


def _device_type_of(toolbox: Any, device_name: str) -> str:
    for device_type, variants in toolbox.variants.items():
        if device_name in variants:
            return device_type
    raise ValueError(f"Device {device_name!r} not found in the device library")


TOOLS: Dict[str, Tuple[ToolSpec, Callable]] = {
    "list_device_candidates": (
        ToolSpec("list_device_candidates", "List available device variants per device type. device_type is a category (e.g. mzm, photodetector); omit it to list all types.", {"type": "object", "properties": {"device_type": {"type": "string"}}}),
        list_device_candidates,
    ),
    "inspect_device_config": (
        ToolSpec("inspect_device_config", "Show the configurable cfgs of one device variant. device_type is the category (e.g. mzm); device_name is a variant from list_device_candidates (e.g. LT_MZM).", {"type": "object", "properties": {"device_type": {"type": "string"}, "device_name": {"type": "string"}}}),
        inspect_device_config,
    ),
}
