"""器件工具箱的器件目录读取工具。

使用者：optical_level。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from photon_tuner.common.tool_types import ToolResult, ToolSpec
from photon_tuner.core.provenance import resolve_variant
from photon_tuner.core.topology_space import coupling_compatible_variants
from photon_tuner.toolbox.common_tool.context import ToolContext


def _coupling_filtered(toolbox: Any, context: ToolContext):
    """按当前设计点的耦合关系过滤库变体。

    返回 ``{"compatible": ..., "excluded": ...}``；当没有绑定当前硬件时（无可
    过滤对象）返回 ``None``。
    """
    hardware = getattr(context, "current_hardware", None)
    if not hardware:
        return None
    return coupling_compatible_variants(toolbox.catalog, hardware)


def list_device_candidates(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    device_type = arguments.get("device_type")
    filt = _coupling_filtered(toolbox, context)
    compatible = filt["compatible"] if filt is not None else toolbox.variants
    excluded = filt["excluded"] if filt is not None else {}
    if device_type:
        choices = toolbox.INSTANCE_CHOICES.get(device_type)
        if choices is not None:
            return ToolResult("list_device_candidates", {device_type: choices})
        return ToolResult("list_device_candidates", {
            device_type: compatible.get(device_type, []),
            "excluded": {device_type: excluded.get(device_type, [])}
            if device_type in excluded else {},
        })
    return ToolResult("list_device_candidates", {
        "variants": compatible,
        "instance_choices": dict(toolbox.INSTANCE_CHOICES),
        "coupling_filtered": filt is not None,
        "excluded": excluded,
    })


def inspect_device_config(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    device_type, device_name = _resolve_device_ref(toolbox, arguments)
    body = toolbox.catalog.get(device_type, {}).get(device_name)
    if body is None:
        # 缺失/为空或未知的名称是可恢复的错误：返回该类型的合法候选，以便
        # 代理选择一个并重试，而不是抛异常（那会浪费该阶段）。
        variants = sorted(toolbox.catalog.get(device_type, {}))
        if not variants:
            raise ValueError(
                f"Unknown device_type {device_type!r}; known types: "
                f"{sorted(toolbox.catalog)}"
            )
        return ToolResult("inspect_device_config", {
            "device_type": device_type,
            "device_name": device_name or None,
            "requested": device_name or None,
            "error": f"Device {device_name!r} not found for type {device_type!r}",
            "candidates": variants,
            "hint": "Call again with one of `candidates` as device_name.",
        })
    provenance = resolve_variant(device_type, device_name, body)
    return ToolResult("inspect_device_config", {
        "device_type": device_type,
        "device_name": device_name,
        "cfgs": body.get("cfgs", {}),
        "ref": body.get("ref", ""),
        "platform": body.get("platform", ""),
        "provenance": provenance["class"],
        "source": provenance["source"],
        "note": provenance.get("note", ""),
    })


def _resolve_device_ref(toolbox: Any, arguments: Dict[str, Any]) -> tuple:
    """容忍 LLM 在应填*类型*的位置传入器件*名称*的情况。"""
    device_type = str(arguments.get("device_type", "") or "")
    device_name = str(arguments.get("device_name", "") or "")
    if device_type in toolbox.catalog:
        return device_type, device_name
    # ``device_type`` 不是类别；若它实际是已知变体，则将其重映射到所属类别
    # 并视作名称。
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
