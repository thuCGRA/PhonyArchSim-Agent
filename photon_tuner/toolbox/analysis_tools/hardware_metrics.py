"""Hardware-config and scalar-metric read tools for the analysis toolbox.

Used by: model_level, layer_level, solution.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from photon_tuner.common.tool_types import ToolResult, ToolSpec
from photon_tuner.core.search_space import hardware_summary, materialize_hardware
from photon_tuner.toolbox.common_tool._report_utils import (
    _active_evaluation,
    _load_report,
    _metrics_payload,
)
from photon_tuner.toolbox.common_tool.context import ToolContext


def summarize_hardware_config(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    hardware = materialize_hardware(context.current_hardware)
    return ToolResult("summarize_hardware_config", {"hardware": hardware_summary(hardware)})


def summarize_metrics(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    evaluation = _active_evaluation(context)
    return ToolResult("summarize_metrics", {"metrics": _metrics_payload(context, evaluation)})


def summarize_energy_breakdown(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    report = _load_report(context)
    energy = report.get("energy", {})
    return ToolResult("summarize_energy_breakdown", {
        "by_device": energy.get("by_device", {}),
        "by_chip_type": energy.get("by_chip_type", {}),
        "total": energy.get("total", {}),
    })


def summarize_area_breakdown(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    report = _load_report(context)
    area = report.get("area", {})
    return ToolResult("summarize_area_breakdown", {
        "by_device": area.get("by_device", {}),
        "by_chip_type": area.get("by_chip_type", {}),
        "total": area.get("total", {}),
    })


def summarize_insertion_loss(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    report = _load_report(context)
    return ToolResult("summarize_insertion_loss", {"insertion_loss": report.get("insertion_loss", {})})


TOOLS: Dict[str, Tuple[ToolSpec, Callable]] = {
    "summarize_hardware_config": (
        ToolSpec("summarize_hardware_config", "Compact summary of the current photonic architecture and device choices.", {"type": "object", "properties": {}}),
        summarize_hardware_config,
    ),
    "summarize_metrics": (
        ToolSpec("summarize_metrics", "Scalar latency / energy / area metrics and the combined objective.", {"type": "object", "properties": {}}),
        summarize_metrics,
    ),
    "summarize_energy_breakdown": (
        ToolSpec("summarize_energy_breakdown", "Per-device and per-chip-type energy breakdown (pJ).", {"type": "object", "properties": {}}),
        summarize_energy_breakdown,
    ),
    "summarize_area_breakdown": (
        ToolSpec("summarize_area_breakdown", "Per-device and per-chip-type area breakdown (um^2).", {"type": "object", "properties": {}}),
        summarize_area_breakdown,
    ),
    "summarize_insertion_loss": (
        ToolSpec("summarize_insertion_loss", "Node and core longest optical path insertion loss (dB).", {"type": "object", "properties": {}}),
        summarize_insertion_loss,
    ),
}
