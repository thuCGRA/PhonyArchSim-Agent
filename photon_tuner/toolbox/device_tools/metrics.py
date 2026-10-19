"""Per-device summary read tools for the device toolbox.

Used by: optical_level.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from photon_tuner.common.tool_types import ToolResult, ToolSpec
from photon_tuner.core.search_space import hardware_summary, materialize_hardware
from photon_tuner.toolbox.common_tool._report_utils import (
    _flatten_sub_archs,
    _load_report,
    _summarize_device_records,
)
from photon_tuner.toolbox.common_tool.context import ToolContext


def summarize_device_energy_breakdown(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    report = _load_report(context)
    by_device = _flatten_sub_archs(report.get("energy", {}).get("by_device", {}))
    summary = _summarize_device_records(by_device, metric="total_energy")
    return ToolResult("summarize_device_energy_breakdown", {"devices": summary})


def summarize_device_area_breakdown(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    report = _load_report(context)
    by_device = _flatten_sub_archs(report.get("area", {}).get("by_device", {}))
    summary = _summarize_device_records(by_device, metric="total_area")
    return ToolResult("summarize_device_area_breakdown", {"devices": summary})


def summarize_optical_structure(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    hardware = materialize_hardware(context.current_hardware)
    summary = hardware_summary(hardware)
    return ToolResult("summarize_optical_structure", {"optical_structure": summary.get("arch", {})})


def summarize_insertion_loss_detail(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    report = _load_report(context)
    return ToolResult("summarize_insertion_loss_detail", {"insertion_loss": report.get("insertion_loss", {})})


TOOLS: Dict[str, Tuple[ToolSpec, Callable]] = {
    "summarize_device_energy_breakdown": (
        ToolSpec("summarize_device_energy_breakdown", "Per-device energy contribution (pJ) with counts.", {"type": "object", "properties": {}}),
        summarize_device_energy_breakdown,
    ),
    "summarize_device_area_breakdown": (
        ToolSpec("summarize_device_area_breakdown", "Per-device area contribution (um^2) with counts.", {"type": "object", "properties": {}}),
        summarize_device_area_breakdown,
    ),
    "summarize_optical_structure": (
        ToolSpec("summarize_optical_structure", "Crossbar geometry, wavelength count, clock, precision, dataflow.", {"type": "object", "properties": {}}),
        summarize_optical_structure,
    ),
    "summarize_insertion_loss_detail": (
        ToolSpec("summarize_insertion_loss_detail", "Node/core longest optical paths and their insertion loss.", {"type": "object", "properties": {}}),
        summarize_insertion_loss_detail,
    ),
}
