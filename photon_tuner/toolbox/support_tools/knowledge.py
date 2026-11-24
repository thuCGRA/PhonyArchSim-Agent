"""设计知识、单位、工作负载和器件来源工具。"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from photon_tuner.common.tool_types import ToolResult, ToolSpec
from photon_tuner.core.provenance import audit_device_provenance
from photon_tuner.core.search_space import build_design_space
from photon_tuner.core.units import describe_units
from photon_tuner.evidence.workload_features import workload_features
from photon_tuner.toolbox.common_tool._report_utils import _load_report
from photon_tuner.toolbox.common_tool.context import ToolContext


def summarize_search_space(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    design_space = context.design_space or build_design_space(context.search_space or {})
    return ToolResult("summarize_search_space", {"parameters": design_space.summary()})


def describe_metric_units(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    return ToolResult("describe_metric_units", {"units": describe_units()})


def summarize_workload_features(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    name = str(arguments.get("workload_name") or context.workload_name or "current")
    return ToolResult(
        "summarize_workload_features",
        {"features": workload_features(name, _load_report(context))},
    )


def summarize_device_provenance(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    report = audit_device_provenance(toolbox.devicelib_root)
    device_type = str(arguments.get("device_type") or "").strip()
    if device_type:
        prefix = f"{device_type}/"
        report = {
            "total": sum(1 for name in report["variants"] if name.startswith(prefix)),
            "by_type": {device_type: report["by_type"].get(device_type, {})},
            "variants": {
                name: body
                for name, body in report["variants"].items()
                if name.startswith(prefix)
            },
        }
    return ToolResult("summarize_device_provenance", report)


TOOLS: Dict[str, Tuple[ToolSpec, Callable]] = {
    "summarize_search_space": (
        ToolSpec("summarize_search_space", "List every layered design parameter, its candidates, layer and constraints.", {"type": "object", "properties": {}}),
        summarize_search_space,
    ),
    "describe_metric_units": (
        ToolSpec("describe_metric_units", "Authoritative units for simulator metrics and report fields.", {"type": "object", "properties": {}}),
        describe_metric_units,
    ),
    "summarize_workload_features": (
        ToolSpec("summarize_workload_features", "Deterministic workload fingerprint: layer count, operator groups, cycles and size.", {"type": "object", "properties": {"workload_name": {"type": "string"}}}),
        summarize_workload_features,
    ),
    "summarize_device_provenance": (
        ToolSpec("summarize_device_provenance", "Audit device variants as traceable, baseline or placeholder with source records.", {"type": "object", "properties": {"device_type": {"type": "string"}}}),
        summarize_device_provenance,
    ),
}
