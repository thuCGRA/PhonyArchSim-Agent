"""Layer-rank / operator-group / layer-detail read tools for the analysis toolbox.

Used by: layer_level, model_level, solution.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from photon_tuner.common.tool_types import ToolResult, ToolSpec
from photon_tuner.common.utils import layer_group
from photon_tuner.toolbox.common_tool._report_utils import (
    _flatten_cycles,
    _flatten_memory,
    _flatten_memory_energy,
    _load_report,
    _rank_layers,
)
from photon_tuner.toolbox.common_tool.context import ToolContext


def summarize_layer_ranks(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    report = _load_report(context)
    ranks = _rank_layers(report, context)
    top = int(arguments.get("top_layers", len(ranks)))
    return ToolResult("summarize_layer_ranks", {"layers": ranks[:top]})


def summarize_operator_groups(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    """Aggregate the per-layer evidence into operator groups.

    Instead of one row per layer the agent sees a handful of groups (attention,
    qkv / output projection, ffn, ...) with their share of total energy and
    cycles.
    """
    report = _load_report(context)
    ranks = _rank_layers(report, context)
    top_per_group = int(arguments.get("top_layers_per_group", 3))

    groups: Dict[str, Dict[str, Any]] = {}
    for item in ranks:
        bucket = groups.setdefault(item["group"], {
            "group": item["group"],
            "layer_count": 0,
            "switching_cycles": 0.0,
            "energy_pj": 0.0,
            "area_share_um2": 0.0,
            "energy_source": item["energy_source"],
            "top_layers": [],
        })
        bucket["layer_count"] += 1
        bucket["switching_cycles"] += item["switching_cycles"]
        bucket["energy_pj"] += item["energy_pj"]
        bucket["area_share_um2"] += item["area_share_um2"]
        if len(bucket["top_layers"]) < top_per_group:
            bucket["top_layers"].append(item["layer_name"])

    total_energy = sum(g["energy_pj"] for g in groups.values()) or 1.0
    total_cycles = sum(g["switching_cycles"] for g in groups.values()) or 1.0
    ordered = sorted(groups.values(), key=lambda g: g["energy_pj"], reverse=True)
    for bucket in ordered:
        bucket["energy_share"] = bucket["energy_pj"] / total_energy
        bucket["cycle_share"] = bucket["switching_cycles"] / total_cycles
    return ToolResult("summarize_operator_groups", {"groups": ordered})


def inspect_layer_details(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    report = _load_report(context)
    workload = report.get("workload", {})
    memory = report.get("memory", {})
    cycles = _flatten_cycles(workload.get("partition_cycles", {}))
    layer_names = arguments.get("layer_names") or list(cycles.keys())
    details = []
    for name in layer_names:
        if name not in cycles:
            continue
        details.append({
            "layer_name": name,
            "group": layer_group(name),
            "cycles": cycles[name],
            "memory_latency": _flatten_memory(memory.get("latency", {})).get(name),
            "memory_energy": _flatten_memory_energy(memory.get("energy", {})).get(name),
        })
    return ToolResult("inspect_layer_details", {"layers": details})


TOOLS: Dict[str, Tuple[ToolSpec, Callable]] = {
    "summarize_layer_ranks": (
        ToolSpec("summarize_layer_ranks", "Rank layers by simulated latency / energy / area cost.", {"type": "object", "properties": {"top_layers": {"type": "integer"}}}),
        summarize_layer_ranks,
    ),
    "summarize_operator_groups": (
        ToolSpec("summarize_operator_groups", "Aggregate cost by operator group (attention / qkv_projection / ffn / convolution / ...).", {"type": "object", "properties": {"top_layers_per_group": {"type": "integer"}}}),
        summarize_operator_groups,
    ),
    "inspect_layer_details": (
        ToolSpec("inspect_layer_details", "Detailed workload and memory evidence for selected layers.", {"type": "object", "properties": {"layer_names": {"type": "array"}}}),
        inspect_layer_details,
    ),
}
