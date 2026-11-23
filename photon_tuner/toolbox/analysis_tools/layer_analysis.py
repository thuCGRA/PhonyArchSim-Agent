"""分析工具箱的层排名 / 算子组 / 层详情读取工具。

使用者：bottleneck（全局瓶颈）、architecture、candidate。
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
    _energy_split_mj,
    _memory_energy_by_level,
    _rank_layers,
)
from photon_tuner.toolbox.common_tool.context import ToolContext


def summarize_layer_ranks(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    report = _load_report(context)
    ranks = _rank_layers(report, context)
    top = int(arguments.get("top_layers", len(ranks)))
    payload: Dict[str, Any] = {"layers": ranks[:top]}
    # 逐层能耗**只覆盖计算**；存储能耗按层级给出。两者合计才是目标函数里
    # 的 energy。不说明这一点，智能体会把"逐层能耗之和"当成全部能耗。
    split = _energy_split_mj(report)
    if split["total"] > 0:
        payload["energy_scope"] = {
            "note": "per-layer energy covers compute devices only; storage energy is "
                    "reported per memory level below. The objective uses compute+memory.",
            "compute_mj": split["compute"],
            "memory_mj": split["memory"],
            "total_mj": split["total"],
            "memory_share": split["memory"] / split["total"],
            "memory_by_level_mj": _memory_energy_by_level(report),
        }
    return ToolResult("summarize_layer_ranks", payload)


def summarize_operator_groups(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    """将逐层证据聚合为算子组。

    agent 看到的不是每个层一行，而是少数几个组（attention、qkv / output
    projection、ffn 等），以及它们在总能耗和周期中的占比。
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
            "energy_mj": 0.0,
            "area_share_mm2": 0.0,
            "memory_latency_ms": 0.0,
            "energy_source": item["energy_source"],
            "top_layers": [],
        })
        bucket["layer_count"] += 1
        bucket["switching_cycles"] += item["switching_cycles"]
        bucket["energy_mj"] += item["energy_mj"]
        bucket["area_share_mm2"] += item["area_share_mm2"]
        bucket["memory_latency_ms"] += item["memory_latency_ms"]
        if len(bucket["top_layers"]) < top_per_group:
            bucket["top_layers"].append(item["layer_name"])

    total_energy = sum(g["energy_mj"] for g in groups.values()) or 1.0
    total_cycles = sum(g["switching_cycles"] for g in groups.values()) or 1.0
    ordered = sorted(groups.values(), key=lambda g: g["energy_mj"], reverse=True)
    for bucket in ordered:
        bucket["energy_share"] = bucket["energy_mj"] / total_energy
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
