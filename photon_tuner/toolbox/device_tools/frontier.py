"""器件工具箱的器件前沿读取工具。"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from photon_tuner.common.tool_types import ToolResult, ToolSpec
from photon_tuner.core.device_frontier import summarize_device_frontier as _summarize
from photon_tuner.toolbox.common_tool.context import ToolContext


def summarize_device_frontier(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    device_type = arguments.get("device_type")
    hardware = getattr(context, "current_hardware", None)
    payload = _summarize(toolbox.catalog, hardware, device_type)
    return ToolResult("summarize_device_frontier", payload)


TOOLS: Dict[str, Tuple[ToolSpec, Callable]] = {
    "summarize_device_frontier": (
        ToolSpec(
            "summarize_device_frontier",
            "Summarize the Pareto frontier of real device variants per device type. "
            "Reports, for each type: compatible variants and their reported speed_GHz/"
            "prec_bit/energy_cost+energy_unit/area_um2, the non-dominated energy/area "
            "frontier, dominated variants, and variants excluded by the current "
            "work_freq/precision coupling. The energy metric is type-specific: mzm uses "
            "efficiency (fJ/bit) and phase_shifter uses P_pi (mW), because their static "
            "power is 0; other types use power/dynamic_power (mW). Real reported values "
            "only — no interpolation. Pass device_type to restrict to one category "
            "(e.g. dac); omit to list all types.",
            {"type": "object", "properties": {"device_type": {"type": "string"}}},
        ),
        summarize_device_frontier,
    ),
}
