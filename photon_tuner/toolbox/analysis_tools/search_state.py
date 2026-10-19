"""Search-state tools (compare / select evaluation bases) for the analysis toolbox.

Used by: model_level, layer_level, solution.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from photon_tuner.common.tool_types import ToolResult, ToolSpec
from photon_tuner.core.search_space import hardware_summary
from photon_tuner.toolbox.common_tool._report_utils import _metrics_payload
from photon_tuner.toolbox.common_tool.context import ToolContext


def compare_search_states(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    states = {}
    for key, evaluation in context.evaluation_bases.items():
        states[key] = {
            "metrics": _metrics_payload(context, evaluation),
            "hardware": hardware_summary(evaluation.hardware),
        }
    return ToolResult("compare_search_states", {"states": states})


def select_analysis_base(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    source = str(arguments.get("source", "current"))
    if source not in context.evaluation_bases:
        raise ValueError(f"Unknown analysis base: {source}")
    context.active_base = {"source": source}
    return ToolResult("select_analysis_base", {"active_base": source})


TOOLS: Dict[str, Tuple[ToolSpec, Callable]] = {
    "compare_search_states": (
        ToolSpec("compare_search_states", "Compare current / previous / best evaluated design points.", {"type": "object", "properties": {}}),
        compare_search_states,
    ),
    "select_analysis_base": (
        ToolSpec("select_analysis_base", "Select which evaluated design point downstream tools use.", {"type": "object", "properties": {"source": {"type": "string"}}}),
        select_analysis_base,
    ),
}
