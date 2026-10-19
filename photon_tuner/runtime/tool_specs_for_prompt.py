"""Tool-spec and tool-plan helpers for the ReAct tool loop."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

from photon_tuner.common.tool_types import ToolResult, ToolSpec


def tool_specs_for_prompt(
    toolbox: Any,
    allowed_tool_names: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Return tool specs filtered to the agent's allow-list."""
    allowed = set(allowed_tool_names) if allowed_tool_names is not None else None
    specs = []
    for spec in toolbox.specs():
        name = spec.name if isinstance(spec, ToolSpec) else str(spec.get("name", ""))
        if allowed is not None and name not in allowed:
            continue
        specs.append(asdict(spec) if isinstance(spec, ToolSpec) else spec)
    return specs


def execute_tool_plan(
    toolbox: Any,
    tool_plan: Dict[str, Any],
    context: Any,
) -> Dict[str, ToolResult]:
    """Execute a plan of tool calls (used by the solution agent)."""
    tools = tool_plan.get("tools")
    if not isinstance(tools, list):
        raise ValueError("Tool plan must contain a 'tools' list.")
    results: Dict[str, ToolResult] = {}
    for item in tools:
        name = item.get("name")
        arguments = item.get("arguments", {})
        if not isinstance(name, str):
            raise ValueError(f"Tool plan item missing valid name: {item!r}")
        results[name] = toolbox.run(name, context, arguments=arguments)
    return results
