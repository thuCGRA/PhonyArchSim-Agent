"""调度信号与仿真适配器工具。"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from photon_tuner.common.tool_types import ToolResult, ToolSpec
from photon_tuner.toolbox.common_tool.context import ToolContext


def _require(value: Any, label: str) -> Any:
    if value is None:
        raise ValueError(f"{label} is not bound to ToolContext")
    return value


def summarize_schedule_signals(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    signals = _require(context.schedule_signals, "schedule_signals")
    cfg = _require(context.schedule_config, "schedule_config")
    payload = signals.as_dict(cfg)
    payload["convergence_window"] = int(cfg.convergence_window)
    payload["convergence_delta"] = float(cfg.convergence_delta)
    payload["warm_precedent_target"] = int(cfg.warm_precedent_target)
    return ToolResult("summarize_schedule_signals", payload)


def preview_evaluation_schema(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    adapter = _require(context.simulator_adapter, "simulator_adapter")
    return ToolResult("preview_evaluation_schema", {"schema": adapter.schema()})


def evaluate_design(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    adapter = _require(context.simulator_adapter, "simulator_adapter")
    hardware = dict(arguments.get("hardware") or context.current_hardware or {})
    if not hardware:
        raise ValueError("evaluate_design requires hardware")
    iteration = int(arguments.get("iteration", context.iteration))
    result = adapter.evaluate(hardware, iteration)
    return ToolResult("evaluate_design", {"evaluation": result.to_dict()})


TOOLS: Dict[str, Tuple[ToolSpec, Callable]] = {
    "summarize_schedule_signals": (
        ToolSpec("summarize_schedule_signals", "Read maturity and objective-improvement signals used by the deterministic workflow scheduler.", {"type": "object", "properties": {}}),
        summarize_schedule_signals,
    ),
    "preview_evaluation_schema": (
        ToolSpec("preview_evaluation_schema", "Describe the simulator input, output and search-space contract without running an evaluation.", {"type": "object", "properties": {}}),
        preview_evaluation_schema,
    ),
    "evaluate_design": (
        ToolSpec("evaluate_design", "HOST ONLY: run one expensive simulator evaluation for a complete hardware design point.", {"type": "object", "properties": {"hardware": {"type": "object"}, "iteration": {"type": "integer"}}}),
        evaluate_design,
    ),
}
