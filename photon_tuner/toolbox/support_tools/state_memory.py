"""搜索状态、Pareto档案与案例记忆工具。"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from photon_tuner.common.tool_types import ToolResult, ToolSpec
from photon_tuner.rag.vector_store import case_outcome, hardware_fingerprint
from photon_tuner.toolbox.common_tool.context import ToolContext


def _require(value: Any, label: str) -> Any:
    if value is None:
        raise ValueError(f"{label} is not bound to ToolContext")
    return value


def summarize_design_identity(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    hardware = context.current_hardware or {}
    fingerprint = hardware_fingerprint(hardware)
    return ToolResult("summarize_design_identity", {
        "fingerprint": fingerprint,
        "seen": fingerprint in context.forbidden_fingerprints,
        "known_fingerprint_count": len(context.forbidden_fingerprints),
    })


def retrieve_similar_cases(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    store = _require(context.case_store, "case_store")
    query = str(arguments.get("query") or "").strip()
    if not query:
        raise ValueError("retrieve_similar_cases requires a non-empty query")
    top_k = max(1, int(arguments.get("top_k", 5)))
    min_score = float(arguments.get("min_score", 0.0))
    cases = store.search(
        query,
        context.current_hardware or {},
        top_k=top_k,
        min_score=min_score,
    )
    summaries = []
    for case in cases:
        record = case.record
        summaries.append({
            "score": round(float(case.score), 4),
            "record_id": record.record_id,
            "outcome": case_outcome(record.metrics, store.improved_delta_pct),
            "solution": record.solution,
            "metrics": record.metrics,
            "metadata": record.metadata,
        })
    return ToolResult("retrieve_similar_cases", {"cases": summaries, "query": query})


def summarize_case_memory(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    store = _require(context.case_store, "case_store")
    return ToolResult("summarize_case_memory", {
        "case_count": len(store),
        "improved_case_count": store.improved_case_count(),
        "improvement_threshold_pct": store.improved_delta_pct,
    })


def commit_verified_case(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    store = _require(context.case_store, "case_store")
    required = ("bottleneck_description", "hardware", "solution")
    missing = [name for name in required if not arguments.get(name)]
    if missing:
        raise ValueError(f"commit_verified_case missing required fields: {missing}")
    record = store.add(
        bottleneck_description=str(arguments["bottleneck_description"]),
        hardware=dict(arguments["hardware"]),
        solution=dict(arguments["solution"]),
        metrics=dict(arguments.get("metrics") or {}),
        metadata=dict(arguments.get("metadata") or {}),
        bottleneck_state=dict(arguments.get("bottleneck_state") or {}),
    )
    return ToolResult("commit_verified_case", {"record_id": record.record_id})


def summarize_pareto_frontier(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    archive = _require(context.pareto_archive, "pareto_archive")
    tag_key = arguments.get("tag_key")
    tag_value = arguments.get("tag_value")
    points = archive.frontier(tag_key, tag_value) if tag_key else archive.frontier()
    return ToolResult("summarize_pareto_frontier", {
        "minimize_keys": list(archive.minimize_keys),
        "point_count": len(points),
        "points": points,
    })


TOOLS: Dict[str, Tuple[ToolSpec, Callable]] = {
    "summarize_design_identity": (
        ToolSpec("summarize_design_identity", "Compute the canonical hardware fingerprint and report whether it has already been evaluated.", {"type": "object", "properties": {}}),
        summarize_design_identity,
    ),
    "retrieve_similar_cases": (
        ToolSpec("retrieve_similar_cases", "Retrieve top-k prior cases conditioned on bottleneck text and current hardware, including verified outcome labels.", {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}, "min_score": {"type": "number"}}}),
        retrieve_similar_cases,
    ),
    "summarize_case_memory": (
        ToolSpec("summarize_case_memory", "Report total and objectively improved cases in the reusable case memory.", {"type": "object", "properties": {}}),
        summarize_case_memory,
    ),
    "commit_verified_case": (
        ToolSpec("commit_verified_case", "HOST ONLY: persist one simulator-verified case and its before/after outcome.", {"type": "object", "properties": {"bottleneck_description": {"type": "string"}, "hardware": {"type": "object"}, "solution": {"type": "object"}, "metrics": {"type": "object"}, "metadata": {"type": "object"}, "bottleneck_state": {"type": "object"}}}),
        commit_verified_case,
    ),
    "summarize_pareto_frontier": (
        ToolSpec("summarize_pareto_frontier", "Return the evaluated non-dominated frontier and its physical minimize axes.", {"type": "object", "properties": {"tag_key": {"type": "string"}, "tag_value": {"type": "string"}}}),
        summarize_pareto_frontier,
    ),
}
