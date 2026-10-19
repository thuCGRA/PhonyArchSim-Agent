"""Result types and coercion helpers for the ReAct tool loop.

``ReactLoopResult`` carries one agent loop's output.  The ``as_*`` helpers
defensively coerce untrusted LLM fields into the shapes the dataclasses expect
(real LLMs do not always follow a JSON schema exactly), so one malformed field
never crashes the whole tuning loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from photon_tuner.common.tool_types import ToolResult


@dataclass
class ReactLoopResult:
    tool_results: Dict[str, ToolResult]
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)


def as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_str(value: Any) -> str:
    return str(value) if value is not None else ""


def as_str_list(value: Any) -> List[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def as_dict_list(value: Any) -> List[Dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def serialize_tool_results(results: Dict[str, ToolResult]) -> Dict[str, Dict[str, Any]]:
    return {name: {"payload": _compact_payload(result.payload)} for name, result in results.items()}


def collect_generated_files(results: Dict[str, ToolResult]) -> Dict[str, str]:
    generated: Dict[str, str] = {}
    for result in results.values():
        generated.update(result.generated_files)
    return generated


def _compact_payload(payload: Any) -> Any:
    """Trim payloads to a bounded size for the LLM context."""
    if isinstance(payload, dict):
        return {str(key): _compact_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_compact_payload(item) for item in payload[:50]]
    return payload
