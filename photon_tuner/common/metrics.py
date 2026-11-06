"""光子 DSE 运行时的运行级聚合指标。

除每次调用的 ``llm_trace.jsonl`` 之外，一次运行还应以紧凑的*聚合*摘要收尾——
包含各阶段的调用次数、错误、token 用量、耗时和门控结果，以及运行总计。
这些数据用于实验跟踪和消融对比。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _tokens(usage: Any) -> int:
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens") or usage.get("total")
    if total:
        return int(total)
    return (
        int(usage.get("prompt_tokens") or 0)
        + int(usage.get("completion_tokens") or 0)
        + int(usage.get("input_tokens") or 0)
        + int(usage.get("output_tokens") or 0)
    )


@dataclass
class RunMetrics:
    """在一次 ``tune()`` 运行期间累计各阶段的调用/门控统计。"""

    stage_calls: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    gate_outcomes: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def record_agent_call(
        self,
        stage: str,
        duration_s: float,
        error: Optional[str],
        token_usage: Dict[str, Any],
    ) -> None:
        self.stage_calls.setdefault(stage, []).append({
            "duration_s": round(duration_s, 4),
            "error": error,
            "token_usage": token_usage or {},
        })

    def record_gate(self, stage: str, decision: Optional[str]) -> None:
        bucket = self.gate_outcomes.setdefault(stage, {})
        bucket[decision or "unknown"] = bucket.get(decision or "unknown", 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        stages: Dict[str, Any] = {}
        total_calls = 0
        total_errors = 0
        total_duration = 0.0
        total_tokens = 0
        for stage, calls in self.stage_calls.items():
            calls_n = len(calls)
            errors = sum(1 for call in calls if call.get("error"))
            duration = sum(call.get("duration_s") or 0.0 for call in calls)
            tokens = sum(_tokens(call.get("token_usage")) for call in calls)
            total_calls += calls_n
            total_errors += errors
            total_duration += duration
            total_tokens += tokens
            stages[stage] = {
                "calls": calls_n,
                "errors": errors,
                "duration_s": round(duration, 4),
                "tokens_total": tokens,
                "gate": self.gate_outcomes.get(stage, {}),
            }
        return {
            "totals": {
                "calls": total_calls,
                "errors": total_errors,
                "duration_s": round(total_duration, 4),
                "tokens_total": total_tokens,
            },
            "stages": stages,
        }
