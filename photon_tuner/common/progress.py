"""Lightweight, dependency-free progress reporting for long tuning runs.

``ProgressReporter`` writes human-scannable one-line events to a stream (stdout
by default).  Everything is optional: pass ``enabled=False`` and every method
becomes a no-op, so a ``--quiet`` run emits nothing.

The reporter keeps a "current agent" so per-step ReAct events, decisions, and
edit outcomes can be attributed to the right component without threading extra
context through every call site.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional, TextIO


def summarize_payload(payload: Any) -> str:
    """One-line, size-aware description of a tool payload.

    Reports the item count plus the JSON byte size.  The byte size matters as
    much as the count: an unbounded tool result is re-sent to the LLM on every
    later call, which is usually the dominant cost of one iteration.
    """
    try:
        size = len(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception:
        size = 0
    size_label = f"{size / 1024:.1f} KB" if size >= 1024 else f"{size} B"

    if isinstance(payload, dict):
        list_fields = [(key, value) for key, value in payload.items() if isinstance(value, list)]
        if len(payload) == 1 and list_fields:
            key, value = list_fields[0]
            return f"{len(value)} {key} ({size_label})"
        if list_fields:
            key, value = max(list_fields, key=lambda item: len(item[1]))
            return f"{len(payload)} keys, {key}={len(value)} ({size_label})"
        return f"{len(payload)} keys ({size_label})"
    if isinstance(payload, list):
        return f"{len(payload)} items ({size_label})"
    return f"{type(payload).__name__} ({size_label})"


class ProgressReporter:
    """Small dependency-free progress reporter for long tuning runs."""

    def __init__(
        self,
        enabled: bool = True,
        stream: Optional[TextIO] = None,
        refresh_interval_s: float = 30.0,
    ) -> None:
        self.enabled = enabled
        self.stream = stream or sys.stdout
        self.refresh_interval_s = refresh_interval_s
        self._run_start = time.monotonic()
        self._current_agent: Optional[Dict[str, Any]] = None
        self._agent_start: Optional[float] = None

    def info(
        self,
        *,
        iteration: Optional[int],
        total_iterations: Optional[int],
        component: str,
        message: str,
    ) -> None:
        """Emit a free-form informational line."""
        if not self.enabled:
            return
        self._write(
            f"[{self._iteration_label(iteration, total_iterations)}]"
            f"[{self._component_label(component)}][INFO]"
            f"[elapsed={self._fmt_duration(time.monotonic() - self._run_start)}]"
            f"[eta=unknown] {message}"
        )

    # -- Live agent / step reporting --------------------------------------- #

    def begin_agent(
        self,
        *,
        component: str,
        iteration: Optional[int],
        total_iterations: Optional[int],
        detail: str = "",
    ) -> None:
        """Start watching one agent run; subsequent :meth:`step` calls report under it."""
        if not self.enabled:
            return
        self._current_agent = {
            "component": component,
            "iteration": iteration,
            "total_iterations": total_iterations,
        }
        self._agent_start = time.monotonic()
        prefix = (
            f"[{self._iteration_label(iteration, total_iterations)}]"
            f"[{self._component_label(component)}]"
            f"[START][elapsed={self._fmt_duration(time.monotonic() - self._run_start)}]"
        )
        self._write(f"{prefix} {detail}" if detail else f"{prefix} working")

    def end_agent(self, *, status: str = "DONE", detail: str = "") -> None:
        """Finish the currently watched agent and report its wall time."""
        if not self.enabled:
            return
        context = self._current_agent or {}
        elapsed = time.monotonic() - self._agent_start if self._agent_start else 0.0
        prefix = (
            f"[{self._iteration_label(context.get('iteration'), context.get('total_iterations'))}]"
            f"[{self._component_label(str(context.get('component', 'agent')))}]"
            f"[{status}][agent_time={self._fmt_short(elapsed)}]"
        )
        self._write(f"{prefix} {self._clip(detail, 160)}" if detail else prefix)
        self._current_agent = None
        self._agent_start = None

    def step(self, event: Dict[str, Any]) -> None:
        """Report one ReAct step (LLM decision or tool execution)."""
        if not self.enabled:
            return
        context = self._current_agent or {}
        head = (
            f"[{self._iteration_label(context.get('iteration'), context.get('total_iterations'))}]"
            f"[{self._component_label(str(context.get('component', 'agent')))}]"
            f"[step {int(event.get('step', 0)) + 1}/{int(event.get('total_steps', 0))}]"
        )
        kind = str(event.get("kind", ""))

        if kind == "llm_start":
            self._write(
                f"{head}[LLM] deciding next tool "
                f"(elapsed={self._fmt_duration(time.monotonic() - self._run_start)})"
            )
            return

        llm_s = self._fmt_short(float(event.get("llm_s", 0.0)))
        if kind == "tool":
            tool_s = self._fmt_short(float(event.get("tool_s", 0.0)))
            arguments = event.get("arguments") or {}
            rendered_args = self._fmt_args(arguments if isinstance(arguments, dict) else {})
            status = event.get("status", "ok")
            if status == "error":
                self._write(
                    f"{head}[ERROR] {event.get('tool', '')}{rendered_args} -> "
                    f"{event.get('error', '')} (llm={llm_s})"
                )
            else:
                self._write(
                    f"{head}[TOOL] {event.get('tool', '')}{rendered_args} -> "
                    f"{event.get('summary', '')} (llm={llm_s}, tool={tool_s})"
                )
            return

        if kind == "finish":
            self._write(f"{head}[END] agent finished tool use (llm={llm_s})")
        elif kind == "duplicate":
            self._write(f"{head}[END] duplicate call to {event.get('tool', '')}, stopping (llm={llm_s})")
        elif kind == "unknown_tool":
            self._write(f"{head}[WARN] unknown tool {event.get('tool', '')!r}, stopping (llm={llm_s})")
        elif kind == "malformed":
            self._write(f"{head}[WARN] malformed ReAct step, stopping (llm={llm_s})")

    def decision(
        self,
        *,
        iteration: Optional[int],
        total_iterations: Optional[int],
        component: str,
        message: str,
    ) -> None:
        """Report the compact outcome of an agent (what it decided)."""
        if not self.enabled:
            return
        self._write(
            f"[{self._iteration_label(iteration, total_iterations)}]"
            f"[{self._component_label(component)}][DECISION] {self._clip(message, 300)}"
        )

    def metric_delta(
        self,
        *,
        iteration: Optional[int],
        total_iterations: Optional[int],
        component: str,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> None:
        """Report the simulated metric change of one proposal."""
        if not self.enabled:
            return

        def _delta(key: str) -> str:
            old = float(before.get(key, 0.0) or 0.0)
            new = float(after.get(key, 0.0) or 0.0)
            if old == 0:
                return f"{key}: {new:.4g}"
            return f"{key}: {old:.4g}->{new:.4g} ({(new - old) / old * 100:+.1f}%)"

        message = " ".join(
            _delta(key) for key in ("latency_s", "energy_pj", "area_um2", "objective")
            if key in before or key in after
        )
        self._write(
            f"[{self._iteration_label(iteration, total_iterations)}]"
            f"[{self._component_label(component)}][RESULT] {message}"
        )

    def edit(
        self,
        *,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        status: str = "ok",
        error: str = "",
        detail: str = "",
    ) -> None:
        """Report one hardware-edit application (applied or rejected).

        Edits happen *after* the ReAct loop, so without this hook a rejected
        edit is invisible while the final ``strategy`` text may still describe
        it as if it had been applied.
        """
        if not self.enabled:
            return
        context = self._current_agent or {}
        head = (
            f"[{self._iteration_label(context.get('iteration'), context.get('total_iterations'))}]"
            f"[{self._component_label(str(context.get('component', 'agent')))}]"
        )
        rendered = self._fmt_args(arguments or {})
        if status == "ok":
            suffix = f" {detail}" if detail else ""
            self._write(f"{head}[EDIT][ok]   {name}{rendered}{suffix}")
        else:
            self._write(f"{head}[EDIT][FAIL] {name}{rendered} -> {self._clip(error, 140)}")

    # -- small formatting helpers ------------------------------------------ #

    def _fmt_short(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes}m{secs:02d}s"

    def _clip(self, text: Any, limit: int = 200) -> str:
        """Collapse whitespace and clip long LLM prose so lines stay scannable."""
        collapsed = " ".join(str(text).split())
        if len(collapsed) <= limit:
            return collapsed
        return collapsed[: limit - 3] + "..."

    def _fmt_args(self, arguments: Dict[str, Any]) -> str:
        if not arguments:
            return "()"
        parts: List[str] = []
        for key, value in arguments.items():
            try:
                text = json.dumps(value, ensure_ascii=False, default=str)
            except Exception:
                text = str(value)
            if len(text) > 40:
                text = text[:37] + "..."
            parts.append(f"{key}={text}")
        rendered = "(" + ", ".join(parts) + ")"
        return rendered if len(rendered) <= 100 else rendered[:96] + "...)"

    def _write(self, line: str) -> None:
        print(line, file=self.stream, flush=True)

    def _iteration_label(self, iteration: Optional[int], total_iterations: Optional[int]) -> str:
        if iteration is None:
            return "iter=n/a"
        if iteration <= 0:
            return "iter=init"
        if total_iterations is None:
            return f"iter={iteration}"
        return f"iter={iteration}/{total_iterations}"

    def _component_label(self, component: str) -> str:
        labels = {
            "agent:model_level": "model_agent",
            "agent:layer_level": "layer_agent",
            "agent:optical_level": "optical_agent",
            "agent:solution_generation": "solution_agent",
        }
        return labels.get(component, component)

    def _fmt_duration(self, seconds: float) -> str:
        total = int(max(seconds, 0.0))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
