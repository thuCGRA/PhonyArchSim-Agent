"""Dependency-free ReAct tool loop for the photonic DSE agents.

A plain imperative loop (no LangGraph) that implements the shared
evidence-gathering protocol used by every agent: one tool call per step, and
duplicate calls end tool use.  Keeping it dependency-free means the framework
still runs offline with a mock LLM.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Sequence

from photon_tuner.runtime.results import ReactLoopResult
from photon_tuner.runtime.tool_specs_for_prompt import tool_specs_for_prompt
from photon_tuner.common.progress import summarize_payload
from photon_tuner.common.tool_types import ToolResult
from photon_tuner.llm.clients import LLMClient


def _report(progress: Any, event: Dict[str, Any]) -> None:
    """Emit a progress event. Reporting must never break a run."""
    if progress is None:
        return
    try:
        progress.step(event)
    except Exception:
        pass


def run_react_tool_loop(
    llm: LLMClient,
    toolbox: Any,
    system_prompt: str,
    task: str,
    agent_name: str,
    agent_goal: str,
    context_payload: Dict[str, Any],
    context: Any,
    max_steps: int = 8,
    allowed_tool_names: Optional[Sequence[str]] = None,
    progress: Any = None,
) -> ReactLoopResult:
    """Run one agent's evidence-gathering loop and return its transcript.

    When ``progress`` is supplied each step emits a live event (LLM decision
    latency, chosen tool, argument summary, result size, tool latency).
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    allowed = set(allowed_tool_names) if allowed_tool_names is not None else None
    prompt_tool_specs = tool_specs_for_prompt(toolbox, allowed_tool_names=sorted(allowed) if allowed else None)
    valid_names = {spec["name"] for spec in prompt_tool_specs}

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": _react_system_prompt(system_prompt, prompt_tool_specs),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": task,
                    "agent_name": agent_name,
                    "agent_goal": agent_goal,
                    "context": context_payload,
                    "instruction": "Choose exactly one next tool call or finish tool use.",
                },
                ensure_ascii=False,
            ),
        },
    ]

    tool_results: Dict[str, ToolResult] = {}
    transcript: List[Dict[str, Any]] = []
    seen_tool_calls: List[str] = []

    for step in range(max_steps):
        _report(progress, {"kind": "llm_start", "step": step, "total_steps": max_steps})
        llm_started = time.monotonic()
        response = _llm_step(llm, messages)
        llm_s = time.monotonic() - llm_started
        action = response.get("action")

        if action == "finish":
            messages.append({"role": "assistant", "content": json.dumps(response, ensure_ascii=False)})
            transcript.append({"step": step, "action": "finish", "response": response})
            _report(progress, {"kind": "finish", "step": step, "total_steps": max_steps, "llm_s": llm_s})
            break

        if action != "tool":
            # A real LLM may occasionally emit a malformed step; end tool use
            # gracefully rather than crashing the whole loop.
            messages.append({"role": "assistant", "content": json.dumps(response, ensure_ascii=False)})
            transcript.append({"step": step, "action": "malformed", "response": response})
            _report(progress, {"kind": "malformed", "step": step, "total_steps": max_steps, "llm_s": llm_s})
            break

        tool = response.get("tool")
        tool = tool if isinstance(tool, dict) else {}
        tool_name = str(tool.get("name", ""))
        arguments = tool.get("arguments")
        arguments = arguments if isinstance(arguments, dict) else {}

        if tool_name not in valid_names:
            messages.append({"role": "assistant", "content": json.dumps(response, ensure_ascii=False)})
            transcript.append({"step": step, "action": "unknown_tool", "tool": tool_name})
            _report(progress, {"kind": "unknown_tool", "step": step, "total_steps": max_steps,
                               "tool": tool_name, "llm_s": llm_s})
            break

        call_key = json.dumps({"name": tool_name, "arguments": arguments}, sort_keys=True)
        if call_key in seen_tool_calls:
            messages.append({"role": "assistant", "content": json.dumps(response, ensure_ascii=False)})
            transcript.append({"step": step, "action": "duplicate", "tool": tool_name})
            _report(progress, {"kind": "duplicate", "step": step, "total_steps": max_steps,
                               "tool": tool_name, "llm_s": llm_s})
            break
        seen_tool_calls.append(call_key)

        tool_started = time.monotonic()
        try:
            result = toolbox.run(tool_name, context, arguments=arguments)
            tool_results[tool_name] = result
            observation = {"name": tool_name, "payload": result.payload}
            _report(progress, {
                "kind": "tool", "step": step, "total_steps": max_steps,
                "tool": tool_name, "arguments": arguments, "status": "ok",
                "summary": summarize_payload(result.payload),
                "llm_s": llm_s, "tool_s": time.monotonic() - tool_started,
            })
        except (ValueError, KeyError, TypeError) as exc:
            # A bad argument must not crash the loop; surface the error so the
            # LLM can correct its next call.
            observation = {"name": tool_name, "error": str(exc)}
            _report(progress, {
                "kind": "tool", "step": step, "total_steps": max_steps,
                "tool": tool_name, "arguments": arguments, "status": "error",
                "error": str(exc),
                "llm_s": llm_s, "tool_s": time.monotonic() - tool_started,
            })
        messages.append({"role": "assistant", "content": json.dumps(response, ensure_ascii=False)})
        messages.append({"role": "user", "content": json.dumps({"tool_observation": observation}, ensure_ascii=False)})
        transcript.append({"step": step, "action": "tool", "tool": tool_name, "observation": observation})

    return ReactLoopResult(tool_results=tool_results, transcript=transcript, messages=messages)


def _llm_step(llm: LLMClient, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One ReAct decision step with a small retry on malformed responses."""
    retries = max(0, int(getattr(llm, "retry_attempts", 0) or 0))
    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            response = llm.complete_json_messages(messages)
            if not isinstance(response, dict):
                raise ValueError("ReAct response must be a JSON object")
            return response
        except (ValueError, KeyError, TypeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {"instruction": "Your previous ReAct response was invalid; re-emit strict JSON with action 'tool' or 'finish'.",
                         "error": str(exc)},
                        ensure_ascii=False,
                    ),
                }
            )
    # After retries, return a benign finish so the agent can still emit its
    # final JSON from the (already rich) context payload.
    return {"action": "finish", "finish_reason": f"invalid_react_response: {last_error}"}


def _react_system_prompt(system_prompt: str, available_tools: List[Dict[str, Any]]) -> str:
    tools_json = json.dumps(available_tools, indent=2, ensure_ascii=False)
    return f"{system_prompt}\n\nAVAILABLE_TOOLS = {tools_json}"
