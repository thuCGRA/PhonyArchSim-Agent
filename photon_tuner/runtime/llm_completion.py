"""LLM JSON completion helpers for the ReAct tool loop."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from photon_tuner.llm.clients import LLMClient


def complete_json_continuing_messages(
    llm: LLMClient,
    messages: List[Dict[str, Any]],
    final_payload: Dict[str, Any],
    validate_response: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Dict[str, Any]:
    """Ask the LLM to continue from the ReAct transcript and emit final JSON."""
    continued = [dict(message) for message in messages]
    final_payload = dict(final_payload)
    # The ReAct system prompt is still present in ``messages`` and tells the
    # model to answer only with a tool call or ``finish``.  A weaker LLM keeps
    # following that protocol for this final step and dumps everything into
    # ``thought`` (then truncates before the real fields).  Make the schema
    # switch explicit so the final answer is emitted directly.
    final_payload["_format"] = (
        "The ReAct tool-use loop is finished. This is the FINAL answer step, not "
        "a ReAct decision: emit one strict JSON object containing ONLY the fields "
        "required by 'instruction'. Do NOT emit 'action', 'tool', 'finish_reason', "
        "or a 'thought' field, and do not wrap the result in markdown."
    )
    continued.append(
        {"role": "user", "content": json.dumps(final_payload, ensure_ascii=False)}
    )
    return complete_json_with_retries(llm, continued, validate_response=validate_response)


def complete_json_with_retries(
    llm: LLMClient,
    messages: List[Dict[str, Any]],
    validate_response: Optional[Callable[[Dict[str, Any]], Any]] = None,
    retry_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call the LLM, retrying on malformed or schema-invalid responses."""
    max_retries = max(0, int(getattr(llm, "retry_attempts", 0) or 0))
    last_error: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            response = llm.complete_json_messages(messages)
            if validate_response is not None:
                validate_response(response)
            return response
        except (ValueError, KeyError, TypeError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "instruction": "The previous response was invalid. Fix it.",
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                            "retry_attempt": attempt + 1,
                        },
                        ensure_ascii=False,
                    ),
                }
            )
    if last_error is not None:
        raise last_error
    raise ValueError("LLM returned no valid response")
