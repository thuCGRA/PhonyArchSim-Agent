"""LLM client implementations for the photonic DSE framework.

Two providers are supported:

``mock``
    A deterministic, dependency-free client that returns schema-valid JSON for
    every agent task.  Used for offline smoke tests and CI.

``openai-compatible`` (and the ``paratera`` / ``deepseek`` aliases)
    A dependency-free chat-completions client implemented with the standard
    library (``urllib``), so no ``openai`` SDK or ``pip`` is required.  It
    speaks the OpenAI ``POST /v1/chat/completions`` protocol and is verified
    against the Paratera-hosted DeepSeek-V4-Pro-0813 endpoint.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from photon_tuner.llm.config import LLMConfig


# Provider presets.  ``paratera`` hosts the model this environment uses
# (DeepSeek-V4-Pro-0813); ``deepseek-official`` is the public DeepSeek API.
PROVIDER_PRESETS: Dict[str, Dict[str, str]] = {
    "paratera": {
        "base_url": "https://llmapi.paratera.com/v1",
        "model": "DeepSeek-V4-Pro-0813",
    },
    "paratera-flash": {
        "base_url": "https://llmapi.paratera.com/v1",
        "model": "DeepSeek-V4-Flash",
    },
    "deepseek": {
        "base_url": "https://llmapi.paratera.com/v1",
        "model": "DeepSeek-V4-Pro-0813",
    },
    "deepseek-official": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
}

# Environment variables tried, in order, when no explicit key is configured.
API_KEY_ENV_NAMES = ("PARATERA_API_KEY", "DEEPSEEK_API_KEY")


class LLMClient:
    """Minimal common interface used by the ReAct loop and agents."""

    trace_enabled: bool = False
    retry_attempts: int = 2
    retry_temperature: float = 0.1

    #: Optional ``photon_tuner.llm.tracing.LLMTraceRecorder``.  When attached,
    #: every request/response pair is persisted so a run can be audited offline
    #: instead of trusting a summary produced after the fact.
    recorder: Optional[Any] = None

    def complete_json_messages(self, messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError

    def complete_json(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        return self.complete_json_messages(messages)

    # -- optional trace hooks ------------------------------------------------ #

    def _trace_begin(self, request_body: Dict[str, Any]) -> Optional[int]:
        """Open a trace record. Tracing must never break a run."""
        recorder = getattr(self, "recorder", None)
        if recorder is None:
            return None
        try:
            return recorder.begin_call(request_body)
        except Exception:
            return None

    def _trace_end(
        self,
        call_index: Optional[int],
        raw_output: Dict[str, Any],
        response_content: Optional[str] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        """Close a trace record with the raw response. Never raises."""
        if call_index is None:
            return
        recorder = getattr(self, "recorder", None)
        if recorder is None:
            return
        try:
            recorder.end_call(
                call_index,
                raw_output,
                response_content=response_content,
                error=error,
            )
        except Exception:
            pass


def extract_json_object(text: str) -> Dict[str, Any]:
    """Return the first balanced JSON object embedded in an LLM response."""
    text = _strip_markdown_fence(text)
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in LLM response")
    value = _parse_balanced_object(text, start)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object, got {type(value).__name__}")
    return value


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[A-Za-z]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped


def _parse_balanced_object(text: str, start: int) -> Any:
    depth = 0
    in_string = False
    escaped = False
    for end in range(start, len(text)):
        char = text[end]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : end + 1])
    raise ValueError("Unbalanced JSON object in LLM response")


class MockLLMClient(LLMClient):
    """Deterministic mock returning schema-valid JSON for each agent task."""

    def complete_json_messages(self, messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": "mock",
            "messages": [dict(message) for message in messages],
        }
        call_index = self._trace_begin(payload)
        try:
            result = self._mock_response(messages)
        except Exception as exc:
            self._trace_end(call_index, {}, None, error=exc)
            raise
        content = json.dumps(result, ensure_ascii=False)
        self._trace_end(call_index, {"response": {"content": content}}, content)
        return result

    @staticmethod
    def _mock_response(messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        task = _detect_task(messages)
        if task == "react_tool_use":
            # The offline mock gathers no tool evidence; agents then produce
            # their final JSON from the (already rich) context payload.
            return {
                "thought": "Context already carries the evidence needed for this role.",
                "action": "finish",
                "finish_reason": "mock: sufficient context",
            }
        return _mock_final(task)

    def complete_json(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        task = str(user_payload.get("task", "unknown"))
        if task == "react_tool_use":
            return {
                "thought": "Context already carries the evidence needed for this role.",
                "action": "finish",
                "finish_reason": "mock: sufficient context",
            }
        return _mock_final(task)


class OpenAICompatibleClient(LLMClient):
    """Dependency-free chat-completions client (stdlib ``urllib`` only)."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.retry_attempts = max(0, int(config.retry_attempts or 0))
        self.retry_temperature = config.retry_temperature
        self.api_key = resolve_api_key(config)
        self.base_url = (config.base_url or "https://api.openai.com/v1").rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            self.base_url = self.base_url[: -len("/chat/completions")]
        self.model = config.model or "DeepSeek-V4-Pro-0813"

    def complete_json_messages(self, messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            payload["max_tokens"] = int(self.config.max_tokens)
        call_index = self._trace_begin(payload)
        raw: Dict[str, Any] = {}
        content = ""
        try:
            content, raw = self._chat_completion(payload)
            parsed = extract_json_object(content)
        except Exception as exc:
            # Record the failure too: an unparseable or failed call is exactly
            # what an offline audit needs to see.
            self._trace_end(call_index, {"response": raw}, content or None, error=exc)
            raise
        self._trace_end(call_index, {"response": raw}, content)
        return parsed

    def _chat_completion(self, payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Return ``(assistant_content, full_response_body)``."""
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_error: Optional[BaseException] = None
        for attempt in range(self.retry_attempts + 1):
            try:
                request = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return str(data["choices"][0]["message"]["content"] or ""), data
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in (429, 500, 502, 503, 504) or attempt >= self.retry_attempts:
                    break
                time.sleep(min(2 ** attempt, 8))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as exc:
                last_error = exc
                if attempt >= self.retry_attempts:
                    break
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(_format_request_error(last_error))


def create_llm_client(config: LLMConfig, recorder: Optional[Any] = None) -> LLMClient:
    provider = (config.provider or "mock").lower()
    if provider == "mock":
        client: LLMClient = MockLLMClient()
    elif provider in ("openai-compatible", "paratera", "paratera-flash", "deepseek", "deepseek-official"):
        client = OpenAICompatibleClient(_apply_provider_preset(config, provider))
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    if recorder is not None:
        client.recorder = recorder
    return client


def resolve_api_key(config: LLMConfig) -> str:
    """Resolve an API key from config, environment, or the DSH credentials file."""
    if config.api_key:
        return config.api_key
    for env_name in API_KEY_ENV_NAMES:
        key = os.environ.get(env_name)
        if key:
            return key
    credentials_path = Path.home() / ".dsh" / ".credentials.yaml"
    if credentials_path.exists():
        try:
            import yaml

            data = yaml.safe_load(credentials_path.read_text(encoding="utf-8")) or {}
            refs = data.get("refs", {}) or {}
            for name in API_KEY_ENV_NAMES:
                if refs.get(name):
                    return str(refs[name])
        except Exception:
            pass
    raise ValueError(
        "No LLM API key available. Set config.api_key or the "
        f"{' / '.join(API_KEY_ENV_NAMES)} environment variable."
    )


def _apply_provider_preset(config: LLMConfig, provider: str) -> LLMConfig:
    """Fill unset base_url/model from the provider preset (if any)."""
    preset = PROVIDER_PRESETS.get(provider, {})
    updates: Dict[str, Any] = {}
    if not config.model and preset.get("model"):
        updates["model"] = preset["model"]
    if not config.base_url and preset.get("base_url"):
        updates["base_url"] = preset["base_url"]
    return replace(config, **updates) if updates else config


def _format_request_error(error: Optional[BaseException]) -> str:
    if isinstance(error, urllib.error.HTTPError):
        try:
            detail = error.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            detail = ""
        return f"LLM request failed with HTTP {error.code}: {detail}".strip()
    return f"LLM request failed: {error}"


# --------------------------------------------------------------------------- #
# Deterministic mock outputs (schema-valid for the agents' validation code).
# --------------------------------------------------------------------------- #

def _detect_task(messages: Sequence[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("task"):
            return str(payload["task"])
    return "unknown"


def _mock_final(task: str) -> Dict[str, Any]:
    if task == "model_level_analysis":
        return {
            "candidate_layers": [
                {"layer_id": 0, "layer_name": "conv1", "group": "convolution", "rank_metric": "energy_pj", "rank_value": 1.0, "impact_types": ["energy"], "root_causes": ["device dynamic energy"]},
                {"layer_id": 1, "layer_name": "linear", "group": "linear", "rank_metric": "latency_s", "rank_value": 1.0, "impact_types": ["latency"], "root_causes": ["memory loading"]},
            ],
            "global_findings": ["photonic crossbar energy dominates the objective"],
            "selected_views": ["energy", "area"],
            "summary": "Mock model-level analysis: energy is the dominant cost driver.",
            "analysis_base": {},
            "llm_notes": {"mock": True},
        }
    if task == "layer_level_analysis":
        return {
            "primary_impact": "energy",
            "dominant_root_cause": "device",
            "layer_diagnoses": [
                {"layer_id": 0, "layer_name": "conv1", "impact_types": ["energy"], "root_causes": ["MZM dynamic energy"], "ratios": {"compute": 0.5, "memory": 0.1, "communication": 0.05, "buffer": 0.05, "device": 0.3}},
            ],
            "retrieval_description": "conv1 energy bottleneck driven by MZM and laser power",
            "root_cause_summary": {"compute": 0.5, "memory": 0.1, "communication": 0.05, "buffer": 0.05, "device": 0.3},
            "recommended_focus": ["mzm efficiency", "laser wall-plug efficiency"],
            "llm_notes": {"mock": True},
        }
    if task == "optical_level_analysis":
        return {
            "loss_budget_diagnosis": "core path loss is dominated by the on-chip laser and the two MRR rerouters.",
            "dominant_loss_contributors": [
                {"device_type": "on_chip_laser", "device_name": "Customized_Laser", "insertion_loss_dB": 2.0},
                {"device_type": "mrr_rerouter", "device_name": "LT_MRR_REROUTER", "insertion_loss_dB": 1.86},
                {"device_type": "mzm", "device_name": "LT_MZM", "insertion_loss_dB": 1.2},
            ],
            "power_budget_margin": "6.22 dB core path loss leaves limited optical power margin.",
            "recommended_link_edits": [
                {"level": "core", "device_type": "on_chip_laser", "device_name": "Customized_Laser", "reason": "dominant loss contributor"},
            ],
            "retrieval_description": "reduce dominant optical path insertion loss",
            "summary": "Mock optical-level analysis: laser and MRR rerouters dominate path loss.",
            "llm_notes": {"mock": True},
        }
    if task == "solution_generation":
        return {
            "strategy": "reduce dominant optical loss and shrink the optical crossbar",
            "actions": [
                {"name": "step_hardware_parameter", "arguments": {"parameter": "num_wavelength", "direction": -1, "scope": "all"}}
            ],
            "link_edits": [
                {"name": "modify_device_parameter", "arguments": {"level": "core", "device_type": "on_chip_laser", "device_name": "Customized_Laser"}}
            ],
            "retrieved_cases": [],
            "rationale": "Mock solution: select a lower-loss laser and reduce wavelength count.",
            "notes": {"expected_objective_direction": "decrease"},
        }
    return {"mock": True, "task": task}
