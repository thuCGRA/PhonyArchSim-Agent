"""LLM client configuration for the photonic DSE framework.

A plain dataclass describing how to call an LLM provider (which provider,
model, temperature, retry policy, ...).  The CLI fills one of these and passes
it to :func:`photon_tuner.llm.clients.create_llm_client`, which builds the
concrete client (mock or OpenAI-compatible).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    provider: str = "mock"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    timeout: float = 600.0
    return_reasoning: bool = False
    retry_attempts: int = 2
    retry_temperature: float = 0.1
