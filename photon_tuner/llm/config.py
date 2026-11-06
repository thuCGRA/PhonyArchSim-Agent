"""光子 DSE 框架的 LLM 客户端配置。

一个普通数据类，描述如何调用 LLM 提供方（提供方、模型、温度、重试策略等）。
CLI 填充其中一个实例并传给 :func:`photon_tuner.llm.model.build_chat_model`，
后者构建 LangGraph 运行时使用的 LangChain ``ChatOpenAI`` 模型。

刻意不设 ``api_key`` 字段：凭据不属于配置对象。密钥来自 ``OPENAI_API_KEY``
环境变量（见 ``configs/api_keys.py``）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    provider: str = "mock"
    model: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    timeout: float = 600.0
    return_reasoning: bool = False
    retry_attempts: int = 2
    retry_temperature: float = 0.1
    enable_cache: bool = True
    stream: bool = False

    def __post_init__(self) -> None:
        # 对明显不合理的配置尽早报错，而非稍后抛出难以定位的 SDK 错误。
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("provider must be a non-empty string")
        if not 0.0 <= float(self.temperature) <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature!r}")
        if float(self.timeout) <= 0:
            raise ValueError(f"timeout must be > 0, got {self.timeout!r}")
        if int(self.retry_attempts) < 0:
            raise ValueError(f"retry_attempts must be >= 0, got {self.retry_attempts!r}")
        if self.max_tokens is not None and int(self.max_tokens) <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens!r}")
        if not isinstance(self.enable_cache, bool):
            raise ValueError(f"enable_cache must be a bool, got {self.enable_cache!r}")
        if not isinstance(self.stream, bool):
            raise ValueError(f"stream must be a bool, got {self.stream!r}")
