"""为 LangGraph 运行时构建聊天模型。

LangGraph 运行时需要一个支持工具调用的 LangChain 聊天模型；Paratera 托管的
DeepSeek 端点兼容 OpenAI 的 chat-completions 协议并支持 ``tools``/``tool_calls``，
因此基于它构建 ``ChatOpenAI``。

凭据约定：密钥存放在 ``OPENAI_API_KEY`` 环境变量中，构造模型时由 LangChain
自动读取。此处不从配置文件、CLI 参数或凭据存储中读取密钥，``OPENAI_API_KEY``
是唯一来源。可用 ``configs/api_keys.py`` 在终端中设置该变量：

    eval "$(python configs/api_keys.py)"
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

import tenacity
from langchain_openai import ChatOpenAI

from photon_tuner.common.logging import get_logger
from photon_tuner.llm.config import LLMConfig

log = get_logger("llm.model")


def _transient_exc_types() -> tuple:
    """返回 OpenAI 瞬时错误类（超时 / 连接 / 限流 / 5xx）。"""
    try:
        import openai  # noqa: PLC0415

        return (
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
        )
    except Exception:  # pragma: no cover - deep 虚拟环境中始终存在 openai
        return ()


TRANSIENT_EXC_TYPES = _transient_exc_types()


def is_transient_llm_error(exc: BaseException) -> bool:
    """当 ``exc`` 是值得重试的瞬时 LLM/网络错误时返回 True。

    刻意排除确定性错误（非法请求、鉴权、校验、``GraphRecursionError`` 等）：
    重试只会浪费 token。
    """
    if isinstance(exc, tenacity.RetryError):
        return False  # 已重试过；不再二次包裹
    if TRANSIENT_EXC_TYPES and isinstance(exc, TRANSIENT_EXC_TYPES):
        return True
    try:
        import openai  # noqa: PLC0415

        if isinstance(exc, openai.APIStatusError):
            return exc.status_code == 429 or exc.status_code >= 500
    except Exception:  # pragma: no cover
        pass
    return False


def llm_retry(attempts: int):
    """tenacity 装饰器：以指数退避重试瞬时 LLM 错误。"""
    return tenacity.retry(
        retry=tenacity.retry_if_exception(is_transient_llm_error),
        stop=tenacity.stop_after_attempt(max(1, int(attempts))),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )

# 提供方预设。``paratera`` 托管本环境使用的模型（DeepSeek-V4-Pro-0813）；
# ``deepseek-official`` 是 DeepSeek 官方 API。
PROVIDER_PRESETS = {
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

# 唯一读取密钥的位置。
API_KEY_ENV_NAME = "OPENAI_API_KEY"
# 可选端点覆盖，用于完全由环境驱动的部署；优先级介于显式 ``config.base_url``
# 与提供方预设之间。
BASE_URL_ENV_NAME = "OPENAI_BASE_URL"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


def require_api_key() -> str:
    """返回 ``OPENAI_API_KEY``，缺失则尽早失败并给出修复提示。

    密钥不会传给 ``ChatOpenAI``——此处仅做前置校验，让缺失密钥以一行可执行的
    提示浮出，而非晦涩的 SDK 错误。值本身始终来自环境变量。
    """
    key = os.environ.get(API_KEY_ENV_NAME, "").strip()
    if not key:
        raise ValueError(
            f"{API_KEY_ENV_NAME} is not set.  Create it in this terminal first:\n"
            '    eval "$(python configs/api_keys.py)"\n'
            f"or export it by hand:  export {API_KEY_ENV_NAME}=\"sk-...\""
        )
    return key


def _apply_provider_preset(config: LLMConfig) -> LLMConfig:
    """从提供方预设（若有）补齐未设置的 base_url/model。"""
    preset = PROVIDER_PRESETS.get(config.provider, {})
    updates = {}
    if not config.model and preset.get("model"):
        updates["model"] = preset["model"]
    # 预设永不覆盖 shell 中显式设置的 OPENAI_BASE_URL。
    if (
        not config.base_url
        and not os.environ.get(BASE_URL_ENV_NAME)
        and preset.get("base_url")
    ):
        updates["base_url"] = preset["base_url"]
    return replace(config, **updates) if updates else config


def resolve_base_url(config: LLMConfig) -> str:
    """端点优先级：配置值 > OPENAI_BASE_URL > 预设 > 默认值。

    先应用提供方预设，因此对裸 ``LLMConfig(provider="paratera")`` 调用本函数，
    会返回模型实际使用的 URL。
    """
    config = _apply_provider_preset(config)
    return (
        config.base_url or os.environ.get(BASE_URL_ENV_NAME) or DEFAULT_BASE_URL
    ).rstrip("/")


def build_chat_model(
    config: LLMConfig,
    *,
    temperature: Optional[float] = None,
) -> ChatOpenAI:
    """构建密钥来自环境变量的、支持工具调用的 ``ChatOpenAI``。"""
    config = _apply_provider_preset(config)
    require_api_key()

    kwargs = {
        "model": config.model or "DeepSeek-V4-Pro-0813",
        # ``api_key`` 刻意省略：ChatOpenAI 构造时自动从环境读取 OPENAI_API_KEY。
        "base_url": resolve_base_url(config),
        "temperature": temperature if temperature is not None else config.temperature,
        "max_retries": max(0, int(config.retry_attempts or 0)),
    }
    if config.max_tokens is not None:
        kwargs["max_tokens"] = int(config.max_tokens)
    if config.timeout:
        kwargs["timeout"] = float(config.timeout)
    # 流式输出：启用时 ChatOpenAI 通过 ``on_llm_new_token`` 逐 token 输出，
    # 由 trace 收集器转发到 stderr。
    kwargs["streaming"] = bool(config.stream)
    return ChatOpenAI(**kwargs)
