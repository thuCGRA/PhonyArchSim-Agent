"""DSE 运行时的进程内 LLM 响应缓存。

相同的 ``(agent, payload)`` 组合直接返回已缓存的结构化输出与 transcript，
无需再次调用模型，从而在 gate 返工和重复相同载荷时节省 token。缓存键由角色
+ 序列化载荷的 SHA-256 摘要组成，任何载荷变化（例如 gate 反馈）都会干净地
未命中。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional


def enable_langchain_llm_cache() -> None:
    """启用 LangChain 全局内存 LLM 缓存。

    缓存的是原始 ``llm(prompt)`` 调用层（prompt → 输出），与智能体级的
    :class:`ResponseCache` 互补：框架缓存跳过模型调用（节省真实 API token），
    而 ``ResponseCache`` 短路整个工具循环回放。导入被延迟，使模块在缺少
    ``langchain_core``（域测试环境）时仍可导入。
    """
    from langchain_core.caches import InMemoryCache
    from langchain_core.globals import set_llm_cache

    set_llm_cache(InMemoryCache())


class ResponseCache:
    """以角色 + 载荷摘要为键的有界进程内 FIFO 缓存。"""

    def __init__(self, enabled: bool = True, max_entries: int = 256) -> None:
        self.enabled = enabled
        self.max_entries = max_entries
        self._store: Dict[str, Dict[str, Any]] = {}

    def key(self, name: str, payload: Dict[str, Any]) -> str:
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()[:24]
        return f"{name}:{digest}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        return self._store.get(key)

    def put(self, key: str, out: Any, transcript: List[Any], error: Optional[str]) -> None:
        if not self.enabled:
            return
        if len(self._store) >= self.max_entries:
            self._store.pop(next(iter(self._store)))  # FIFO 淘汰
        self._store[key] = {"out": out, "transcript": transcript, "error": error}
