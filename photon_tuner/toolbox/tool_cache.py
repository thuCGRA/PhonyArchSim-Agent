"""确定性的报告工具结果缓存。

只读报告工具（``summarize_*`` / ``inspect_*`` / ``list_*`` /
``compare_*``）是 ``(hardware, arguments)`` 的纯函数；缓存其结果可避免重复
计算相同的 SimPhony 派生证据。会变更状态的工具（edits）以及
``select_analysis_base``（它会切换 ``ToolContext.active_base``）绝不被缓存。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def _stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


class ToolResultCache:
    """以 ``(工具名, hardware, arguments)`` 为键的有界 FIFO 缓存。"""

    def __init__(self, max_entries: int = 256) -> None:
        self.max_entries = max_entries
        self._store: Dict[str, Any] = {}

    def key(self, name: str, hardware: Any, arguments: Dict[str, Any]) -> str:
        hw = hardware if isinstance(hardware, dict) else {}
        digest = hashlib.sha256(
            f"{name}|{_stable(hw)}|{_stable(arguments)}".encode("utf-8")
        ).hexdigest()[:24]
        return digest

    def get(self, key: str) -> Any:
        return self._store.get(key)

    def put(self, key: str, value: Any) -> None:
        if len(self._store) >= self.max_entries:
            self._store.pop(next(iter(self._store)))  # FIFO 淘汰
        self._store[key] = value
