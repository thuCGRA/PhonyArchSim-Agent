"""agent 载荷的上下文长度预算管理。

LLM prompt 必须保持在 token 预算之内；当无法满足时，最高优先级的信息必须
保留下来。这些无依赖的辅助函数按深度优先截断载荷——长字符串被裁剪（保留
头部 + 尾部）、长列表被限制条目数——同时保留每个 dict 键和整体结构。
"""

from __future__ import annotations

import json
from typing import Any

# 默认载荷总预算（字符数，按保守的 1 token ≈ 4 字符计约 6k token）。
# agent 仍需为工具结果预留空间，因此 *prompt* 载荷被保持在远小于模型
# 完整上下文窗口的水平。
DEFAULT_MAX_PAYLOAD_CHARS = 24_000

_OMIT = "...[truncated]..."


def truncate_string(text: str, max_chars: int) -> str:
    """将 ``text`` 裁剪到 ``max_chars`` 个字符，同时保留头部和尾部。"""
    if len(text) <= max_chars:
        return text
    keep_head = max_chars * 2 // 3
    keep_tail = max_chars - keep_head - len(_OMIT)
    if keep_tail <= 0:
        return text[:max_chars]
    return text[:keep_head] + _OMIT + text[-keep_tail:]


def _fit(value: Any, field_cap: int, list_cap: int) -> Any:
    if isinstance(value, str):
        return truncate_string(value, field_cap)
    if isinstance(value, (list, tuple)):
        return [_fit(item, field_cap, list_cap) for item in list(value)[:list_cap]]
    if isinstance(value, dict):
        return {key: _fit(item, field_cap, list_cap) for key, item in value.items()}
    return value


def fit_payload(payload: Any, max_chars: int = DEFAULT_MAX_PAYLOAD_CHARS) -> Any:
    """返回 ``payload`` 的 JSON 安全副本，序列化后不超过 ``max_chars``。

    共两轮：先做温和的一轮（单字符串上限 = 预算/8，单列表上限 = 预算/1200
    项）；若序列化结果仍超限，再做更严格的一轮（单字符串上限 = 预算/64，
    单列表上限 = 预算/6000 项）。dict 键绝不被丢弃。
    """
    gentle = _fit(payload, max_chars // 8, max(1, max_chars // 1200))
    if len(json.dumps(gentle, ensure_ascii=False, default=str)) <= max_chars:
        return gentle
    return _fit(payload, max_chars // 64, max(1, max_chars // 6000))
