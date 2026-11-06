"""光子调优器共享的、无外部依赖的辅助函数。"""

from __future__ import annotations

import re
from typing import Any, Dict, Sequence


def clamp_to_candidates(value: int, candidates: Sequence[int], direction: int) -> int:
    """返回最接近 ``value`` 的候选值，或沿 ``direction`` 方向的相邻候选值。

    当 ``value`` 本身不是合法候选时，返回最接近的候选；否则返回其前一个
    （``direction=-1``）或后一个（``direction=+1``）候选，并钳制结果使其
    始终不超出排序后的候选列表。
    """
    legal = sorted({int(candidate) for candidate in candidates})
    if not legal:
        raise ValueError("clamp_to_candidates requires at least one candidate")
    try:
        position = legal.index(value)
    except ValueError:
        return min(legal, key=lambda candidate: abs(candidate - value))
    position = max(0, min(len(legal) - 1, position + direction))
    return legal[position]


def layer_group(name: str) -> str:
    """将光子 / 量化神经网络层名映射为粗粒度的算子组。

    此处顺序很重要：``"qkv"`` 必须先于 attention 相关 token 判断，
    因为 ``"qkv"`` 本身包含 ``"qk"``。
    """
    lowered = name.lower()
    if "qkv" in lowered or any(token in lowered for token in ("_q", "_k", "_v")):
        return "qkv_projection"
    if any(token in lowered for token in ("qk", "att", "attention")):
        return "attention"
    if any(token in lowered for token in ("out_proj", "proj")):
        return "output_projection"
    if any(token in lowered for token in ("ffn", "mlp")):
        return "ffn"
    if "patch_embed" in lowered or "embed" in lowered:
        return "embedding"
    if "conv" in lowered:
        return "convolution"
    if "linear" in lowered or "fc" in lowered:
        return "linear"
    if "matmul" in lowered or "mm" in lowered:
        return "matmul"
    if "elt" in lowered or "add" in lowered:
        return "elementwise"
    return "other"


def operator_features(layer_name: str) -> Dict[str, Any]:
    """从层名中提取轻量结构特征。"""
    head = _int_match(layer_name, r"head(\d+)")
    request = _int_match(layer_name, r"req(\d+)")
    tiling = _int_match(layer_name, r"tiling[_-]?(\d+)")
    return {
        "group": layer_group(layer_name),
        "head": head,
        "request": request,
        "tiling": tiling,
        "is_projection": "proj" in layer_name.lower() or "qkv" in layer_name.lower(),
        "is_elementwise": "elt" in layer_name.lower(),
    }


def _int_match(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None
