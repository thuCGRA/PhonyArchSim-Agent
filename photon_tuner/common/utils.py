"""Dependency-free helper functions shared across the photonic tuner."""

from __future__ import annotations

import re
from typing import Any, Dict, Sequence


def clamp_to_candidates(value: int, candidates: Sequence[int], direction: int) -> int:
    """Return the candidate nearest ``value``, or the adjacent one in ``direction``.

    When ``value`` is not itself a legal candidate, the closest candidate is
    returned.  Otherwise the candidate immediately before (``direction=-1``) or
    after (``direction=+1``) is returned, clamped so the result never leaves the
    sorted candidate list.
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
    """Map a photonic / quantised-NN layer name to a coarse operator group.

    Order matters here: ``"qkv"`` must be tested *before* the attention tokens,
    because ``"qkv"`` itself contains ``"qk"``.  In the original ordering every
    fused QKV projection was reported as ``attention`` and the
    ``qkv_projection`` branch was unreachable.
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
    """Extract lightweight structural features from a layer name."""
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
