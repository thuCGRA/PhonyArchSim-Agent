"""Deterministic workload feature extraction (NOT LLM-based).

These structural features are computed once per workload and threaded through
every ablation record as the cross-cutting condition.  Keeping them
deterministic is what makes ``null`` results trustworthy and reproducible.
"""

from __future__ import annotations

from typing import Any, Dict

_ATTENTION_TOKENS = ("attn", "qkv", "quantized_matmul", "proj.linear")
_FFN_TOKENS = ("linear1", "linear2", "ffn", "mlp")
_EMBED_TOKENS = ("embed", "conv", "patch", "pos")


def workload_features(workload_name: str, report: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structural workload features from a SimPhony report.

    ``report["workload"]`` contains ``layer_sizes`` and ``partition_cycles``
    (per sub-arch ``LT``).  We reduce them to a small, comparable fingerprint.
    """
    wl = report.get("workload") if isinstance(report.get("workload"), dict) else {}
    sizes = wl.get("layer_sizes") or {}
    cycles_by_arch = wl.get("partition_cycles") or {}
    cycles = cycles_by_arch.get("LT") if isinstance(cycles_by_arch, dict) else {}

    layer_names = list(sizes.keys()) if isinstance(sizes, dict) else []

    groups = {"attention": 0, "ffn": 0, "embed": 0, "other": 0}
    for name in layer_names:
        n = str(name).lower()
        if any(t in n for t in _ATTENTION_TOKENS):
            groups["attention"] += 1
        elif any(t in n for t in _FFN_TOKENS):
            groups["ffn"] += 1
        elif any(t in n for t in _EMBED_TOKENS):
            groups["embed"] += 1
        else:
            groups["other"] += 1

    total_cycles = 0
    for c in cycles.values():
        if isinstance(c, (list, tuple)):
            total_cycles += sum(float(x) for x in c if isinstance(x, (int, float)))

    total_size = 0
    for s in sizes.values():
        if isinstance(s, (list, tuple)):
            total_size += sum(float(x) for x in s if isinstance(x, (int, float)))
        elif isinstance(s, (int, float)):
            total_size += float(s)

    return {
        "name": workload_name,
        "layer_count": len(layer_names),
        "total_cycles": total_cycles,
        "total_size_bytes": total_size,
        "operator_groups": groups,
        "layer_names": layer_names[:64],  # capped fingerprint for matching
    }
