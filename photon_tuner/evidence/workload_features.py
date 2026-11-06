"""确定性的工作负载特征提取（不基于 LLM）。

这些结构化特征按每个工作负载计算一次，并作为横切条件贯穿每条消融记录。
保持其确定性使 ``null`` 结果可信且可复现。
"""

from __future__ import annotations

from typing import Any, Dict

from photon_tuner.simulators.netlist import first_key

_ATTENTION_TOKENS = ("attn", "qkv", "quantized_matmul", "proj.linear")
_FFN_TOKENS = ("linear1", "linear2", "ffn", "mlp")
_EMBED_TOKENS = ("embed", "conv", "patch", "pos")


def workload_features(workload_name: str, report: Dict[str, Any]) -> Dict[str, Any]:
    """从 SimPhony 报告中提取结构化工作负载特征。

    ``report["workload"]`` 包含 ``layer_sizes`` 和 ``partition_cycles``
    （按子架构，键为 ``core_kind``）。将它们归约为一组小而可比较的特征指纹。
    """
    wl = report.get("workload") if isinstance(report.get("workload"), dict) else {}
    sizes = wl.get("layer_sizes") or {}
    cycles_by_arch = wl.get("partition_cycles") or {}
    cycles = (
        cycles_by_arch.get(first_key(cycles_by_arch))
        if isinstance(cycles_by_arch, dict)
        else {}
    ) or {}

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
        "layer_names": layer_names[:64],  # 用于匹配的截断特征指纹
    }
