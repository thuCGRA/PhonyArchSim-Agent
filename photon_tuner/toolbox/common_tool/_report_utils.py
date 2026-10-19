"""Shared report-reading helpers used by the analysis and device tools.

Pulled out of ``analysis_tools.py`` / ``device_tools.py`` so both toolboxes
read SimPhony reports through one place, instead of a device tool importing a
private helper (``_load_report``) from an analysis module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from photon_tuner.common.io import read_json
from photon_tuner.core.schemas import EvaluationResult
from photon_tuner.common.utils import layer_group
from photon_tuner.core.objective import metrics_with_objective
from photon_tuner.toolbox.common_tool.context import ToolContext


def _load_report(context: ToolContext) -> Dict[str, Any]:
    if context.evaluation is None:
        raise ValueError("No evaluation available in tool context")
    report_path = context.evaluation.detail_files.get("report")
    if report_path is None or not Path(report_path).exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")
    return read_json(report_path)


def _metrics_payload(context: ToolContext, evaluation: EvaluationResult) -> Dict[str, Any]:
    metrics = metrics_with_objective(
        evaluation.metrics, evaluation.hardware, context.search_space
    )
    return {key: round(float(value), 6) for key, value in metrics.items()}


def _active_evaluation(context: ToolContext) -> EvaluationResult:
    """Return the currently selected evaluation base (default: current)."""
    source = context.active_base.get("source", "current")
    if source in context.evaluation_bases:
        return context.evaluation_bases[source]
    return context.evaluation or next(iter(context.evaluation_bases.values()))


def _flatten_cycles(partition_cycles: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten {sub_arch: {layer: cycle}} into {layer: cycle}."""
    flattened: Dict[str, Any] = {}
    for sub_arch, layers in partition_cycles.items():
        if isinstance(layers, dict):
            flattened.update(layers)
    return flattened


def _flatten_memory(latency: Dict[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for sub_arch, layers in latency.items():
        if isinstance(layers, dict):
            flattened.update(layers)
    return flattened


def _flatten_memory_energy(energy: Dict[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for sub_arch, layers in energy.items():
        if isinstance(layers, dict):
            for layer, value in layers.items():
                flattened[layer] = value
    return flattened


def _rank_layers(report: Dict[str, Any], context: ToolContext) -> List[Dict[str, Any]]:
    """Rank layers by cost, preferring the simulator's *native* per-layer energy.

    The analytic mirror reports ``energy.by_layer`` (captured while it already
    iterates the layers).  When present it is used directly.  Older reports and
    the real ONN simulator -- which only breaks energy down per device -- fall
    back to a cycles-proportional estimate, flagged via ``energy_source``.
    """
    cycles = _flatten_cycles(report.get("workload", {}).get("partition_cycles", {}))
    memory_latency = _flatten_memory(report.get("memory", {}).get("latency", {}))
    memory_energy = _flatten_memory_energy(report.get("memory", {}).get("energy", {}))
    native_energy = _flatten_memory_energy(report.get("energy", {}).get("by_layer", {}))

    metrics = report.get("metrics", {})
    total_energy = float(metrics.get("energy_pj", 1.0))
    total_area = float(metrics.get("area_um2", 1.0))
    # Work frequency (GHz) is needed for the per-layer latency estimate; read
    # it from the evaluated hardware so step-frequency moves are reflected.
    freq = _work_freq_g_hz(report, context)

    switching_cycles = {name: _cycle_switching(cycle) for name, cycle in cycles.items()}
    total_switching = sum(switching_cycles.values()) or 1

    ranks = []
    for name, cycle in cycles.items():
        share = switching_cycles[name] / total_switching
        if name in native_energy:
            energy_pj = float(native_energy[name])
            energy_source = "simulated"
        else:
            energy_pj = total_energy * share
            energy_source = "estimated"
        ranks.append({
            "layer_name": name,
            "group": layer_group(name),
            "cycles": cycle,
            "switching_cycles": switching_cycles[name],
            "latency_s": switching_cycles[name] / freq * 1e-9,
            "memory_latency_s": _memory_latency_value(memory_latency.get(name)),
            "energy_pj": energy_pj,
            "energy_source": energy_source,
            # Backward-compatible alias for earlier tool consumers.
            "energy_share_pj": energy_pj,
            "area_share_um2": total_area * share,
            "memory_energy_pj": memory_energy.get(name, 0.0),
        })
    ranks.sort(key=lambda item: item["energy_pj"], reverse=True)
    for index, item in enumerate(ranks):
        item["layer_id"] = index
    return ranks


def _work_freq_g_hz(report: Dict[str, Any], context: ToolContext) -> float:
    """Read the crossbar clock frequency (GHz) from hardware, with fallbacks."""
    candidates = [
        (context.current_hardware or {}).get("arch", {}).get("optical_core", {}).get("work_freq"),
        (report.get("hardware") or {}).get("arch", {}).get("optical_core", {}).get("work_freq"),
    ]
    for value in candidates:
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 5.0


def _cycle_switching(cycle: Any) -> float:
    """iter_N * iter_D * iter_M from a SimPhony cycle tuple."""
    if isinstance(cycle, (list, tuple)) and len(cycle) >= 3:
        try:
            return float(cycle[0]) * float(cycle[1]) * float(cycle[2])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _memory_latency_value(value: Any) -> float:
    if isinstance(value, dict):
        return float(value.get("input_latency_s", value.get("total", 0.0)) or 0.0)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _flatten_sub_archs(by_device: Dict[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for sub_arch, devices in by_device.items():
        if isinstance(devices, dict):
            flattened.update(devices)
    return flattened


def _summarize_device_records(by_device: Dict[str, Any], metric: str) -> List[Dict[str, Any]]:
    rows = []
    for device, record in by_device.items():
        if not isinstance(record, dict):
            continue
        rows.append({
            "device": device,
            metric: record.get(metric, 0.0),
            "count": record.get("count"),
            "chip_type": record.get("chip_type"),
        })
    rows.sort(key=lambda row: float(row[metric] or 0.0), reverse=True)
    return rows
