"""Fine-grained, machine-readable report generation for SimPhony.

SimPhony's original :class:`~onnarchsim.simulator.ONNArchSimulator` emits its
results one workflow at a time through ``log_report``, which writes a
human-readable YAML-ish text log.  That is hard to consume programmatically by
an LLM-driven design-space-exploration loop.

This module adds a *small, self-contained* report layer on top of the existing
``simu_*`` methods.  It does **not** change any of the simulation math; it only
aggregates the already-computed, per-device / per-chip / per-layer results into
a single nested ``dict`` and (optionally) serialises it to JSON so downstream
agents can rely on a stable, fine-grained schema.
"""

from __future__ import annotations

import json
from typing import Any, Dict

__all__ = ["build_report", "write_report_json", "json_safe"]


def json_safe(value: Any) -> Any:
    """Recursively convert a value into JSON-serialisable plain Python types.

    Handles the two custom containers used across SimPhony: ``pyutils.Config``
    (a ``dict`` subclass) and NumPy scalar types.  Any object with a ``dict()``
    method (``Config``, ``DataBase``) is converted first; NumPy scalars fall
    back to ``float()``/``int()``; unknown objects use ``str()`` so a report
    can never fail to serialise.
    """

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        return value
    if hasattr(value, "item") and not isinstance(value, (dict, list, str)):
        # NumPy scalar / 0-d array.
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "dict") and callable(value.dict):
        return json_safe(value.dict())
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def build_report(sim: Any) -> Dict[str, Any]:
    """Run every SimPhony workflow once and aggregate a fine-grained report.

    Parameters
    ----------
    sim:
        An initialised :class:`onnarchsim.simulator.ONNArchSimulator`.

    Returns
    -------
    dict
        A nested report with the following top-level keys:

        - ``metadata``: simulator version and the original run parameters.
        - ``workload``: per-layer partition cycles (``cycles``) and layer sizes.
        - ``insertion_loss``: per-sub-architecture node/core longest paths (dB).
        - ``energy``: per-device and per-chip-type breakdowns plus totals (pJ).
        - ``area``: per-device and per-chip-type breakdowns plus totals (um^2).
        - ``latency``: computation, memory, and total latency (s).
        - ``memory``: per-layer memory latency/energy and the memory spec.
        - ``metrics``: scalar summary used by the DSE objective.

    The report reuses the exact return values of the ``simu_*`` methods, so its
    granularity is identical to SimPhony's native per-device breakdowns while
    being structured for downstream tool calls.
    """

    partition_cycles = sim.simu_partition_cycles(sim.layer_workloads, sim.layer_sizes)

    insertion_loss = sim.simu_insertion_loss()

    energy_cost, total_energy, computation_latency = sim.simu_energy(
        partition_cycles, insertion_loss
    )
    chip_energy = sim.simu_chip_energy(energy_cost)

    area_cost, total_area = sim.simu_area()
    chip_area = sim.simu_chip_area(area_cost)

    memory_latency, memory_energy, memory_spec = sim.simu_memory_cost(partition_cycles)

    total_latency = sim.simu_latency(memory_latency, computation_latency)

    # Scalar summary metrics.  SimPhony reports per sub-architecture; flatten
    # those per-sub-architecture scalars into whole-design totals for the DSE
    # objective while keeping the structured breakdowns above.
    energy_pj = _sum_scalar_map(total_energy)
    area_um2 = _sum_scalar_map(total_area)
    latency_s = _sum_scalar_map(total_latency)

    metrics = {
        "energy_pj": energy_pj,
        "area_um2": area_um2,
        "latency_s": latency_s,
    }

    report = {
        "metadata": {
            "simulator": "SimPhony",
            "params": json_safe(getattr(sim, "params", {})),
        },
        "workload": {
            "partition_cycles": json_safe(partition_cycles),
            "layer_sizes": json_safe(getattr(sim, "layer_sizes", {})),
        },
        "insertion_loss": json_safe(insertion_loss),
        "energy": {
            "by_device": json_safe(energy_cost),
            "by_chip_type": json_safe(chip_energy),
            "total": json_safe(total_energy),
        },
        "area": {
            "by_device": json_safe(area_cost),
            "by_chip_type": json_safe(chip_area),
            "total": json_safe(total_area),
        },
        "latency": {
            "computation": json_safe(computation_latency),
            "memory": json_safe(memory_latency),
            "total": json_safe(total_latency),
        },
        "memory": {
            "latency": json_safe(memory_latency),
            "energy": json_safe(memory_energy),
            "spec": json_safe(memory_spec),
        },
        "metrics": metrics,
    }
    return report


def _is_scalar_map(value: Any) -> bool:
    """True when every value of a dict is a scalar (flat sub-arch -> number)."""
    if not isinstance(value, dict):
        return False
    return all(not isinstance(v, dict) for v in value.values())


def _sum_scalar_map(value: Any) -> float:
    if isinstance(value, dict):
        return sum(float(v) for v in value.values() if _is_number(v))
    return float(value) if _is_number(value) else 0.0


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def write_report_json(report: Dict[str, Any], path: str) -> str:
    """Serialise a report to a JSON file and return the resolved path."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, default=str)
    return path
