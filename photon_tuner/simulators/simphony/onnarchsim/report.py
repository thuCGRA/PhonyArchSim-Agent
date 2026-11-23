"""Fine-grained, machine-readable report generation for SimPhony.

SimPhony's original :class:`~onnarchsim.simulator.ONNArchSimulator` emits its
results one workflow at a time through ``log_report``, which writes a
human-readable YAML-ish text log.  That is hard to consume programmatically by
an LLM-driven design-space-exploration loop.

This module aggregates per-device, per-chip and per-layer results from the
``simu_*`` methods and adds the separately reported digital reduction cost for
grouped MRR weight banks.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from .digital_reduction import estimate_weight_bank_reduction

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
        - ``energy``: per-device / per-chip-type **compute** breakdowns plus totals
          (pJ). Storage energy lives in ``memory.energy``; ``metrics.energy_mj``
          is the sum of the two.
        - ``area``: per-device and per-chip-type breakdowns plus totals (um^2).
        - ``latency``: computation, memory, and total latency (s).
        - ``memory``: per-layer memory latency/energy and the memory spec.
        - ``metrics``: scalar summary used by the DSE objective.

    Grouped MRR weight banks additionally expose the digital reduction model,
    its per-layer costs and assumptions in ``digital_reduction``.
    """

    partition_cycles = sim.simu_partition_cycles(sim.layer_workloads, sim.layer_sizes)

    insertion_loss = sim.simu_insertion_loss()

    energy_cost, total_energy, computation_latency = sim.simu_energy(
        partition_cycles, insertion_loss
    )
    area_cost, vendor_total_area = sim.simu_area()
    digital_reduction = _apply_weight_bank_reduction_costs(
        sim, partition_cycles, energy_cost, total_energy, computation_latency, area_cost,
    )
    chip_energy = sim.simu_chip_energy(energy_cost)
    composition = _area_composition(area_cost)
    # 厂商 ``simu_area`` 的求和条件写成了 ``if "node" not in v``——判断的是**字典
    # 内部是否含名为 "node" 的键**，而不是键名是否含 node。因此节点级器件既被
    # 计入逐器件求和，随后又把节点边界框加了一遍：真实模式下面积被多算约 43%
    # （实测 111.12 vs 77.75 mm²）。这里按解析镜像的口径重算：逐器件面积 +
    # 节点边界框（节点内部器件已包含在边界框内）。原始值保留在
    # ``area.vendor_total_um2`` 里以便追溯，不静默丢弃。
    corrected_area = {
        arch: rec["device_area_um2"] + rec["node_bbox_um2"]
        for arch, rec in composition.items()
    } if composition else vendor_total_area
    total_area = corrected_area

    memory_latency, memory_energy, memory_spec = sim.simu_memory_cost(partition_cycles)

    total_latency = sim.simu_latency(memory_latency, computation_latency)

    # Scalar summary metrics.  SimPhony reports per sub-architecture; flatten
    # those per-sub-architecture scalars into whole-design totals for the DSE
    # objective while keeping the structured breakdowns above.
    #
    # NOTE ON UNITS: the ``energy`` / ``area`` / ``latency`` blocks above stay in
    # SimPhony's native SI-ish units (pJ, um^2, s) because they are the
    # simulator's own per-device outputs and must remain directly traceable to
    # its physics. The ``metrics`` block below is the project-level canonical
    # summary and is emitted in engineering units (ms / mJ / mm^2) so that
    # downstream consumers never juggle two conventions; the ``units`` block
    # declares exactly which block uses which unit.
    # ``simu_energy`` 只覆盖**计算**能耗（光子/电器件）；存储能耗由
    # ``simu_memory_cost`` 单独给出。``simu_latency`` 却把两者都算进延时。
    # 若目标函数只取计算能耗，就会出现"计时算存储、计价不算存储"的口径
    # 不一致——实测存储占总能耗 75–90%，这会让优化方向系统性偏移。
    # 因此这里显式合成总能耗，并把两个分量都暴露出来供审计与敏感性分析。
    compute_pj = _sum_scalar_map(total_energy)
    memory_pj = _sum_numeric_leaves(memory_energy)
    energy_pj = compute_pj + memory_pj

    area_um2 = _sum_scalar_map(total_area)
    latency_s = _sum_scalar_map(total_latency)

    metrics = {
        "latency_ms": latency_s * 1e3,
        "energy_mj": energy_pj * 1e-9,
        "energy_compute_mj": compute_pj * 1e-9,
        "energy_memory_mj": memory_pj * 1e-9,
        "area_mm2": area_um2 * 1e-6,
    }
    # SI 镜像：保留换算前的原始量，使适配器可以继续在 SI 下做派生推导
    # （每 MAC 能耗、每吞吐面积），且任何人可核对"规范值 = SI 值 × 系数"。
    si_metrics = {
        "latency_s": latency_s,
        "energy_pj": energy_pj,
        "energy_compute_pj": compute_pj,
        "energy_memory_pj": memory_pj,
        "area_um2": area_um2,
    }

    report = {
        "metadata": {
            "simulator": "SimPhony",
            "params": json_safe(getattr(sim, "params", {})),
        },
        "units": {
            "metrics": {
                "latency_ms": "ms",
                "energy_mj": "mJ (compute + storage)",
                "energy_compute_mj": "mJ (photonic/electronic devices only)",
                "energy_memory_mj": "mJ (HBM/GLB/RF)",
                "area_mm2": "mm^2",
            },
            "insertion_loss": "dB",
            "energy": "pJ",
            "area": "um^2",
            "latency": "s",
            "memory": "pJ (energy) / s (latency)",
            "note": "Only the `metrics` block is in engineering units; the "
                    "per-device blocks keep SimPhony's native units so they stay "
                    "traceable to its physical model. `energy` covers compute "
                    "devices only — storage energy is `memory.energy`; the DSE "
                    "objective uses their sum.",
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
            # 用**自建**的按 chip 分组替换厂商的 ``chip_area_calculator`` 结果：
            # 后者按设备名过滤，会把节点级器件（移相器/耦合器/探测器）以及节点
            # 边界框排除在外，导致各 chip 求和比 total 少约 30%。消费方按 chip
            # 求和会得到与目标函数不一致的面积，且没有任何提示。
            "by_chip_type": json_safe(_complete_chip_area(area_cost)),
            "composition": json_safe(composition),
            "total": json_safe(total_area),
            "vendor_total_um2": json_safe(vendor_total_area),
            "total_note": "total = composition.device_area_um2 + composition.node_bbox_um2 "
                          "(the vendor total double counted node devices and the node "
                          "bounding box; see area.vendor_total_um2 for the raw value).",
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
        "si_metrics": si_metrics,
    }
    if digital_reduction:
        report["digital_reduction"] = json_safe(digital_reduction)
    return report


def _apply_weight_bank_reduction_costs(
    sim: Any,
    partition_cycles: Dict[str, Any],
    energy_cost: Dict[str, Any],
    total_energy: Dict[str, Any],
    computation_latency: Dict[str, Any],
    area_cost: Dict[str, Any],
) -> Dict[str, Any]:
    costs = {}
    for arch, config in getattr(sim, "sub_arch_dictionary", {}).items():
        if config.get("core", {}).get("weight_bank_groups", 1) == 1:
            continue
        reduction = estimate_weight_bank_reduction(config, partition_cycles[arch])
        if reduction is None:
            continue
        costs[arch] = reduction
        key = "core_digital_adder_0_groupreduce-Linear_estimate"
        energy_cost[arch][key] = {
            "count": reduction["adder_count"],
            "chip_type": "RF_EIC",
            "total_dynamic_energy": reduction["energy_pj"],
            "total_static_energy": 0.0,
            "total_energy": reduction["energy_pj"],
        }
        total_energy[arch] += reduction["energy_pj"]
        computation_latency[arch]["digital_reduction_latency_s"] = reduction["latency_s"]
        area_cost[arch][key] = {
            "count": reduction["adder_count"],
            "chip_type": "RF_EIC",
            "total_area": reduction["area_um2"],
        }
    return costs


def _is_scalar_map(value: Any) -> bool:
    """True when every value of a dict is a scalar (flat sub-arch -> number)."""
    if not isinstance(value, dict):
        return False
    return all(not isinstance(v, dict) for v in value.values())


def _sum_scalar_map(value: Any) -> float:
    if isinstance(value, dict):
        return sum(float(v) for v in value.values() if _is_number(v))
    return float(value) if _is_number(value) else 0.0


def _is_node_record(key: str) -> bool:
    """厂商 ``simu_area`` 的约定：键名含 ``node`` 的条目不计入逐器件求和。"""
    return "node" in str(key)


def _complete_chip_area(area_cost: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """按 chip_type 分组面积，且**保证求和等于 total**。

    厂商的 ``chip_area_calculator`` 会 ``continue`` 掉名字含 ``node`` 且属于
    off-chip 的条目，于是节点级器件与节点边界框从 chip 视图里消失。这里沿用
    ``simu_area`` 自身的口径（非 node 条目求和 + node 边界框一次），把边界框
    作为独立 chip 组 ``node_bbox`` 显式列出，使任何一个 chip 的分组都能加总到
    ``area.total``。
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    for sub_arch, records in (area_cost or {}).items():
        buckets: Dict[str, Dict[str, Any]] = {}
        for key, record in (records or {}).items():
            if not isinstance(record, dict):
                continue
            chip_type = str(record.get("chip_type", "off_chip"))
            if _is_node_record(key):
                # 节点边界框单独成组；节点内的逐器件面积已包含在边界框里，
                # 单独再加会重复计算。
                if key == "node":
                    bucket = buckets.setdefault("node_bbox", {"devices": [], "total_area": 0.0})
                    bucket["devices"].append(key)
                    bucket["total_area"] += float(record.get("total_area", 0.0) or 0.0)
                continue
            bucket = buckets.setdefault(chip_type, {"devices": [], "total_area": 0.0})
            bucket["devices"].append(key)
            bucket["total_area"] += float(record.get("total_area", 0.0) or 0.0)
        grouped[sub_arch] = buckets
    return grouped


def _area_composition(area_cost: Dict[str, Any]) -> Dict[str, Any]:
    """显式拆出面积构成：逐器件面积 + 节点边界框。

    没有这个块，``by_chip_type`` 与 ``total`` 的差额就是"沉默的 30%"。
    """
    composition: Dict[str, Any] = {}
    for sub_arch, records in (area_cost or {}).items():
        device_area = 0.0
        node_bbox = 0.0
        for key, record in (records or {}).items():
            if not isinstance(record, dict):
                continue
            total = float(record.get("total_area", 0.0) or 0.0)
            if key == "node":
                node_bbox += total
            elif _is_node_record(key):
                continue  # 已包含在节点边界框内
            else:
                device_area += total
        composition[sub_arch] = {
            "device_area_um2": device_area,
            "node_bbox_um2": node_bbox,
            "total_um2": device_area + node_bbox,
            "note": "device_area covers core/PIC devices from by_device; node_bbox is "
                    "the node physical footprint (its per-device breakdown is inside it, "
                    "so do not add both).",
        }
    return composition


def _sum_numeric_leaves(value: Any) -> float:
    """递归求和任意嵌套字典里的数值叶子（用于 ``{arch: {level: pJ}}`` 形状）。"""
    if isinstance(value, dict):
        return sum(_sum_numeric_leaves(v) for v in value.values())
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
