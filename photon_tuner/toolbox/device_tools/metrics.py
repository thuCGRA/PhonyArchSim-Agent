"""器件工具箱的逐器件汇总读取工具。

使用者：optical_level。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from photon_tuner.common.tool_types import ToolResult, ToolSpec
from photon_tuner.core.search_space import hardware_summary, materialize_hardware
from photon_tuner.simulators.netlist import first_key
from photon_tuner.toolbox.common_tool._report_utils import (
    _flatten_sub_archs,
    _load_report,
    _summarize_device_records,
)
from photon_tuner.toolbox.common_tool.context import ToolContext


def summarize_device_energy_breakdown(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    report = _load_report(context)
    by_device = _flatten_sub_archs(report.get("energy", {}).get("by_device", {}))
    summary = _summarize_device_records(by_device, metric="total_energy")
    return ToolResult("summarize_device_energy_breakdown", {"devices": summary})


def summarize_device_area_breakdown(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    report = _load_report(context)
    by_device = _flatten_sub_archs(report.get("area", {}).get("by_device", {}))
    summary = _summarize_device_records(by_device, metric="total_area")
    return ToolResult("summarize_device_area_breakdown", {"devices": summary})


def summarize_optical_structure(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    hardware = materialize_hardware(context.current_hardware)
    summary = hardware_summary(hardware)
    # 保留几何/拓扑块，并同时呈现*实际使用的器件变体*。没有这些信息，代理
    # 无法验证如激光器是在片还是离片等问题。
    structure = dict(summary.get("arch", {}))
    for key in ("core_devices", "node_devices", "mesh_core_devices", "mesh_node_devices"):
        table = summary.get(key) or {}
        if table:
            structure[key] = table
    return ToolResult("summarize_optical_structure", {"optical_structure": structure})


def _device_lookup(report: Dict[str, Any]) -> Dict[Tuple[str, str], Tuple[Optional[str], Optional[str]]]:
    """从报告映射网表实例 id -> (device_type, device_name)。

    真实仿真器将其 ``by_device`` 表键为
    ``<level>_<device_type>_<index>_<instance>-<DeviceName>``（例如
    ``core_off_chip_laser_0_i_laser-LT_Laser``），这足以在不重建网表的情况下
    为插入损耗贡献背后的物理组件命名。
    """
    lookup: Dict[Tuple[str, str], Tuple[Optional[str], Optional[str]]] = {}
    tables = [report.get("energy", {}).get("by_device", {}),
              report.get("area", {}).get("by_device", {})]
    for table in tables:
        for sub in (table or {}).values():
            for key in (sub or {}):
                # "<level>_<device_type>_<index>_<instance>-<DeviceName>"，其中
                # device_type 本身可能包含下划线（如 off_chip_laser）。
                head, _, dev_name = str(key).partition("-")
                parts = head.split("_")
                idx_pos = next(
                    (i for i in range(1, len(parts)) if parts[i].isdigit()), None
                )
                if idx_pos is None or idx_pos + 1 >= len(parts):
                    continue
                if idx_pos < 2:
                    continue
                level = parts[0]
                device_type = "_".join(parts[1:idx_pos])
                instance = "_".join(parts[idx_pos + 1:])
                # 节点与核心网表复用实例名 (i0..i3)，因此按 (level, instance)
                # 作为键以避免跨作用域冲突。
                lookup.setdefault((level, instance), (device_type, dev_name or None))
    return lookup


def _normalize_scope(il: Dict[str, Any], lookup: Dict[Tuple[str, str], Tuple[Optional[str], Optional[str]]]) -> None:
    node_il = il.get("node_insertion_loss")

    def _resolve(scope: str, instance: str, loss: float) -> Tuple[Optional[str], Optional[str]]:
        found = lookup.get((scope, str(instance)))
        if found:
            return found
        # 核心网表将整个节点子电路折叠为一个伪实例（device_type "node"）；其
        # 贡献是节点路径损耗的整数倍（16 级 MZI -> 16 x 节点 IL）。按与分析
        # 镜像相同的方式命名。
        if node_il and float(loss) > 0:
            ratio = float(loss) / float(node_il)
            if abs(ratio - round(ratio)) < 1e-6 and round(ratio) >= 1:
                return ("node", "node_subcircuit")
        return (None, None)

    for scope in ("core", "node"):
        detail_key = f"{scope}_path_detail"
        comp_key = f"{scope}_component_loss"
        if il.get(detail_key):
            continue
        comp = il.get(comp_key)
        if not isinstance(comp, dict) or not comp:
            continue
        il[detail_key] = [
            {
                "instance": inst,
                "device_type": _resolve(scope, inst, loss)[0],
                "device_name": _resolve(scope, inst, loss)[1],
                "insertion_loss_dB": float(loss or 0.0),
            }
            for inst, loss in sorted(comp.items(), key=lambda kv: -float(kv[1] or 0.0))
        ]


def _normalize_il_detail(il: Dict[str, Any], report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """将两种插入损耗报告形状归一化为一种契约。

    分析镜像输出 ``*_path_detail``（由 ``_path_detail`` 构建的逐组件字典列表）。
    真实 ONNArchSimulator 输出 ``*_component_loss``（``instance -> dB`` 映射）。
    代理无需关心证据来自哪个仿真器，因此将真实映射展开为分析列表形状（按贡献
    降序排列）。报告按子架构名为键（该键即 ``core_kind``，例如
    ``"dptc_crossbar"``），因此向下深入一层。
    """
    lookup = _device_lookup(report) if report else {}
    for scope, value in il.items():
        if isinstance(value, dict):
            _normalize_scope(value, lookup)
    return il


def summarize_insertion_loss_detail(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    report = _load_report(context)
    il = _normalize_il_detail(dict(report.get("insertion_loss", {}) or {}), report)
    return ToolResult("summarize_insertion_loss_detail", {"insertion_loss": il})


def summarize_topology_evidence(toolbox: Any, context: ToolContext, arguments: Dict[str, Any]) -> ToolResult:
    """拓扑阶段损耗/功率任务的一体化证据包。

    拓扑代理需要逐组件损耗归属、实际使用的器件变体以及激光器/PD 电平来计算
    链路裕量。本函数将其打包，且 ``power_budget_inputs`` 块携带已解析好的
    数值，代理只需做算术计算。
    """
    hardware = materialize_hardware(context.current_hardware)
    structure = summarize_optical_structure(toolbox, context, arguments).payload["optical_structure"]
    report = _load_report(context)
    il = _normalize_il_detail(dict(report.get("insertion_loss", {}) or {}), report)
    light = il.get(first_key(il), il) if isinstance(il, dict) else {}

    def _variant(level_key: str, device_type: str) -> Optional[str]:
        table = structure.get(level_key) or {}
        return table.get(device_type)

    laser_name = _variant("mesh_core_devices", "off_chip_laser") or _variant(
        "mesh_core_devices", "on_chip_laser"
    ) or _variant("core_devices", "off_chip_laser") or _variant("core_devices", "on_chip_laser")
    laser_type = (
        "off_chip_laser"
        if _variant("mesh_core_devices", "off_chip_laser") or _variant("core_devices", "off_chip_laser")
        else "on_chip_laser"
    )
    pd_name = _variant("mesh_core_devices", "photodetector") or _variant(
        "core_devices", "photodetector"
    )

    inputs: Dict[str, Any] = {}
    for label, dtype, dname in (("laser", laser_type, laser_name), ("photodetector", "photodetector", pd_name)):
        if not dname:
            continue
        body = (getattr(toolbox, "catalog", {}) or {}).get(dtype, {}).get(dname)
        if body:
            inputs[label] = {
                "device_type": dtype,
                "device_name": dname,
                "cfgs": dict(body.get("cfgs", {}) or {}),
            }

    return ToolResult("summarize_topology_evidence", {
        "optical_structure": structure,
        "insertion_loss": light,
        "core_component_loss": light.get("core_component_loss", {}),
        "node_component_loss": light.get("node_component_loss", {}),
        "core_path_detail": light.get("core_path_detail", []),
        "node_path_detail": light.get("node_path_detail", []),
        "power_budget_inputs": inputs,
        "note": (
            "core_insertion_loss / node_insertion_loss are totals; the *_path_detail "
            "entries attribute them per component. Compute "
            "link_margin_db = laser_output_dbm - (core_insertion_loss + node_insertion_loss)"
            " - pd_sensitivity_dbm (sensitivity is negative)."
        ),
    })


TOOLS: Dict[str, Tuple[ToolSpec, Callable]] = {
    "summarize_topology_evidence": (
        ToolSpec(
            "summarize_topology_evidence",
            "ONE-CALL topology evidence bundle: optical structure (incl. the device "
            "variants actually in use), the insertion-loss totals, the per-component "
            "core_component_loss / node_component_loss maps and core_path_detail / "
            "node_path_detail lists, plus power_budget_inputs carrying the laser "
            "(power, dBm) and photodetector (sensitivity, dBm) cfgs. Prefer this over "
            "several separate calls: it is all the loss and power-budget evidence the "
            "topology diagnosis needs.",
            {"type": "object", "properties": {}},
        ),
        summarize_topology_evidence,
    ),
    "summarize_device_energy_breakdown": (
        ToolSpec(
            "summarize_device_energy_breakdown",
            "Per-device energy contribution (pJ), ranked high to low. Each row carries "
            "level (core/node), device_type, device_name, instance, count, chip_type and "
            "total_energy_share (fraction of the total), so the dominant energy consumer "
            "is directly visible.",
            {"type": "object", "properties": {}},
        ),
        summarize_device_energy_breakdown,
    ),
    "summarize_device_area_breakdown": (
        ToolSpec(
            "summarize_device_area_breakdown",
            "Per-device area contribution (um^2), ranked high to low. Each row carries "
            "level (core/node), device_type, device_name, instance, count, chip_type and "
            "total_area_share (fraction of the total).",
            {"type": "object", "properties": {}},
        ),
        summarize_device_area_breakdown,
    ),
    "summarize_optical_structure": (
        ToolSpec(
            "summarize_optical_structure",
            "Full circuit-organisation snapshot: core_kind, node_kind, sharing_policy, "
            "temporal_accum, laser_distribution, plus geometry/wavelength/clock/precision/"
            "dataflow. Also reports the device variants actually in use (core_devices / "
            "node_devices / mesh_core_devices / mesh_node_devices), e.g. which laser "
            "(off_chip_laser vs on_chip_laser) and phase shifter are selected.",
            {"type": "object", "properties": {}},
        ),
        summarize_optical_structure,
    ),
    "summarize_insertion_loss_detail": (
        ToolSpec(
            "summarize_insertion_loss_detail",
            "Node/core longest optical paths with per-component insertion loss. "
            "Returns node_path/core_path (the instance chain) plus node_path_detail/"
            "core_path_detail: one {instance, device_type, device_name, "
            "insertion_loss_dB} entry per component along that path, so you can "
            "attribute which component loses how many dB (and which dominates).",
            {"type": "object", "properties": {}},
        ),
        summarize_insertion_loss_detail,
    ),
}
