"""向拓扑代理暴露电路结构拓扑的 LangChain 工具。"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.tools import BaseTool, StructuredTool

from photon_tuner.core.topology_space import (
    inspect_topology_template,
    list_topology_templates,
    param_scope_table,
    physical_violations,
)
from photon_tuner.core.topology_spec import TopologyProvider
from photon_tuner.toolbox.bindings import RunState


def build_topology_tools(
    provider: TopologyProvider,
    run_state: RunState,
    design_space: Optional[object] = None,
    mode: str = "real",
) -> List[BaseTool]:
    def _graph() -> dict:
        return provider.topology(run_state.context.current_hardware).to_dict()

    def _summarize_topology() -> dict:
        """完整电路结构：所有组件及有序光路。"""
        return _graph()

    def _inspect_light_path() -> dict:
        """有序光路（光传播顺序）及所选变体。"""
        g = provider.topology(run_state.context.current_hardware)
        return {
            "source": g.source,
            "optical_path": g.optical_path,
            "path_components": [
                {
                    "device_type": n.device_type,
                    "device_name": n.device_name,
                    "level": n.level,
                }
                for n in g.nodes
                if n.on_optical_path
            ],
        }

    def _list_templates() -> dict:
        """合法电路结构模板 (core_kind/node_kind) + 耦合矩阵。"""
        return list_topology_templates(mode)

    def _inspect_template(name: str) -> dict:
        """按名称查看单个模板（core_kind 或 node_kind）的详情。"""
        try:
            return inspect_topology_template(name)
        except ValueError as exc:
            # 将可读的错误返回给代理，而不是让 ValueError 使整个 LangGraph
            # 工具循环崩溃。
            return {"error": str(exc)}

    def _validate_coupling() -> dict:
        """依据跨层耦合矩阵与拓扑物理强制值校验当前硬件。

        除"这个 (core, node) 组合能否装配"之外，还要回答"这个设计点里有没有物理上
        不成立的取值"——例如网格核心却声明 ``dataflow=output_stationary`` 或
        ``num_wavelength=12``。装配器会忽略这些声明并按拓扑写死，因此不在此拦下
        就会得到一个"从未按该配置仿真过"的设计点。
        """
        violations = physical_violations(
            run_state.context.current_hardware, design_space=design_space, mode=mode
        )
        return {
            "consistent": not violations,
            "violations": violations,
            "note": "scope='forced_value' means the declared value is physically "
                    "overridden by the topology; normalize() rewrites it to the "
                    "effective value.",
        }

    def _searchable_parameters() -> dict:
        """当前拓扑下真正生效 / 已被禁用 / 可实现的参数缺口。"""
        hw = run_state.context.current_hardware or {}
        core_kind = ((hw.get("arch") or {}).get("topology") or {}).get("core_kind")
        table = param_scope_table(core_kind)
        params = table["params"]
        return {
            "core_kind": table["core_kind"],
            "family": table["family"],
            "family_label": table["family_label"],
            "searchable": sorted(p for p, s in params.items() if s["status"] == "effective"),
            "disabled": {
                p: {"status": s["status"], "by": s.get("by"),
                    "reason": s.get("reason"), "action": s.get("action")}
                for p, s in params.items()
                if s["status"] != "effective"
            },
            "note": "Adjusting a disabled parameter has no effect on the simulation "
                    "result; the gate will reject it. Follow `action` instead.",
        }

    summarize = StructuredTool.from_function(
        func=_summarize_topology,
        name="summarize_topology",
        description="Inferred circuit-structure topology: every component plus the ordered optical path (laser -> ... -> photodetector).",
    )
    inspect = StructuredTool.from_function(
        func=_inspect_light_path,
        name="inspect_light_path",
        description="The ordered optical light path with the selected device variant at each position.",
    )
    list_templates = StructuredTool.from_function(
        func=_list_templates,
        name="list_topology_templates",
        description="Legal circuit-structure templates (core_kind/node_kind) plus the core->node coupling matrix and which templates are evaluable today.",
    )
    inspect_template = StructuredTool.from_function(
        func=_inspect_template,
        name="inspect_topology_template",
        description="Detail one topology template by name (a core_kind like 'dptc_crossbar' or a node_kind like 'dode'): metadata, compatible kinds, required device slots.",
    )
    validate = StructuredTool.from_function(
        func=_validate_coupling,
        name="validate_coupling",
        description="Validate the current hardware design point: cross-layer coupling matrix (kind "
                    "compatibility, device slots, work_freq/precision chain) plus physically forced values "
                    "(values the netlist builder overrides for this topology, reported with scope='forced_value').",
    )
    searchable = StructuredTool.from_function(
        func=_searchable_parameters,
        name="searchable_parameters",
        description="Which search parameters actually affect the simulation under the current topology. "
                    "Returns `searchable` (use these), `disabled` (changing these has no effect — each "
                    "carries a reason and an actionable `action`), and the physical rationale.",
    )
    return [summarize, inspect, list_templates, inspect_template, validate, searchable]
