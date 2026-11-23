"""全项目确定性工具的统一资源目录。

工具按其操作的资源分类，而不是按 Agent 角色复制。原子技能使用另一套
``diagnosis / decision / verification / learning / scheduling`` 任务分类，两层因而
不会同名重叠。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Tuple

from photon_tuner.toolbox.analysis_tools import _REGISTRY as ANALYSIS_REGISTRY
from photon_tuner.toolbox.device_tools import _REGISTRY as DEVICE_REGISTRY
from photon_tuner.toolbox.support_tools import _REGISTRY as SUPPORT_REGISTRY


class ToolCategory(str, Enum):
    REPORTS = "reports"
    DESIGN_SPACE = "design_space"
    TOPOLOGY = "topology"
    DEVICES = "devices"
    SIMULATION = "simulation"
    STATE_MEMORY = "state_memory"
    VALIDATION = "validation"
    EDITING = "editing"


class ToolAccess(str, Enum):
    READ = "read"
    CONTROL = "control"
    WRITE = "write"


@dataclass(frozen=True)
class ToolDescriptor:
    name: str
    category: ToolCategory
    access: ToolAccess
    source: str
    description: str
    domains: Tuple[str, ...]
    agent_visible: bool

    def to_dict(self) -> Dict[str, object]:
        body = asdict(self)
        body["category"] = self.category.value
        body["access"] = self.access.value
        body["domains"] = list(self.domains)
        return body


TOPOLOGY_TOOL_DESCRIPTIONS: Mapping[str, str] = {
    "summarize_topology": "Return every topology component and the ordered optical path.",
    "inspect_light_path": "Inspect the ordered optical path and selected device variants.",
    "list_topology_templates": "List legal topology templates and their coupling matrix.",
    "inspect_topology_template": "Inspect one topology template and its required slots.",
    "validate_coupling": "Validate cross-layer coupling and physically forced values.",
    "searchable_parameters": "Report effective and disabled parameters for this topology.",
}


_CATEGORY_BY_NAME: Mapping[str, ToolCategory] = {
    "summarize_hardware_config": ToolCategory.REPORTS,
    "summarize_metrics": ToolCategory.REPORTS,
    "summarize_energy_breakdown": ToolCategory.REPORTS,
    "summarize_area_breakdown": ToolCategory.REPORTS,
    "summarize_insertion_loss": ToolCategory.REPORTS,
    "summarize_layer_ranks": ToolCategory.REPORTS,
    "summarize_operator_groups": ToolCategory.REPORTS,
    "inspect_layer_details": ToolCategory.REPORTS,
    "summarize_device_energy_breakdown": ToolCategory.REPORTS,
    "summarize_device_area_breakdown": ToolCategory.REPORTS,
    "summarize_insertion_loss_detail": ToolCategory.REPORTS,
    "summarize_topology_evidence": ToolCategory.REPORTS,
    "summarize_workload_features": ToolCategory.REPORTS,
    "describe_metric_units": ToolCategory.REPORTS,
    "summarize_search_space": ToolCategory.DESIGN_SPACE,
    "searchable_parameters": ToolCategory.DESIGN_SPACE,
    "summarize_optical_structure": ToolCategory.TOPOLOGY,
    "summarize_topology": ToolCategory.TOPOLOGY,
    "inspect_light_path": ToolCategory.TOPOLOGY,
    "list_topology_templates": ToolCategory.TOPOLOGY,
    "inspect_topology_template": ToolCategory.TOPOLOGY,
    "list_device_candidates": ToolCategory.DEVICES,
    "inspect_device_config": ToolCategory.DEVICES,
    "summarize_device_frontier": ToolCategory.DEVICES,
    "summarize_device_provenance": ToolCategory.DEVICES,
    "preview_evaluation_schema": ToolCategory.SIMULATION,
    "evaluate_design": ToolCategory.SIMULATION,
    "compare_search_states": ToolCategory.STATE_MEMORY,
    "select_analysis_base": ToolCategory.STATE_MEMORY,
    "summarize_design_identity": ToolCategory.STATE_MEMORY,
    "retrieve_similar_cases": ToolCategory.STATE_MEMORY,
    "summarize_case_memory": ToolCategory.STATE_MEMORY,
    "commit_verified_case": ToolCategory.STATE_MEMORY,
    "summarize_pareto_frontier": ToolCategory.STATE_MEMORY,
    "summarize_schedule_signals": ToolCategory.STATE_MEMORY,
    "validate_coupling": ToolCategory.VALIDATION,
    "modify_hardware_parameter": ToolCategory.EDITING,
    "step_hardware_parameter": ToolCategory.EDITING,
    "modify_topology_template": ToolCategory.EDITING,
    "modify_device_parameter": ToolCategory.EDITING,
}

_WRITE_TOOLS = {
    "modify_hardware_parameter",
    "step_hardware_parameter",
    "modify_topology_template",
    "modify_device_parameter",
    "evaluate_design",
    "commit_verified_case",
}
_CONTROL_TOOLS = {"select_analysis_base"}

_DOMAINS: Mapping[str, Tuple[str, ...]] = {
    "summarize_layer_ranks": ("workload", "architecture"),
    "summarize_operator_groups": ("workload", "architecture"),
    "inspect_layer_details": ("workload", "architecture"),
    "summarize_workload_features": ("workload",),
    "summarize_topology": ("topology",),
    "inspect_light_path": ("topology", "device"),
    "list_topology_templates": ("topology",),
    "inspect_topology_template": ("topology",),
    "validate_coupling": ("architecture", "topology", "device"),
    "searchable_parameters": ("architecture", "topology", "device"),
    "summarize_optical_structure": ("topology", "device"),
    "summarize_insertion_loss": ("topology", "device"),
    "summarize_insertion_loss_detail": ("topology", "device"),
    "summarize_topology_evidence": ("topology", "device"),
    "list_device_candidates": ("device",),
    "inspect_device_config": ("device",),
    "summarize_device_frontier": ("device",),
    "summarize_device_provenance": ("device",),
    "modify_topology_template": ("topology",),
    "modify_device_parameter": ("device",),
}


def _access(name: str) -> ToolAccess:
    if name in _WRITE_TOOLS:
        return ToolAccess.WRITE
    if name in _CONTROL_TOOLS:
        return ToolAccess.CONTROL
    return ToolAccess.READ


def _domains(name: str) -> Tuple[str, ...]:
    return _DOMAINS.get(name, ("global",))


def _registry_descriptors(registry: Mapping[str, object], source: str) -> List[ToolDescriptor]:
    descriptors: List[ToolDescriptor] = []
    for name, entry in registry.items():
        spec = entry[0]
        access = _access(name)
        descriptors.append(ToolDescriptor(
            name=name,
            category=_CATEGORY_BY_NAME[name],
            access=access,
            source=source,
            description=spec.description,
            domains=_domains(name),
            agent_visible=access is not ToolAccess.WRITE,
        ))
    return descriptors


def tool_descriptors() -> List[ToolDescriptor]:
    descriptors = _registry_descriptors(ANALYSIS_REGISTRY, "analysis_tools")
    descriptors.extend(_registry_descriptors(DEVICE_REGISTRY, "device_tools"))
    descriptors.extend(_registry_descriptors(SUPPORT_REGISTRY, "support_tools"))
    descriptors.extend(
        ToolDescriptor(
            name=name,
            category=_CATEGORY_BY_NAME[name],
            access=_access(name),
            source="topology_tools",
            description=description,
            domains=_domains(name),
            agent_visible=True,
        )
        for name, description in TOPOLOGY_TOOL_DESCRIPTIONS.items()
    )
    return sorted(descriptors, key=lambda item: (item.category.value, item.name))


def capability_manifest() -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for descriptor in tool_descriptors():
        grouped.setdefault(descriptor.category.value, []).append(descriptor.to_dict())
    return grouped


def validate_catalog(available_names: Iterable[str]) -> None:
    expected = {item.name for item in tool_descriptors()}
    actual = set(available_names)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"Capability catalog mismatch: missing={missing}, extra={extra}")
