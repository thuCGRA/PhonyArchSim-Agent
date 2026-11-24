"""统一导出工具、原子技能、组合工作流与角色绑定。

该模块是实现、测试和论文表格共同使用的单一事实源。工具按操作资源分类，
原子技能按专业任务分类，组合工作流按控制流组织；三者不会混为同一层。
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Mapping, Sequence

from photon_tuner.skills import build_default_skill_registry
from photon_tuner.toolbox.catalog import ToolAccess, tool_descriptors
from photon_tuner.workflows import build_default_workflow_registry


def role_bindings() -> Dict[str, Sequence[str]]:
    """返回六类 Agent 的显式能力白名单。延迟导入以保持领域层轻量。"""
    from photon_tuner.agents.builders import (
        ARCHITECTURE_TOOLS,
        BOTTLENECK_TOOLS,
        CANDIDATE_TOOLS,
        DEVICE_TOOLS,
        EVALUATE_TOOLS,
        TOPOLOGY_TOOLS,
    )

    return {
        "bottleneck": tuple(BOTTLENECK_TOOLS),
        "architecture": tuple(ARCHITECTURE_TOOLS),
        "topology": tuple(TOPOLOGY_TOOLS),
        "device": tuple(DEVICE_TOOLS),
        "candidate": tuple(CANDIDATE_TOOLS),
        "evaluate": tuple(EVALUATE_TOOLS),
    }


def build_capability_manifest(*, include_roles: bool = True) -> Dict[str, Any]:
    tools = tool_descriptors()
    skills = build_default_skill_registry().list()
    workflows = build_default_workflow_registry().list()
    manifest: Dict[str, Any] = {
        "summary": {
            "tool_count": len(tools),
            "agent_visible_tool_count": sum(item.agent_visible for item in tools),
            "atomic_skill_count": len(skills),
            "workflow_count": len(workflows),
            "tool_access": dict(sorted(Counter(item.access.value for item in tools).items())),
            "tool_categories": dict(sorted(Counter(item.category.value for item in tools).items())),
            "skill_categories": dict(sorted(Counter(item.category.value for item in skills).items())),
        },
        "tools": [item.to_dict() for item in tools],
        "atomic_skills": [
            {
                "name": item.name,
                "goal": item.goal,
                "category": item.category.value,
                "preconditions": list(item.preconditions),
                "required_evidence": list(item.required_evidence),
                "tools": list(item.tool_names),
                "decision_rule": item.decision_rule,
                "failure_policy": item.failure_policy,
                "allowed_roles": list(item.allowed_roles),
                "version": item.version,
            }
            for item in skills
        ],
        "composite_workflows": [
            {
                "name": item.name,
                "goal": item.goal,
                "entry": item.entry,
                "nodes": [
                    {
                        "name": node.name,
                        "skill": node.skill_name,
                        "conditional_targets": [edge.target for edge in node.transitions],
                        "default_target": node.default_target,
                    }
                    for node in item.nodes
                ],
            }
            for item in workflows
        ],
    }
    if include_roles:
        manifest["role_bindings"] = {name: list(items) for name, items in role_bindings().items()}
    return manifest


def validate_agent_capability_surface(bindings: Mapping[str, Sequence[str]] | None = None) -> None:
    """验证角色仅引用已注册能力，且不能直接调用宿主写工具。"""
    tools = {item.name: item for item in tool_descriptors()}
    skills = {item.name for item in build_default_skill_registry().list()}
    bindings = bindings or role_bindings()
    available = set(tools) | skills
    errors = []
    for role, names in bindings.items():
        if len(names) != len(set(names)):
            errors.append(f"{role}: duplicate capability names")
        missing = sorted(set(names) - available)
        if missing:
            errors.append(f"{role}: unknown={missing}")
        writes = sorted(name for name in names if name in tools and tools[name].access is ToolAccess.WRITE)
        if writes:
            errors.append(f"{role}: host-only writes={writes}")
    if errors:
        raise ValueError("Invalid Agent capability surface: " + "; ".join(errors))
