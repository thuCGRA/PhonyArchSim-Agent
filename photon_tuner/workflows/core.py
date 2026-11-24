"""组合工作流：以条件边编排多个原子技能。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from photon_tuner.skills import SkillRegistry, SkillResult, SkillRuntime


END = "__end__"


@dataclass(frozen=True)
class WorkflowTransition:
    target: str
    field: str
    equals: Any

    def matches(self, conclusion: Mapping[str, Any]) -> bool:
        return conclusion.get(self.field) == self.equals


@dataclass(frozen=True)
class WorkflowNode:
    name: str
    skill_name: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    transitions: Tuple[WorkflowTransition, ...] = ()
    default_target: str = END


@dataclass(frozen=True)
class CompositeWorkflowSpec:
    name: str
    goal: str
    entry: str
    nodes: Tuple[WorkflowNode, ...]
    max_steps: int = 16

    def node_map(self) -> Dict[str, WorkflowNode]:
        return {node.name: node for node in self.nodes}


@dataclass(frozen=True)
class WorkflowResult:
    name: str
    terminal_node: str
    conclusions: Mapping[str, Mapping[str, Any]]
    skill_trace: Tuple[SkillResult, ...]


class WorkflowRegistry:
    def __init__(self, workflows: Iterable[CompositeWorkflowSpec] = ()) -> None:
        self._workflows: Dict[str, CompositeWorkflowSpec] = {}
        for workflow in workflows:
            self.register(workflow)

    def register(self, workflow: CompositeWorkflowSpec) -> None:
        if workflow.name in self._workflows:
            raise ValueError(f"Composite workflow already registered: {workflow.name}")
        nodes = workflow.node_map()
        if workflow.entry not in nodes:
            raise ValueError(f"Workflow entry is not a node: {workflow.entry}")
        for node in workflow.nodes:
            targets = [transition.target for transition in node.transitions] + [node.default_target]
            unknown = [target for target in targets if target != END and target not in nodes]
            if unknown:
                raise ValueError(f"Workflow {workflow.name} has unknown targets: {unknown}")
        self._workflows[workflow.name] = workflow

    def get(self, name: str) -> CompositeWorkflowSpec:
        return self._workflows[name]

    def list(self) -> List[CompositeWorkflowSpec]:
        return sorted(self._workflows.values(), key=lambda item: item.name)

    def validate_skills(self, skills: SkillRegistry) -> None:
        known = {skill.name for skill in skills.list()}
        missing = {
            workflow.name: sorted({node.skill_name for node in workflow.nodes} - known)
            for workflow in self._workflows.values()
            if {node.skill_name for node in workflow.nodes} - known
        }
        if missing:
            raise ValueError(f"Composite workflows reference missing skills: {missing}")


class WorkflowRuntime:
    def __init__(self, skills: SkillRegistry, skill_runtime: SkillRuntime) -> None:
        self.skills = skills
        self.skill_runtime = skill_runtime

    def execute(
        self,
        workflow: CompositeWorkflowSpec,
        inputs: Optional[Mapping[str, Any]] = None,
    ) -> WorkflowResult:
        nodes = workflow.node_map()
        current = workflow.entry
        shared_inputs = dict(inputs or {})
        conclusions: Dict[str, Mapping[str, Any]] = {}
        trace: List[SkillResult] = []
        last_node = current
        for _ in range(max(1, workflow.max_steps)):
            node = nodes[current]
            last_node = node.name
            skill_inputs = {**shared_inputs, **dict(node.inputs)}
            result = self.skill_runtime.execute(self.skills.get(node.skill_name), skill_inputs)
            conclusions[node.name] = result.conclusion
            trace.append(result)
            target = node.default_target
            for transition in node.transitions:
                if transition.matches(result.conclusion):
                    target = transition.target
                    break
            if target == END:
                return WorkflowResult(workflow.name, last_node, conclusions, tuple(trace))
            current = target
        raise RuntimeError(f"Composite workflow {workflow.name!r} exceeded max_steps={workflow.max_steps}")
