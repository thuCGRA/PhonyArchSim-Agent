"""原子技能契约、注册表与执行器。

工具提供确定性接口；原子技能用领域规则组织工具证据并回答一个专业问题。技能
不得调用其他技能，条件分支、循环和跨技能重试属于组合工作流。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple


class SkillCategory(str, Enum):
    DIAGNOSIS = "diagnosis"
    DECISION = "decision"
    VERIFICATION = "verification"
    LEARNING = "learning"
    SCHEDULING = "scheduling"


_MISSING = object()


@dataclass(frozen=True)
class InputRef:
    key: str
    default: Any = field(default=_MISSING, repr=False)


@dataclass(frozen=True)
class SkillStep:
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


SkillEvaluator = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class AtomicSkillSpec:
    name: str
    goal: str
    description: str
    category: SkillCategory
    preconditions: Tuple[str, ...]
    required_evidence: Tuple[str, ...]
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    steps: Tuple[SkillStep, ...]
    decision_rule: str
    failure_policy: str
    evaluator: SkillEvaluator = field(repr=False, compare=False)
    allowed_roles: Tuple[str, ...] = ()
    version: str = "1.0"

    @property
    def tool_names(self) -> Tuple[str, ...]:
        return tuple(step.tool_name for step in self.steps)


@dataclass(frozen=True)
class SkillTraceEntry:
    step: int
    tool_name: str
    arguments: Mapping[str, Any]
    output: Any


@dataclass(frozen=True)
class SkillResult:
    name: str
    conclusion: Mapping[str, Any]
    evidence: Mapping[str, Any]
    trace: Tuple[SkillTraceEntry, ...]


class SkillExecutionError(RuntimeError):
    pass


class SkillRegistry:
    def __init__(self, skills: Iterable[AtomicSkillSpec] = ()) -> None:
        self._skills: Dict[str, AtomicSkillSpec] = {}
        for skill in skills:
            self.register(skill)

    def register(self, skill: AtomicSkillSpec, *, replace: bool = False) -> None:
        if skill.name in self._skills and not replace:
            raise ValueError(f"Atomic skill already registered: {skill.name}")
        if not skill.steps:
            raise ValueError(f"Atomic skill has no tool steps: {skill.name}")
        if not skill.goal.strip():
            raise ValueError(f"Atomic skill has no single goal: {skill.name}")
        self._skills[skill.name] = skill

    def get(self, name: str) -> AtomicSkillSpec:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise KeyError(f"Unknown atomic skill: {name}") from exc

    def list(self, category: Optional[SkillCategory] = None) -> List[AtomicSkillSpec]:
        skills: Iterable[AtomicSkillSpec] = self._skills.values()
        if category is not None:
            skills = (skill for skill in skills if skill.category == category)
        return sorted(skills, key=lambda skill: skill.name)

    def validate_tools(self, available_tool_names: Iterable[str]) -> None:
        available = set(available_tool_names)
        missing = {
            skill.name: sorted(set(skill.tool_names) - available)
            for skill in self._skills.values()
            if set(skill.tool_names) - available
        }
        if missing:
            raise ValueError(f"Atomic skills reference missing tools: {missing}")
        # 技能名不能与工具名重合，否则两层在 Agent 工具面上不可区分。
        overlap = sorted(set(self._skills) & available)
        if overlap:
            raise ValueError(f"Tool/skill names overlap: {overlap}")


def _resolve(value: Any, inputs: Mapping[str, Any]) -> Any:
    if isinstance(value, InputRef):
        if value.key in inputs:
            return inputs[value.key]
        if value.default is not _MISSING:
            return value.default
        raise SkillExecutionError(f"Missing atomic-skill input: {value.key}")
    if isinstance(value, dict):
        return {key: _resolve(item, inputs) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_resolve(item, inputs) for item in value]
    return value


def _payload(result: Any) -> Any:
    return getattr(result, "payload", result)


class SkillRuntime:
    """执行预定义工具步骤，再应用一个领域判断规则。"""

    def __init__(self, invoke_tool: Callable[[str, Mapping[str, Any]], Any]) -> None:
        self.invoke_tool = invoke_tool

    def execute(
        self,
        skill: AtomicSkillSpec,
        inputs: Optional[Mapping[str, Any]] = None,
        *,
        role: Optional[str] = None,
    ) -> SkillResult:
        inputs = dict(inputs or {})
        if role and skill.allowed_roles and role not in skill.allowed_roles:
            raise SkillExecutionError(f"Role {role!r} cannot invoke skill {skill.name!r}")
        missing_inputs = [
            key for key in (skill.input_schema.get("required") or []) if key not in inputs
        ]
        if missing_inputs:
            raise SkillExecutionError(f"Missing atomic-skill inputs: {missing_inputs}")

        trace: List[SkillTraceEntry] = []
        evidence: Dict[str, Any] = {}
        for index, step in enumerate(skill.steps, start=1):
            arguments = _resolve(dict(step.arguments), inputs)
            try:
                output = _payload(self.invoke_tool(step.tool_name, arguments))
            except Exception as exc:  # noqa: BLE001
                raise SkillExecutionError(
                    f"Atomic skill {skill.name!r} failed at tool step {index} "
                    f"({step.tool_name}): {exc}"
                ) from exc
            if isinstance(output, Mapping) and output.get("error"):
                raise SkillExecutionError(
                    f"Atomic skill {skill.name!r} failed at tool step {index} "
                    f"({step.tool_name}): {output['error']}"
                )
            evidence[step.tool_name] = output
            trace.append(SkillTraceEntry(index, step.tool_name, arguments, output))

        try:
            conclusion = dict(skill.evaluator(inputs, evidence))
        except Exception as exc:  # noqa: BLE001
            raise SkillExecutionError(
                f"Atomic skill {skill.name!r} could not apply its decision rule: {exc}"
            ) from exc
        return SkillResult(skill.name, conclusion, evidence, tuple(trace))
