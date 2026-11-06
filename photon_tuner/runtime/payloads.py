"""各角色阶段输入的类型化载荷契约。

这些构建函数集中了各阶段的 ``task`` 常量与必填字段约束，
确保阶段不会静默地向模型发送缺失字段——构造时立即报错，而不是在后续环节降级。
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

# 各角色阶段的规范 ``task`` 判别值。
TASK_MODEL = "model_level_analysis"
TASK_ARCHITECTURE = "architecture_analysis"
TASK_TOPOLOGY = "topology_analysis"
TASK_DEVICE = "device_analysis"
TASK_SOLUTION = "solution_generation"
TASK_EVALUATE = "evidence_evaluation"


class ModelPayload(TypedDict, total=False):
    task: str
    iteration: int
    metrics: Dict[str, Any]
    hardware_summary: Dict[str, Any]


class ArchitecturePayload(TypedDict, total=False):
    task: str
    candidate_layers: List[str]
    metrics: Dict[str, Any]


class TopologyPayload(TypedDict, total=False):
    task: str
    bottleneck_state: Dict[str, Any]
    metrics: Dict[str, Any]


class DevicePayload(TypedDict, total=False):
    task: str
    topology_diagnosis: Dict[str, Any]
    metrics: Dict[str, Any]


class SolutionPayload(TypedDict, total=False):
    task: str
    bottleneck_state: Dict[str, Any]
    optical_analysis: Dict[str, Any]
    retrieved_cases: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    hardware_summary: Dict[str, Any]
    search_space: Dict[str, Any]
    search_strategy: str
    focus_layer: str


class EvaluatePayload(TypedDict, total=False):
    task: str
    stage: str
    checklist: List[str]
    upstream_output: Dict[str, Any]
    context: Dict[str, Any]


def _build(task: str, required: tuple, **fields: Any) -> Dict[str, Any]:
    """返回固定 ``task`` 并强制必填字段的载荷字典。"""
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError(f"{task} payload missing required field(s): {missing}")
    return {"task": task, **fields}


def model_payload(iteration: int, metrics: Dict[str, Any], hardware_summary: Dict[str, Any]) -> ModelPayload:
    return _build(TASK_MODEL, ("metrics", "hardware_summary"), iteration=iteration, metrics=metrics, hardware_summary=hardware_summary)


def architecture_payload(candidate_layers: List[str], metrics: Dict[str, Any]) -> ArchitecturePayload:
    return _build(TASK_ARCHITECTURE, ("candidate_layers", "metrics"), candidate_layers=candidate_layers, metrics=metrics)


def topology_payload(bottleneck_state: Dict[str, Any], metrics: Dict[str, Any]) -> TopologyPayload:
    return _build(TASK_TOPOLOGY, ("bottleneck_state", "metrics"), bottleneck_state=bottleneck_state, metrics=metrics)


def device_payload(topology_diagnosis: Dict[str, Any], metrics: Dict[str, Any]) -> DevicePayload:
    return _build(TASK_DEVICE, ("topology_diagnosis", "metrics"), topology_diagnosis=topology_diagnosis, metrics=metrics)


def solution_payload(**fields: Any) -> SolutionPayload:
    return _build(
        TASK_SOLUTION,
        ("bottleneck_state", "optical_analysis", "retrieved_cases", "metrics",
         "hardware_summary", "search_space", "search_strategy", "focus_layer"),
        **fields,
    )


def evaluate_payload(stage: str, checklist: List[str], upstream_output: Dict[str, Any], context: Dict[str, Any]) -> EvaluatePayload:
    return _build(
        TASK_EVALUATE,
        ("stage", "checklist", "upstream_output", "context"),
        stage=stage, checklist=checklist, upstream_output=upstream_output, context=context,
    )
