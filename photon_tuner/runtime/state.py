"""贯穿父 LangGraph 各节点的显式逐迭代状态。

``IterationContext`` 是每个图节点读取和写入的显式可序列化对象；
``snapshot`` / ``restore`` 将其通过纯 JSON 安全字典往返转换，
使 ``SqliteSaver`` 能在每个节点之后做检查点，并在崩溃后从最后完成的阶段恢复运行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set

from photon_tuner.agents.outputs import (
    ArchitectureOutput,
    DeviceOutput,
    ModelOutput,
    SolutionOutput,
    TopologyOutput,
)
from photon_tuner.core.pareto import ParetoArchive
from photon_tuner.core.schemas import EvaluationResult

# 五个分析/求解阶段的输出模型往返映射。
_OUTPUT_MODELS = {
    "model_out": ModelOutput,
    "arch_out": ArchitectureOutput,
    "topo_out": TopologyOutput,
    "device_out": DeviceOutput,
    "sol_out": SolutionOutput,
}

# 当 ``snapshot`` 布局变化时递增此值。``restore`` 会拒绝加载不兼容版本的
# 快照（并给出明确提示），而不是静默错配字段。
SNAPSHOT_SCHEMA_VERSION = 1


@dataclass
class IterationContext:
    """单次调优迭代所需的一切，保持显式且可序列化。"""

    iteration: int
    output_dir: Path
    current: EvaluationResult
    state: Any  # TuningState（保持无类型以规避导入循环）
    forbidden: Set[str]
    pareto: ParetoArchive
    search_strategy: str
    state_strategy: str
    rework_state: Dict[str, int] = field(default_factory=lambda: {"count": 0})
    rework_total: int = 0

    # 阶段输出，随流水线推进而填充，并传递给后续节点。
    model_out: Any = None
    arch_out: Any = None
    bottleneck_state: Dict[str, Any] = field(default_factory=dict)
    topo_out: Any = None
    device_out: Any = None
    optical_analysis: Dict[str, Any] = field(default_factory=dict)
    focus_layer: str = "joint"
    sol_out: Any = None
    sol_actions: List[Dict[str, Any]] = field(default_factory=list)
    focused_actions: List[Dict[str, Any]] = field(default_factory=list)
    focused_link: List[Dict[str, Any]] = field(default_factory=list)
    deferred: List[str] = field(default_factory=list)
    updated_hardware: Dict[str, Any] = field(default_factory=dict)
    applied: List[str] = field(default_factory=list)
    failed: List[Dict[str, str]] = field(default_factory=list)


def _snapshot_result(result: Any) -> Any:
    return result.to_dict() if result is not None else None


def _restore_result(data: Any) -> Any:
    return EvaluationResult.from_dict(data) if data is not None else None


def _check_schema_version(data: Dict[str, Any]) -> None:
    """拒绝本代码版本无法解释其布局的快照。"""
    version = data.get("schema_version")
    if version is None:
        raise ValueError(
            "snapshot is missing 'schema_version' (written by an older code "
            "version); re-run the tuning instead of resuming this run_dir"
        )
    if int(version) != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"snapshot schema_version {version!r} is incompatible with this code "
            f"(expected {SNAPSHOT_SCHEMA_VERSION}); re-run the tuning instead "
            "of resuming this run_dir"
        )


def snapshot(ctx: IterationContext) -> Dict[str, Any]:
    """将迭代上下文序列化为可用于检查点的 JSON 安全字典。"""
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "iteration": ctx.iteration,
        "output_dir": str(ctx.output_dir),
        "current": ctx.current.to_dict(),
        "state": {
            "best": _snapshot_result(ctx.state.best),
            "history": list(ctx.state.history),
            "iteration": ctx.state.iteration,
        },
        "forbidden": sorted(ctx.forbidden),
        "pareto": ctx.pareto.to_dict(),
        "search_strategy": ctx.search_strategy,
        "state_strategy": ctx.state_strategy,
        "rework_state": dict(ctx.rework_state),
        "rework_total": ctx.rework_total,
        "focus_layer": ctx.focus_layer,
        "bottleneck_state": dict(ctx.bottleneck_state),
        "optical_analysis": dict(ctx.optical_analysis),
        # 供 commit 节点消费的求解阶段输出（均为 JSON 安全）。
        "sol_actions": list(ctx.sol_actions),
        "focused_actions": list(ctx.focused_actions),
        "focused_link": list(ctx.focused_link),
        "deferred": list(ctx.deferred),
        "updated_hardware": dict(ctx.updated_hardware),
        "applied": list(ctx.applied),
        "failed": list(ctx.failed),
        # Pydantic 阶段输出（通过 model_dump 保证 JSON 安全）。
        "model_out": ctx.model_out.model_dump() if ctx.model_out is not None else None,
        "arch_out": ctx.arch_out.model_dump() if ctx.arch_out is not None else None,
        "topo_out": ctx.topo_out.model_dump() if ctx.topo_out is not None else None,
        "device_out": ctx.device_out.model_dump() if ctx.device_out is not None else None,
        "sol_out": ctx.sol_out.model_dump() if ctx.sol_out is not None else None,
    }


def restore(data: Dict[str, Any], tuner_state_cls: type) -> IterationContext:
    """从 ``snapshot`` 字典重建迭代上下文。

    ``tuner_state_cls`` 是 ``TuningState`` 数据类（通过注入传入，以规避本模块与
    ``runtime/orchestrator.py`` 之间的导入循环）。
    """
    _check_schema_version(data)
    state = tuner_state_cls(
        best=_restore_result(data["state"]["best"]),
        history=list(data["state"]["history"]),
        iteration=data["state"]["iteration"],
    )
    ctx = IterationContext(
        iteration=data["iteration"],
        output_dir=Path(data["output_dir"]),
        current=EvaluationResult.from_dict(data["current"]),
        state=state,
        forbidden=set(data["forbidden"]),
        pareto=ParetoArchive.from_dict(data["pareto"]),
        search_strategy=data["search_strategy"],
        state_strategy=data["state_strategy"],
        rework_state=dict(data["rework_state"]),
        rework_total=data["rework_total"],
        focus_layer=data["focus_layer"],
        bottleneck_state=dict(data["bottleneck_state"]),
        optical_analysis=dict(data["optical_analysis"]),
        sol_actions=list(data.get("sol_actions") or []),
        focused_actions=list(data.get("focused_actions") or []),
        focused_link=list(data.get("focused_link") or []),
        deferred=list(data.get("deferred") or []),
        updated_hardware=dict(data.get("updated_hardware") or {}),
        applied=list(data.get("applied") or []),
        failed=list(data.get("failed") or []),
    )
    for key, model in _OUTPUT_MODELS.items():
        raw = data.get(key)
        setattr(ctx, key, model.model_validate(raw) if raw is not None else None)
    return ctx
