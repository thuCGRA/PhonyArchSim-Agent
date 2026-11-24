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
    BottleneckOutput,
    CandidateOutput,
    TopologyOutput,
)
from photon_tuner.core.pareto import ParetoArchive
from photon_tuner.core.schemas import EvaluationResult

# 五个分析/求解阶段的输出模型往返映射。
_OUTPUT_MODELS = {
    "bottleneck_out": BottleneckOutput,
    "arch_out": ArchitectureOutput,
    "topo_out": TopologyOutput,
    "device_out": DeviceOutput,
    "cand_out": CandidateOutput,
}

# 当 ``snapshot`` 布局变化时递增此值。``restore`` 会拒绝加载不兼容版本的
# 快照（并给出明确提示），而不是静默错配字段。
SNAPSHOT_SCHEMA_VERSION = 2


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
    bottleneck_out: Any = None
    arch_out: Any = None
    bottleneck_state: Dict[str, Any] = field(default_factory=dict)
    topo_out: Any = None
    device_out: Any = None
    optical_analysis: Dict[str, Any] = field(default_factory=dict)
    focus_layer: str = "joint"
    cand_out: Any = None
    cand_actions: List[Dict[str, Any]] = field(default_factory=list)
    focused_actions: List[Dict[str, Any]] = field(default_factory=list)
    focused_link: List[Dict[str, Any]] = field(default_factory=list)
    deferred: List[str] = field(default_factory=list)
    updated_hardware: Dict[str, Any] = field(default_factory=dict)
    applied: List[str] = field(default_factory=list)
    failed: List[Dict[str, str]] = field(default_factory=list)

    # -- RAG 驱动的动态调度状态 ------------------------------------------ #
    #: 本轮调度决策（``scheduler.ScheduleDecision.as_dict()``）。
    schedule: Dict[str, Any] = field(default_factory=dict)
    #: 逐轮决策，写入 trajectory / summary，作为论文里的调度轨迹。
    schedule_history: List[Dict[str, Any]] = field(default_factory=list)
    #: 逐轮**记忆成熟度**ρ：有效先例数与跨层覆盖率两者中较低者。
    memory_maturity: List[float] = field(default_factory=list)
    #: 逐轮已评估目标值，供收敛窗口计算历史最优改善。
    objective_history: List[float] = field(default_factory=list)
    #: 逐轮全局提名层数。
    top_layer_counts: List[int] = field(default_factory=list)
    #: 本轮候选阶段未能产出合法结构化输出（例如模型返回缺字段的 tool 调用）。
    #: 置位后 commit 不再评估、当前点保持不变——坏输出应浪费一轮，而不是浪费一次
    #: 仿真，更不应截断整次运行。
    stage_failed: bool = False
    #: 阶段执行总次数（含返工重跑）。κ 的分母必须用它，而不是 LLM 调用总数——
    #: 后者里混入了大量证据门调用，会把返工率稀释一个数量级，使 κ 规则永不触发。
    stage_invocations: int = 0
    #: 阶段 -> {迭代号字符串: 该阶段输出 dump}。供 pipeline/serial 复用偏移证据。
    evidence_history: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)


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
        "cand_actions": list(ctx.cand_actions),
        "focused_actions": list(ctx.focused_actions),
        "focused_link": list(ctx.focused_link),
        "deferred": list(ctx.deferred),
        "updated_hardware": dict(ctx.updated_hardware),
        "applied": list(ctx.applied),
        "failed": list(ctx.failed),
        # Pydantic 阶段输出（通过 model_dump 保证 JSON 安全）。
        "bottleneck_out": ctx.bottleneck_out.model_dump() if ctx.bottleneck_out is not None else None,
        "arch_out": ctx.arch_out.model_dump() if ctx.arch_out is not None else None,
        "topo_out": ctx.topo_out.model_dump() if ctx.topo_out is not None else None,
        "device_out": ctx.device_out.model_dump() if ctx.device_out is not None else None,
        "cand_out": ctx.cand_out.model_dump() if ctx.cand_out is not None else None,
        # 调度状态：必须随检查点一起存，否则恢复后会丢掉 ρ/π 的窗口。
        "schedule": dict(ctx.schedule),
        "schedule_history": list(ctx.schedule_history),
        "memory_maturity": list(ctx.memory_maturity),
        "objective_history": list(ctx.objective_history),
        "top_layer_counts": list(ctx.top_layer_counts),
        "stage_invocations": int(ctx.stage_invocations),
        "stage_failed": bool(ctx.stage_failed),
        "evidence_history": {
            stage: {str(it): dict(dump) for it, dump in by_iter.items()}
            for stage, by_iter in ctx.evidence_history.items()
        },
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
        cand_actions=list(data.get("cand_actions") or []),
        focused_actions=list(data.get("focused_actions") or []),
        focused_link=list(data.get("focused_link") or []),
        deferred=list(data.get("deferred") or []),
        updated_hardware=dict(data.get("updated_hardware") or {}),
        applied=list(data.get("applied") or []),
        failed=list(data.get("failed") or []),
        schedule=dict(data.get("schedule") or {}),
        schedule_history=list(data.get("schedule_history") or []),
        memory_maturity=list(data.get("memory_maturity") or []),
        objective_history=list(data.get("objective_history") or []),
        top_layer_counts=list(data.get("top_layer_counts") or []),
        stage_invocations=int(data.get("stage_invocations") or 0),
        stage_failed=bool(data.get("stage_failed") or False),
        evidence_history={
            stage: {str(it): dict(dump) for it, dump in by_iter.items()}
            for stage, by_iter in (data.get("evidence_history") or {}).items()
        },
    )
    for key, model in _OUTPUT_MODELS.items():
        raw = data.get(key)
        setattr(ctx, key, model.model_validate(raw) if raw is not None else None)
    return ctx
