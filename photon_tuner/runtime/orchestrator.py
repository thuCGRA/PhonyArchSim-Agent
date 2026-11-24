"""面向光子 DSE 的深度智能体编排器。

以固定顺序运行五个编译后的 LangGraph 智能体的确定性五阶段流水线
（顺序由宿主而非模型决定）：

    bottleneck -> architecture -> topology -> device -> candidate -> evaluate

其中 ``bottleneck`` 做全局瓶颈提名、``candidate`` 汇总上游诊断并产出这一个设计点
的编辑。哪些分析层在某轮真正下场，由 ``runtime/scheduler.py`` 依 RAG 就绪度等
观测量决定（``--schedule-strategy rag``）；固定模式下三层每轮都跑。

宿主维持研究的可复现性不变量：keep-best、禁用手纹去重、
``_repair_duplicate`` 回退、打包运行目录以及 ``trajectory.json``。
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from photon_tuner.agents.evaluate import (
    STAGE_CHECKLISTS,
    STRUCTURE_LEGALITY_ROLES,
    architecture_legality_check,
    decide,
    deterministic_checks,
    structure_legality_check,
    evidence_completeness_check,
    topology_binding_check,
)
from photon_tuner.agents.outputs import (
    ArchitectureOutput,
    DeviceOutput,
    EvaluateOutput,
    BottleneckOutput,
    CandidateOutput,
    TopologyOutput,
)
from photon_tuner.agents.builders import (
    build_architecture_agent,
    build_device_agent,
    build_evaluate_agent,
    build_bottleneck_agent,
    build_candidate_agent,
    build_topology_agent,
    build_generalist_agent_adapters,
)
from photon_tuner.common.checkpoint import build_checkpointer
from photon_tuner.common.io import append_jsonl, write_json
from photon_tuner.common.logging import get_logger
from photon_tuner.common.metrics import RunMetrics
from photon_tuner.common.paths import MEMORY_DB_PATH
from photon_tuner.common.tracing import LLMTraceCollector
from photon_tuner.common.truncate import fit_payload
from photon_tuner.runtime.payloads import (
    architecture_payload,
    device_payload,
    evaluate_payload,
    bottleneck_payload,
    candidate_payload,
    topology_payload,
)
from photon_tuner.runtime.scheduler import (
    ANALYSIS_LAYERS,
    ScheduleDecision,
    ScheduleSignals,
    SchedulerConfig,
    decide as schedule_decide,
    fixed_mode_decision,
    summarize_history as summarize_schedule,
)
from photon_tuner.runtime.state import IterationContext, restore, snapshot
from photon_tuner.core.objective import metrics_with_objective
from photon_tuner.core.pareto import ParetoArchive
from photon_tuner.core.schemas import EvaluationResult
from photon_tuner.core.topology_spec import StaticTopologyProvider
from photon_tuner.core.search_space import (
    ARCH_PARAM_KEYS,
    DesignSpace,
    build_design_space,
    discover_device_catalog,
    hardware_summary,
    materialize_hardware,
    normalize_hardware_to_design_space,
    step_arch_parameter,
    switch_topology_template,
)
from photon_tuner.core.topology_space import evaluable_template_names
from photon_tuner.llm.cache import ResponseCache, enable_langchain_llm_cache
from photon_tuner.llm.config import LLMConfig
from photon_tuner.llm.model import build_chat_model, is_transient_llm_error, llm_retry
from photon_tuner.rag.vector_store import (
    HistoryVectorStore,
    hardware_fingerprint,
    summarize_cases,
    summarize_documents,
)
from photon_tuner.skills import build_default_skill_registry
from photon_tuner.simulators.base import SimulatorAdapter
from photon_tuner.toolbox.analysis_tools import AnalysisToolbox
from photon_tuner.toolbox.bindings import (
    ANALYSIS_REGISTRY,
    DEVICE_REGISTRY,
    RunState,
    wrap_registry,
)
from photon_tuner.toolbox.bindings.topology_tools import build_topology_tools
from photon_tuner.toolbox.bindings.skill_tools import build_atomic_skill_tools
from photon_tuner.toolbox.catalog import validate_catalog
from photon_tuner.toolbox.common_tool.context import ToolContext
from photon_tuner.toolbox.device_tools import DeviceToolbox
from photon_tuner.toolbox.support_tools import SupportToolbox
from photon_tuner.toolbox.support_tools import _REGISTRY as SUPPORT_REGISTRY

log = get_logger("runtime.orchestrator")

EDIT_TOOLS = {
    "modify_hardware_parameter",
    "step_hardware_parameter",
    "modify_topology_template",
}
LINK_EDIT_TOOLS = {"modify_device_parameter"}

_DEFAULT_EDIT_DESIGN_SPACE = DesignSpace()
_SCALAR_EDIT_LAYERS = frozenset({"architecture", "topology"})
_SEQUENTIAL_LAYERS = ("architecture", "topology", "device")


def _limit_edits(
    actions: List[Dict[str, Any]],
    link_edits: List[Dict[str, Any]],
    limit: int,
) -> tuple:
    """把一次提案的应用编辑数限制到 ``limit``，其余作为推迟项返回。

    多参数同时改在乘积目标下有两个问题：各项可能相互抵消（例如面积 +45% 抵消
    能耗 −11%），而且**无法归因**——下一轮不知道该保留哪一半。因此默认一次
    只承载一个编辑假设，其余推迟到后续轮次；模型给出的顺序即其优先级。

    ``limit <= 0`` 表示不限制（用于消融：多编辑 vs 单编辑）。
    """
    if limit is None or int(limit) <= 0:
        return list(actions), list(link_edits), []
    budget = int(limit)
    kept_actions: List[Dict[str, Any]] = []
    kept_links: List[Dict[str, Any]] = []
    deferred: List[str] = []
    for action in actions:
        updates = ((action.get("arguments") or {}).get("updates") or [])
        if budget <= 0:
            deferred.extend(str(u.get("parameter")) for u in updates if isinstance(u, dict))
            continue
        if len(updates) > budget:
            # 单个动作内就超预算：切出前 budget 个参数，其余推迟。
            kept = dict(action)
            arguments = dict(action.get("arguments") or {})
            arguments["updates"] = updates[:budget]
            kept["arguments"] = arguments
            kept_actions.append(kept)
            deferred.extend(str(u.get("parameter")) for u in updates[budget:])
            budget = 0
            continue
        kept_actions.append(action)
        budget -= max(len(updates), 1)
    for link in link_edits:
        if budget > 0:
            kept_links.append(link)
            budget -= 1
        else:
            deferred.append(str((link.get("arguments") or {}).get("device_type") or "link_edit"))
    return kept_actions, kept_links, deferred


def _pct_delta(before: Any, after: Any) -> Optional[float]:
    """相对变化百分比（正 = 变好，即目标下降）。"""
    try:
        b, a = float(before), float(after)
    except (TypeError, ValueError):
        return None
    if b == 0:
        return None
    return round((b - a) / b * 100.0, 2)


def _log_delta(before: Any, after: Any) -> Optional[float]:
    """自然对数差 ``ln(after) - ln(before)``；任一值非正时返回 None。"""
    try:
        b, a = float(before), float(after)
    except (TypeError, ValueError):
        return None
    if b <= 0 or a <= 0:
        return None
    return math.log(a) - math.log(b)


def _json_safe(value: Any) -> Any:
    """递归地将值转换为可用于评判载荷的 JSON 可序列化形式。

    不可序列化的对象（例如门控 ``stage_context`` 中携带的 ``DesignSpace``）
    在提供 ``summary()`` 时用它概括，否则用 ``repr()``。转换之前，原始对象
    仍被确定性的 ``structure_legality_check`` 使用，因此数值耦合约束保持完好，
    而语义评判的载荷保持 JSON 安全。
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "summary"):
        try:
            return _json_safe(value.summary())
        except Exception:  # noqa: BLE001 — 尽力而为的概括
            pass
    return repr(value)


def _aggregate_tokens(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总追踪缓冲区中每个 ``llm_end`` 事件的 token 用量字典。"""
    total: Dict[str, Any] = {}
    for record in records:
        if record.get("event") != "llm_end":
            continue
        for key, value in (record.get("token_usage") or {}).items():
            if isinstance(value, (int, float)):
                total[key] = total.get(key, 0) + int(value)
    return total


def _safe_delete_thread(delete_thread, thread_id: str) -> None:
    """尽力而为地删除检查点线程；绝不抛出异常。"""
    try:
        delete_thread(thread_id)
    except Exception:  # noqa: BLE001 — 清理尽力而为
        pass


def _edit_tool_name(edit: Any) -> str:
    """提取提议编辑的工具名，兼容 LLM 的 ``action`` 别名。

    协调器自身的提示使用 ``name``，但模型有时会输出 ``action``（或空 ``{}``）。
    同时读取两个键——并对非字典/空条目返回 ``""``——可避免单个畸形编辑
    使整个应用环节崩溃；门控第 5 行的编辑一致性检查仍会标记它。
    """
    if not isinstance(edit, dict):
        return ""
    return str(edit.get("name") or edit.get("action") or "")


def _action_layer(action: Dict[str, Any], design_space: Optional[DesignSpace] = None) -> str:
    """Return the primary layer, preserving topology priority for mixed edits."""
    layers = _action_layers(action, design_space)
    return next((layer for layer in ("topology", "device", "architecture") if layer in layers), "architecture")


def _action_layers(action: Dict[str, Any], design_space: Optional[DesignSpace] = None) -> Set[str]:
    """Return every layer touched by an edit, including mixed parameter batches."""
    name = _edit_tool_name(action)
    args = (action or {}).get("arguments", {}) or {}
    if name in LINK_EDIT_TOOLS:
        return {"device"}
    if name == "modify_topology_template":
        return {"topology"}
    if name == "modify_hardware_parameter":
        updates = (update for update in (args.get("updates") or []) if isinstance(update, dict))
    elif name == "step_hardware_parameter":
        updates = (args,)
    else:
        return set()
    space = design_space or _DEFAULT_EDIT_DESIGN_SPACE
    return {
        layer
        for update in updates
        if (layer := space.layer_of(str(update.get("parameter", "")))) is not None
    }


def _filter_edits_for_layer(
    actions: List[Dict[str, Any]],
    link_edits: List[Dict[str, Any]],
    layer: str,
    design_space: Optional[DesignSpace] = None,
):
    """返回限定到 ``layer``（"joint" 时为全部）的 ``(actions, link_edits)``。

    混合架构+拓扑更新的 ``modify_hardware_parameter`` 会被拆分，使顺序迭代
    只应用聚焦层的更新。
    """
    if layer == "joint":
        return list(actions), list(link_edits)
    if layer == "device":
        # Device-slot variants use link_edits. Generic scalar tools do not write
        # the simulator's laser-source or splitter instance selectors.
        return [], list(link_edits)
    if layer not in _SCALAR_EDIT_LAYERS:
        return [], []

    space = design_space or _DEFAULT_EDIT_DESIGN_SPACE
    kept_actions: List[Dict[str, Any]] = []
    for a in actions:
        name = _edit_tool_name(a)
        args = (a or {}).get("arguments", {}) or {}
        if name == "modify_topology_template":
            if layer == "topology":
                kept_actions.append(a)
            continue
        if name == "modify_hardware_parameter":
            kept_updates = [
                update for update in (args.get("updates") or [])
                if isinstance(update, dict) and space.layer_of(str(update.get("parameter", ""))) == layer
            ]
            if kept_updates:
                fa = dict(a)
                fa["arguments"] = dict(args)
                fa["arguments"]["updates"] = kept_updates
                kept_actions.append(fa)
        elif name == "step_hardware_parameter":
            if space.layer_of(str(args.get("parameter", ""))) == layer:
                kept_actions.append(a)
    return kept_actions, []


@dataclass
class TuningState:
    """单条调优轨迹的累积状态。"""

    best: Optional[EvaluationResult] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    iteration: int = 0


class _RunGraphState(TypedDict, total=False):
    """对五个阶段排序的父 StateGraph 的状态。

    ``snapshot`` 携带序列化后的 :class:`IterationContext`，使 ``SqliteSaver``
    能在每个节点之后做检查点；其余键是 commit 节点的循环/终止路由所需的
    运行级常量。
    """

    snapshot: Dict[str, Any]
    run_dir: str
    max_iterations: int
    trajectory_logging: bool
    done: bool


class DeepAgentTuner:
    """围绕 SimPhony 适配器编排五个深度角色智能体。"""

    def __init__(
        self,
        adapter: SimulatorAdapter,
        llm_config: LLMConfig,
        devicelib_root: str,
        store: Optional[HistoryVectorStore] = None,
        output_root: Optional[Path] = None,
        search_space: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        model: Optional[Any] = None,
        enable_gate: bool = True,
        max_gate_retries: int = 2,
        max_stage_rework: int = 3,
        recursion_limit: int = 80,
        max_payload_chars: int = 24000,
        min_retrieval_score: float = 0.0,
        max_store_records: Optional[int] = None,
        max_edits_per_proposal: int = 0,
        schedule_strategy: str = "rag",
        fixed_schedule_mode: str = "serial",
        scheduler_config: Optional[SchedulerConfig] = None,
        agent_mode: str = "multi",
        analysis_mode: str = "layered",
        enable_rag: bool = True,
    ) -> None:
        self.adapter = adapter
        # 调度策略：``rag`` = 由 RAG 就绪度等可观测量动态决定；
        # ``fixed`` = 固定模式（消融对照，不读信号）。
        if schedule_strategy not in ("rag", "fixed"):
            raise ValueError(
                f"schedule_strategy must be 'rag' or 'fixed', got {schedule_strategy!r}"
            )
        # 单次提案最多应用多少个编辑。``0`` = 不限制（历史行为）；``1`` = 与
        # 传统基线同粒度（每次评估只承载一个编辑假设），使"谁更会选方向"成为
        # 唯一比较变量。设为 1 属于设计选择，需用消融数据支撑，故不设为默认。
        self.max_edits_per_proposal = int(max_edits_per_proposal)
        self.schedule_strategy = schedule_strategy
        self.fixed_schedule_mode = fixed_schedule_mode
        self.scheduler_config = scheduler_config or SchedulerConfig()
        if agent_mode not in ("multi", "single"):
            raise ValueError(f"agent_mode must be 'multi' or 'single', got {agent_mode!r}")
        if analysis_mode not in ("layered", "one-shot"):
            raise ValueError(
                f"analysis_mode must be 'layered' or 'one-shot', got {analysis_mode!r}"
            )
        self.agent_mode = agent_mode
        self.analysis_mode = analysis_mode
        self.enable_rag = bool(enable_rag)
        self.max_iterations = 0
        self.enable_gate = enable_gate
        self.max_gate_retries = max(0, int(max_gate_retries))
        # 每轮返工预算：整个迭代中最多进行这么多次阶段重执行（默认 3），
        # 独立于每阶段至多 1 次的评判。
        self.max_stage_rework = max(0, int(max_stage_rework))
        # 单个角色智能体工具循环的模型轮次预算。收集证据的角色在回答前会发起
        # 多次报告调用，因此该值须明显大于预期调用次数。
        self.recursion_limit = max(10, int(recursion_limit))
        self._last_transcript: Optional[List[Any]] = None
        self._last_invoke_error: Optional[str] = None
        # 跳过未变化检测：上一迭代的阶段输出签名。
        self._prev_stage_outputs: Dict[str, str] = {}
        self.devicelib_root = str(devicelib_root)
        self.output_root = Path(output_root or Path("runs/run_agent")).resolve()
        self.search_space = search_space or {}
        self.top_k = top_k
        self.min_retrieval_score = float(min_retrieval_score)
        self.store = store if store is not None else HistoryVectorStore(
            MEMORY_DB_PATH, max_records=max_store_records
        )
        self.simulator_schema = adapter.schema()
        # 可求值模板模式固定为 real：智能体侧只支持真实 ONNArchSimulator 路径。
        # 通用网格（Reck / Clements）与 crossbar 家族、WDM 权重阵列均可在此路径上构建。
        self._evaluable_mode = "real"

        self.design_space = build_design_space(
            self.search_space,
            device_catalog=discover_device_catalog(self.devicelib_root),
        )
        self.analysis_toolbox = AnalysisToolbox(search_space=self.search_space)
        self.device_toolbox = DeviceToolbox(self.devicelib_root, search_space=self.search_space)
        self.support_toolbox = SupportToolbox(self.devicelib_root)

        # 共享的可变运行状态：智能体的工具读写此上下文，且每轮迭代重新绑定
        # （见 _bind_context）。
        self.run_state = RunState(context=ToolContext(output_dir=self.output_root))

        # 一次性构建全部确定性工具和原子技能；每个智能体只选择自己的白名单。
        self.all_tools: Dict[str, Any] = {}
        for t in wrap_registry(self.analysis_toolbox, self.run_state, ANALYSIS_REGISTRY):
            self.all_tools[t.name] = t
        for t in wrap_registry(self.device_toolbox, self.run_state, DEVICE_REGISTRY):
            self.all_tools[t.name] = t
        for t in wrap_registry(self.support_toolbox, self.run_state, SUPPORT_REGISTRY):
            self.all_tools[t.name] = t
        self.topology_provider = StaticTopologyProvider()
        for t in build_topology_tools(
            self.topology_provider, self.run_state,
            design_space=self.design_space, mode=self._evaluable_mode,
        ):
            self.all_tools[t.name] = t
        # 工具与原子技能使用不同命名空间：先验证完整的 40 个确定性工具，再把
        # 15 个只读专业判断技能包装成 Agent 可调用的 StructuredTool。
        validate_catalog(self.all_tools)
        self.skill_registry = build_default_skill_registry()
        self.skill_registry.validate_tools(self.all_tools)
        for skill_tool in build_atomic_skill_tools(self.skill_registry, self.all_tools):
            self.all_tools[skill_tool.name] = skill_tool

        self.llm_config = llm_config
        self.model = model if model is not None else build_chat_model(llm_config)
        self.max_payload_chars = max(1000, int(max_payload_chars))
        # 运行级聚合指标与进程内响应缓存；指标在每次 tune() 开始时重置。
        self._run_metrics = RunMetrics()
        self._response_cache = ResponseCache(
            enabled=bool(getattr(llm_config, "enable_cache", True))
        )
        # 框架级 LLM 缓存：缓存原始模型调用。
        if getattr(llm_config, "enable_cache", True):
            enable_langchain_llm_cache()
        self._current_iteration = 0
        self._thread_seq = 0

    # ------------------------------------------------------------------ #

    def tune(
        self,
        initial_hardware: Dict[str, Any],
        max_iterations: int = 4,
        state_strategy: str = "best",
        trajectory_logging: bool = True,
        il_budget: Optional[float] = None,
        search_strategy: str = "joint",
        topology_sweep: bool = False,
        resume_run_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """运行父 StateGraph 的 DSE 循环并返回调优摘要。

        ``search_strategy`` 选择联合搜索模式：

        * ``"joint"``      —— 方案在每次迭代提出跨层编辑，并全部一起应用（默认）。
        * ``"sequential"`` —— 宿主每轮迭代循环一个层
          （architecture -> topology -> device），仅应用该层的编辑，
          其余推迟到各自所属的迭代。

        ``topology_sweep`` 在循环开始前，对初始硬件评估每一个*其他*可求值的
        电路模板（core_kind），将结果归档到 Pareto 前沿，使拓扑轴得到系统性的
        探索，而不是仅在 LLM 恰好提出时才探索。轨迹的 ``current`` 点保持不变。

        ``resume_run_dir`` 用于恢复此前中断的运行：传入崩溃运行的目录，
        父图将从其最后做检查点的阶段继续，而不再重跑初始化和之前的阶段。
        """
        self.max_iterations = int(max_iterations)
        self._search_strategy_arg = search_strategy
        if search_strategy not in ("joint", "sequential"):
            raise ValueError(f"search_strategy must be 'joint' or 'sequential', got {search_strategy!r}")
        hardware = normalize_hardware_to_design_space(
            materialize_hardware(initial_hardware), self.search_space
        )
        if il_budget is not None:
            self.search_space["il_budget"] = il_budget

        if resume_run_dir is not None:
            run_dir = Path(resume_run_dir)
        else:
            run_dir = self._next_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        self.adapter.output_root = run_dir

        self._agents = self._build_agents()
        self._prev_stage_outputs = {}
        self._run_metrics = RunMetrics()
        # 每次运行的树状追踪森林 + 阶段节点耗时。
        self._trace_forest: List[Dict[str, Any]] = []
        self._stage_timings: Dict[str, List[float]] = {}
        # 记录智能体线程序列水位线，使*已完成的*运行能精确删除其生成的子图检查点。
        start_seq = self._thread_seq

        graph = self._build_run_graph()
        self._write_graph_viz(graph, run_dir)
        config = {"configurable": {"thread_id": run_dir.name}}

        if resume_run_dir is not None:
            if not self._has_interrupted_checkpoint(graph, config):
                raise ValueError(
                    f"no interrupted checkpoint found for {run_dir}; "
                    "pass the directory of a crashed run to resume it"
                )
            result = graph.invoke(None, config)
        else:
            current = self._evaluate(hardware, 0)
            state = TuningState(best=current, iteration=0)
            state.history.append(self._history_entry(current, is_best=True))
            forbidden: Set[str] = {hardware_fingerprint(hardware)}
            rework_total = 0
            pareto = ParetoArchive()
            pareto.add(current.metrics, self._pareto_meta(current.hardware, 0, state_strategy, search_strategy))

            if topology_sweep:
                current_kind = str(
                    (current.hardware.get("arch", {}).get("topology", {}) or {}).get(
                        "core_kind", "dptc_crossbar"
                    )
                )
                for core_kind in evaluable_template_names(self._evaluable_mode):
                    if core_kind == current_kind:
                        continue
                    try:
                        sweep_hw = normalize_hardware_to_design_space(
                            switch_topology_template(current.hardware, core_kind),
                            self.search_space,
                        )
                    except ValueError:
                        continue
                    sweep_result = self._evaluate(sweep_hw, 0)
                    pareto.add(
                        sweep_result.metrics,
                        self._pareto_meta(sweep_result.hardware, 0, state_strategy, search_strategy),
                    )
                    forbidden.add(hardware_fingerprint(sweep_hw))
                    if self._is_better(sweep_result, state.best):
                        state.best = sweep_result

            initial_ctx = IterationContext(
                iteration=1,
                output_dir=run_dir / "iter_001" / "analysis",
                current=current,
                state=state,
                forbidden=forbidden,
                pareto=pareto,
                search_strategy=search_strategy,
                state_strategy=state_strategy,
                rework_state={"count": 0},
                rework_total=rework_total,
            )
            initial_ctx.output_dir.mkdir(parents=True, exist_ok=True)
            result = graph.invoke(
                {
                    "snapshot": snapshot(initial_ctx),
                    "run_dir": str(run_dir),
                    "max_iterations": max_iterations,
                    "trajectory_logging": trajectory_logging,
                    "done": False,
                },
                config,
            )

        final_ctx = restore(result["snapshot"], TuningState)

        if trajectory_logging:
            write_json(run_dir / "trajectory.json", {
                "best": final_ctx.state.best.metrics if final_ctx.state.best else None,
                "best_hardware": final_ctx.state.best.hardware if final_ctx.state.best else None,
                "history": final_ctx.state.history,
                "rework_total": final_ctx.rework_total,
                "search_strategy": search_strategy,
                "schedule_strategy": self.schedule_strategy,
                "agent_mode": self.agent_mode,
                "analysis_mode": self.analysis_mode,
                "rag_enabled": self.enable_rag,
                "schedule_history": final_ctx.schedule_history,
                "schedule_summary": summarize_schedule(final_ctx.schedule_history),
                "memory_maturity": final_ctx.memory_maturity,
                "pareto_frontier": final_ctx.pareto.to_dict(),
            })

        write_json(run_dir / "pareto_frontier.json", final_ctx.pareto.to_dict())
        write_json(run_dir / "summary.json", {
            "iterations": final_ctx.state.iteration,
            "best": final_ctx.state.best.metrics if final_ctx.state.best else None,
            "best_hardware": final_ctx.state.best.hardware if final_ctx.state.best else None,
            "rework_total": final_ctx.rework_total,
            "search_strategy": search_strategy,
            "schedule_strategy": self.schedule_strategy,
            "agent_mode": self.agent_mode,
            "analysis_mode": self.analysis_mode,
            "rag_enabled": self.enable_rag,
            "fixed_schedule_mode": self.fixed_schedule_mode if self.schedule_strategy == "fixed" else None,
            "scheduler_config": {
                "improved_delta_pct": self.store.improved_delta_pct,
                "warm_precedent_target": self.scheduler_config.warm_precedent_target,
                "minimum_mature_layer_coverage": self.scheduler_config.minimum_mature_layer_coverage,
                "convergence_window": self.scheduler_config.convergence_window,
                "convergence_delta": self.scheduler_config.convergence_delta,
                "focus_exit_delta": self.scheduler_config.focus_exit_delta,
                "max_pipeline_evidence_age": self.scheduler_config.max_pipeline_evidence_age,
                "max_consecutive_focus": self.scheduler_config.max_consecutive_focus,
            },
            "schedule_summary": summarize_schedule(final_ctx.schedule_history),
            "schedule_history": final_ctx.schedule_history,
            "metrics": self._run_metrics.to_dict(),
        })
        # 聚合调用树 + 各阶段节点耗时。
        write_json(run_dir / "llm_trace_tree.json", {"stages": self._trace_forest})
        write_json(run_dir / "stage_timings.json", {
            name: {
                "calls": len(durations),
                "total_s": round(sum(durations), 4),
                "mean_s": round(sum(durations) / len(durations), 4) if durations else None,
                "max_s": round(max(durations), 4) if durations else None,
            }
            for name, durations in self._stage_timings.items()
        })

        # 已完成的运行不再需要其检查点用于恢复；删除它们以使两个 SQLite
        # 存储保持有界。
        self._cleanup_checkpoints(run_dir.name, start_seq)

        return {
            "iterations": final_ctx.state.iteration,
            "best": final_ctx.state.best.metrics if final_ctx.state.best else None,
            "best_hardware": final_ctx.state.best.hardware if final_ctx.state.best else None,
            "history": final_ctx.state.history,
            "run_dir": str(run_dir),
            "rework_total": final_ctx.rework_total,
            "search_strategy": search_strategy,
            "schedule_strategy": self.schedule_strategy,
            "schedule_summary": summarize_schedule(final_ctx.schedule_history),
            "schedule_history": final_ctx.schedule_history,
            "pareto_frontier": final_ctx.pareto.to_dict(),
        }

    # -- Internals --------------------------------------------------------- #

    def _build_agents(self) -> Dict[str, Any]:
        # 一个共享的检查点器将每次智能体调用置于唯一的 thread_id 命名空间下
        # （见 _invoke），使每个阶段的最终状态都被记录检查点以供审计/回放，
        # 而不会累积跨调用的历史。
        checkpointer = build_checkpointer(self.output_root / "checkpoints.sqlite")
        if self.agent_mode == "single":
            return build_generalist_agent_adapters(
                self.model, self.all_tools, checkpointer
            )
        return {
            "bottleneck": build_bottleneck_agent(self.model, self.all_tools, checkpointer),
            "architecture": build_architecture_agent(self.model, self.all_tools, checkpointer),
            "topology": build_topology_agent(self.model, self.all_tools, checkpointer),
            "device": build_device_agent(self.model, self.all_tools, checkpointer),
            "candidate": build_candidate_agent(self.model, self.all_tools, checkpointer),
            "evaluate": build_evaluate_agent(self.model, self.all_tools, checkpointer),
        }

    def _build_run_graph(self):
        """编译父 StateGraph：五个阶段节点 + 一个 commit 节点。

        每个节点从图状态恢复序列化后的 :class:`IterationContext`，运行一个阶段，
        然后重新序列化回去——这样 SqliteSaver 检查点器在每个节点之后都持久化
        一个快照，崩溃的运行可从最后完成的阶段恢复。
        """
        graph = StateGraph(_RunGraphState)

        # 节点级瞬时错误重试：``_invoke`` 内部的细粒度 tenacity 重试处理瞬时
        # LLM 调用；此 RetryPolicy 是阶段节点本身任何其他瞬时失败的粗粒度兜底，
        # 且错误处理器保证即使在重试耗尽时运行也不会崩溃。
        retry_policy = RetryPolicy(
            retry_on=is_transient_llm_error,
            max_attempts=max(1, int(getattr(self.llm_config, "retry_attempts", 2) or 2)) + 1,
        )

        def _stage_node(name, stage_method):
            def _node(state: Dict[str, Any]) -> Dict[str, Any]:
                before = state["snapshot"]
                ctx = restore(before, TuningState)
                started = time.monotonic()
                try:
                    stage_method(ctx)
                except Exception:  # noqa: BLE001 — 保留原异常供图重试，同时记录根因
                    log.exception("stage %s failed at iteration %d", name, ctx.iteration)
                    raise
                self._stage_timings.setdefault(name, []).append(time.monotonic() - started)
                after = snapshot(ctx)
                self._log_state_change(name, before, after, Path(state.get("run_dir", str(self.output_root))))
                return {"snapshot": after}

            def _error_handler(state: Dict[str, Any]) -> Dict[str, Any]:
                # 重试已耗尽：将阶段标记为降级，使门控的 agent_completion 行判定其
                # 失败，以便返工循环重试，而不是让异常中止整个运行。
                self._last_invoke_error = self._last_invoke_error or f"{name} stage failed after retries"
                return {}

            return _node, _error_handler

        def _commit_node(state: Dict[str, Any]) -> Dict[str, Any]:
            before = state["snapshot"]
            ctx = restore(before, TuningState)
            started = time.monotonic()
            self._run_commit_stage(ctx)
            self._stage_timings.setdefault("commit", []).append(time.monotonic() - started)
            if ctx.iteration >= int(state.get("max_iterations", 0)):
                after = snapshot(ctx)
                self._log_state_change("commit", before, after, Path(state.get("run_dir", str(self.output_root))))
                return {"snapshot": after, "done": True}
            # 推进到下一迭代：新的输出目录 + 返工预算，并清空逐迭代阶段输出。
            ctx.iteration += 1
            ctx.output_dir = Path(state["run_dir"]) / f"iter_{ctx.iteration:03d}" / "analysis"
            ctx.output_dir.mkdir(parents=True, exist_ok=True)
            ctx.rework_state = {"count": 0}
            self._reset_stage_outputs(ctx)
            after = snapshot(ctx)
            self._log_state_change("commit", before, after, Path(state.get("run_dir", str(self.output_root))))
            return {"snapshot": after, "done": False}

        def _commit_error_handler(state: Dict[str, Any]) -> Dict[str, Any]:
            self._last_invoke_error = self._last_invoke_error or "commit stage failed after retries"
            return {}

        def _route_after_commit(state: Dict[str, Any]) -> str:
            return END if state.get("done") else "bottleneck"

        bottleneck_node, bottleneck_err = _stage_node("bottleneck", self._run_bottleneck_stage)
        arch_node, arch_err = _stage_node("architecture", self._run_architecture_stage)
        topo_node, topo_err = _stage_node("topology", self._run_topology_stage)
        device_node, device_err = _stage_node("device", self._run_device_stage)
        candidate_node, candidate_err = _stage_node("candidate", self._run_candidate_stage)

        graph.add_node("bottleneck", bottleneck_node, retry_policy=retry_policy, error_handler=bottleneck_err)
        graph.add_node("architecture", arch_node, retry_policy=retry_policy, error_handler=arch_err)
        graph.add_node("topology", topo_node, retry_policy=retry_policy, error_handler=topo_err)
        graph.add_node("device", device_node, retry_policy=retry_policy, error_handler=device_err)
        graph.add_node("candidate", candidate_node, retry_policy=retry_policy, error_handler=candidate_err)
        graph.add_node("commit", _commit_node, retry_policy=retry_policy, error_handler=_commit_error_handler)
        graph.add_edge(START, "bottleneck")
        graph.add_edge("bottleneck", "architecture")
        graph.add_edge("architecture", "topology")
        graph.add_edge("topology", "device")
        graph.add_edge("device", "candidate")
        graph.add_edge("candidate", "commit")
        graph.add_conditional_edges("commit", _route_after_commit)
        return graph.compile(
            checkpointer=build_checkpointer(self.output_root / "run_checkpoints.sqlite")
        )

    def _has_interrupted_checkpoint(self, graph: Any, config: Dict[str, Any]) -> bool:
        """当 ``graph`` 存在 ``config`` 线程的未完成检查点时返回 True。"""
        try:
            snap = graph.get_state(config)
            return bool(snap.values and snap.next)
        except Exception:  # noqa: BLE001 — 全新线程不应使运行崩溃
            return False

    def _write_graph_viz(self, graph: Any, run_dir: Path) -> None:
        """持久化父图及每个智能体子图的 Mermaid 源。

        使一次运行无需重跑即可可视化/审计。尽力而为：渲染失败绝不应中止调优运行。
        """
        try:
            (run_dir / "graph.mmd").write_text(
                graph.get_graph().draw_mermaid(), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001 — 可视化非关键
            log.debug("parent graph mermaid skipped: %s", exc)
        for name, agent in (self._agents or {}).items():
            try:
                target = run_dir / "agents" / f"{name}.mmd"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(agent.get_graph().draw_mermaid(), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                log.debug("agent %s mermaid skipped: %s", name, exc)

    def _log_state_change(
        self,
        node_name: str,
        before: Dict[str, Any],
        after: Dict[str, Any],
        run_dir: Path,
    ) -> None:
        """将 ``node_name`` 的顶层状态差异追加到 ``state_changes.jsonl``。

        仅记录发生变化的顶层快照键（不记录其值），形成“哪个阶段修改了哪些字段”
        的审计轨迹，同时避免将大载荷泄漏到日志中。
        """
        changed = sorted(
            key for key in set(before) | set(after) if before.get(key) != after.get(key)
        )
        if not changed:
            return
        append_jsonl(run_dir / "state_changes.jsonl", {
            "node": node_name,
            "iteration": after.get("iteration"),
            "changed_keys": changed,
            "timestamp": time.time(),
        })

    def _cleanup_checkpoints(self, run_dir_name: str, start_seq: int) -> None:
        """删除本运行已完成的检查点（尽力而为）。

        已完成的运行不再需要其父图线程及其生成的智能体子图线程，因此删除它们
        以防止两个 SQLite 检查点存储跨运行无界增长。*中断*的运行永远不会到达
        此处，因此其检查点会保留给 ``tune(resume_run_dir=...)``。
        """
        for db_path in ("run_checkpoints.sqlite", "checkpoints.sqlite"):
            try:
                checkpointer = build_checkpointer(self.output_root / db_path)
            except Exception:  # noqa: BLE001 — 清理绝不能打断运行
                continue
            delete_thread = getattr(checkpointer, "delete_thread", None)
            if delete_thread is None:
                continue
            if db_path == "run_checkpoints.sqlite":
                _safe_delete_thread(delete_thread, run_dir_name)
            else:
                for name in ("bottleneck_agent", "architecture_agent", "topology_agent",
                             "device_agent", "candidate_agent", "evaluate_agent"):
                    for seq in range(start_seq, self._thread_seq + 1):
                        _safe_delete_thread(delete_thread, f"{name}:{seq}")

    def _bind_context(
        self,
        output_dir: Path,
        evaluation: EvaluationResult,
        current_hardware: Dict[str, Any],
        evaluation_bases: Dict[str, EvaluationResult],
    ) -> None:
        self.run_state.context = ToolContext(
            output_dir=output_dir,
            evaluation=evaluation,
            current_hardware=current_hardware,
            search_space=self.simulator_schema.get("search_space", {}),
            evaluation_bases=evaluation_bases,
        )

    # -- per-stage pipeline nodes (the parent StateGraph body) ------------- #

    def _reset_stage_outputs(self, ctx: IterationContext) -> None:
        """在下一迭代开始前清空逐迭代的阶段输出。"""
        ctx.bottleneck_out = None
        ctx.arch_out = None
        ctx.topo_out = None
        ctx.device_out = None
        ctx.cand_out = None
        ctx.bottleneck_state = {}
        ctx.optical_analysis = {}
        ctx.cand_actions = []
        ctx.focused_actions = []
        ctx.focused_link = []
        ctx.deferred = []
        ctx.updated_hardware = {}
        ctx.applied = []
        ctx.failed = []
        ctx.stage_failed = False

    def _bind_ctx(self, ctx: IterationContext) -> None:
        """将（可检查点的）ctx 的核心字段绑定到工具上下文。

        在每个阶段开始时调用，使*恢复的*节点从恢复出的快照重新填充可变工具上下文，
        而不是读取陈旧的 ``self`` 状态。
        """
        self._current_iteration = ctx.iteration
        self._bind_context(
            output_dir=ctx.output_dir,
            evaluation=ctx.current,
            current_hardware=ctx.current.hardware,
            evaluation_bases={"current": ctx.current, "best": ctx.state.best},
        )
        tool_ctx = self.run_state.context
        tool_ctx.design_space = self.design_space
        # The no-RAG ablation keeps the scheduler/runtime operational but makes
        # case memory completely invisible to the model-facing tool surface.
        tool_ctx.case_store = self.store if self.enable_rag else None
        tool_ctx.pareto_archive = ctx.pareto
        tool_ctx.forbidden_fingerprints = set(ctx.forbidden)
        tool_ctx.simulator_adapter = self.adapter
        tool_ctx.iteration = ctx.iteration
        tool_ctx.schedule_config = self.scheduler_config
        try:
            precedent_stats = self.store.improved_case_stats()
        except (AttributeError, sqlite3.Error):
            precedent_stats = {"count": 0, "layers": []}
        tool_ctx.schedule_signals = ScheduleSignals(
            iteration=ctx.iteration,
            objective_history=[
                float(item.get("metrics", {}).get("objective", 0.0) or 0.0)
                for item in ctx.state.history
            ],
            focus_history=[
                (str(item.get("focus_layer")), bool(item.get("accepted")))
                for item in ctx.schedule_history
            ],
            improved_case_count=int(precedent_stats.get("count", 0)),
            improved_case_layers=list(precedent_stats.get("layers") or []),
            pipeline_evidence_ages=self._pipeline_evidence_ages(ctx),
            previous_mode=(str(ctx.schedule_history[-1].get("mode"))
                           if ctx.schedule_history else None),
            pipeline_count=sum(1 for item in ctx.schedule_history if item.get("mode") == "pipeline"),
            mode_history=[str(item.get("mode")) for item in ctx.schedule_history],
        )

    def _pipeline_evidence_ages(self, ctx: IterationContext) -> Dict[str, Optional[int]]:
        """Age of all three layers' evidence required to enter/resume pipeline mode.

        Architecture and topology retain their pipeline dependency offsets.
        Device readiness uses its latest completed output as a cross-layer baseline.
        """
        ages: Dict[str, Optional[int]] = {}
        for stage, offset in (("architecture", 1), ("topology", 2), ("device", 1)):
            record = self._evidence(ctx, stage, offset)
            if record is None:
                ages[stage] = None
            else:
                source_iteration = int(record.get("_iteration", ctx.iteration))
                ages[stage] = max(0, ctx.iteration - source_iteration)
        return ages

    # -- RAG 驱动的动态调度 ---------------------------------------------- #

    def _schedule_iteration(self, ctx: IterationContext) -> None:
        """在每轮开始时决定：这一轮让哪几个分析层下场、各自看多旧的证据。

        决策是**宿主侧确定性函数**（``runtime/scheduler.py``），不是模型输出：
        调度必须成为可复现的实验条件，否则第 5 章的消融无法成立。
        """
        if self.schedule_strategy == "fixed":
            # 消融对照：不读任何信号。``--search-strategy sequential`` 显式要求
            # 三层轮转，因此固定策略在这里映射为 serial；否则用指定模式。
            mode = "serial" if self._search_strategy_arg == "sequential" else self.fixed_schedule_mode
            decision = fixed_mode_decision(
                mode,
                focus=_SEQUENTIAL_LAYERS[(ctx.iteration - 1) % len(_SEQUENTIAL_LAYERS)],
            )
            if self._search_strategy_arg == "sequential":
                # sequential 仍执行三层分析以刷新证据，但候选编辑必须按层轮转；
                # fixed_mode_decision(serial) 的默认 joint 在这里改为本轮目标层。
                decision.focus_layer = _SEQUENTIAL_LAYERS[
                    (ctx.iteration - 1) % len(_SEQUENTIAL_LAYERS)
                ]
        else:
            try:
                precedent_stats = self.store.improved_case_stats()
            except (AttributeError, sqlite3.Error):
                precedent_stats = {"count": 0, "layers": []}
            signals = ScheduleSignals(
                iteration=ctx.iteration,
                objective_history=[float(item.get("metrics", {}).get("objective", 0.0) or 0.0)
                                   for item in ctx.state.history],
                focus_history=[
                    (str(item.get("focus_layer")), bool(item.get("accepted")))
                    for item in ctx.schedule_history
                ],
                improved_case_count=int(precedent_stats.get("count", 0)),
                improved_case_layers=list(precedent_stats.get("layers") or []),
                pipeline_evidence_ages=self._pipeline_evidence_ages(ctx),
                previous_mode=(str(ctx.schedule_history[-1].get("mode"))
                               if ctx.schedule_history else None),
                pipeline_count=sum(1 for item in ctx.schedule_history if item.get("mode") == "pipeline"),
                mode_history=[str(item.get("mode")) for item in ctx.schedule_history],
            )
            decision = schedule_decide(signals, self.scheduler_config)

        ctx.schedule = decision.as_dict()
        ctx.schedule_history.append({
            **ctx.schedule,
            "iteration": ctx.iteration,
            "accepted": None,  # 由 commit 阶段回填
        })
        write_json(ctx.output_dir / "schedule.json", ctx.schedule)
        log.info(
            "schedule iter %d: mode=%s focus=%s stages=%s — %s",
            ctx.iteration, decision.mode, decision.focus_layer,
            decision.stages_to_run, decision.reason,
        )

    def _scheduled(self, ctx: IterationContext, stage: str) -> bool:
        """本轮该分析阶段是否要运行。没有调度信息时默认运行（向后兼容）。"""
        if not ctx.schedule:
            return True
        stages = ctx.schedule.get("stages_to_run")
        if stages is None:
            return True
        return stage in stages

    def _record_evidence(self, ctx: IterationContext, stage: str, dump: Dict[str, Any]) -> None:
        """把本阶段本轮输出存入证据缓存，供后续轮次以偏移方式复用。

        ``_iteration`` 必须随输出一起存：下游要用它算出**真实的证据年龄**。
        早先漏存这一位，使 ``evidence_age`` 恒被算成 1、``evidence_iteration``
        恒为 None——流水线模式声称"容忍 1–2 轮陈旧"，但实际可能用了更旧的证据
        而不自知。
        """
        bucket = ctx.evidence_history.setdefault(stage, {})
        record = dict(dump)
        record["_iteration"] = ctx.iteration
        bucket[str(ctx.iteration)] = record

    def _evidence(self, ctx: IterationContext, stage: str, offset: int) -> Optional[Dict[str, Any]]:
        """取 ``stage`` 在 ``iteration - offset`` 轮（或最近可用轮）的输出。

        返回 ``None`` 表示该阶段还没有任何历史输出——此时调用方必须自己兜底，
        而不是假装有一份"新鲜证据"。
        """
        bucket = ctx.evidence_history.get(stage) or {}
        if not bucket:
            return None
        target = ctx.iteration - max(int(offset), 0)
        # 精确命中优先；否则退回到不晚于目标轮的最新一份，并接受"更旧"这件事。
        exact = bucket.get(str(target))
        if exact is not None:
            return exact
        candidates = sorted(
            ((int(it), dump) for it, dump in bucket.items() if int(it) <= target),
            key=lambda pair: pair[0],
        )
        if not candidates:
            return None
        return candidates[-1][1]

    def _carry_forward(self, ctx: IterationContext, stage: str) -> None:
        """跳过某阶段时，把它最近一次的输出接续到本轮上下文。

        跳过不等于"没有证据"：下游仍需上一轮的诊断才可能提出一致的编辑。差别
        在于它现在**明确是旧的**，且这个"旧"会写进 ``*_skipped.json`` 与载荷，
        让模型知道自己在基于陈旧证据推理，而不是误以为一切新鲜。
        """
        cached = self._evidence(ctx, stage, 1)
        age = None
        marker = {"stage": stage, "iteration": ctx.iteration, "ran": False}
        if cached is not None:
            age = ctx.iteration - int(cached.get("_iteration", ctx.iteration - 1))
            marker.update({"evidence_iteration": cached.get("_iteration"),
                           "evidence_age": age})
            payload = {k: v for k, v in cached.items() if k != "_iteration"}
            if stage == "architecture":
                ctx.arch_out = ArchitectureOutput.model_validate(payload)
                ctx.bottleneck_state = {
                    "primary_impact": ctx.arch_out.primary_impact,
                    "dominant_root_cause": ctx.arch_out.dominant_root_cause,
                    "recommended_focus": ctx.arch_out.recommended_focus,
                    "circuit_recommendations": list(
                        getattr(ctx.arch_out, "circuit_recommendations", None) or []
                    ),
                    "evidence_age": age,
                }
            elif stage == "topology":
                ctx.topo_out = TopologyOutput.model_validate(payload)
                ctx.optical_analysis = {
                    "loss_budget_diagnosis": ctx.topo_out.loss_budget_diagnosis,
                    "power_budget_margin": ctx.topo_out.power_budget_margin,
                    "power_budget": dict(getattr(ctx.topo_out, "power_budget", None) or {}),
                    "dominant_loss_contributors": ctx.topo_out.dominant_loss_contributors,
                    "evidence_age": age,
                }
            elif stage == "device":
                ctx.device_out = DeviceOutput.model_validate(payload)
        write_json(ctx.output_dir / f"{stage}_skipped.json", marker)

    def _run_bottleneck_stage(self, ctx: IterationContext) -> None:
        self._bind_ctx(ctx)
        ctx.state.iteration = ctx.iteration
        # 每轮的第一件事是决定"这轮谁下场"——必须在任何分析开始前定，否则
        # 各阶段已经在跑了再判断就失去了省成本的意义。
        self._schedule_iteration(ctx)
        objective = float(ctx.current.metrics.get("objective", 0.0) or 0.0)
        if objective > 0:
            ctx.objective_history.append(objective)
        bottleneck_out, model_report = self._invoke_stage(
            ctx, self._agents, "bottleneck", ctx.output_dir,
            bottleneck_payload(ctx.iteration, ctx.current.metrics, hardware_summary(ctx.current.hardware)),
            BottleneckOutput,
            {"metrics": ctx.current.metrics},
            ctx.rework_state,
        )
        if bottleneck_out is None:
            return
        write_json(ctx.output_dir / "bottleneck_result.json", bottleneck_out.model_dump())
        write_json(ctx.output_dir / "bottleneck_gate_report.json", model_report)
        self.run_state.context.model_analysis = {"candidate_layers": bottleneck_out.candidate_layers}
        ctx.bottleneck_out = bottleneck_out
        if self.analysis_mode == "one-shot":
            ctx.bottleneck_state = {
                "summary": bottleneck_out.summary,
                "candidate_layers": bottleneck_out.candidate_layers,
                "global_findings": bottleneck_out.global_findings,
            }
        ctx.top_layer_counts.append(max(len(bottleneck_out.candidate_layers), 1))
        self._record_evidence(ctx, "bottleneck", bottleneck_out.model_dump())

    def _run_architecture_stage(self, ctx: IterationContext) -> None:
        if ctx.stage_failed:
            return
        if self.analysis_mode == "one-shot":
            write_json(ctx.output_dir / "architecture_skipped.json", {
                "stage": "architecture", "iteration": ctx.iteration,
                "ran": False, "reason": "one-shot unclassified ablation",
            })
            return
        if not self._scheduled(ctx, "architecture"):
            self._carry_forward(ctx, "architecture")
            return
        self._bind_ctx(ctx)
        arch_out, arch_report = self._invoke_stage(
            ctx, self._agents, "architecture", ctx.output_dir,
            architecture_payload(ctx.bottleneck_out.candidate_layers, ctx.current.metrics),
            ArchitectureOutput,
            {
                "candidate_layers": ctx.bottleneck_out.candidate_layers,
                "metrics": ctx.current.metrics,
                # 架构闸门需要当前硬件点来判断"某个架构参数在该拓扑下是否真正
                # 生效"（见 topology_space.PARAM_SCOPE）；同时它也要据此校验推荐值
                # 是否属于枚举候选。缺少这些字段时闸门只能退化为无拓扑信息，
                # 要么漏检、要么误判。
                "hardware": ctx.current.hardware,
                "design_space": self.design_space,
                "mode": self._evaluable_mode,
            },
            ctx.rework_state,
        )
        if arch_out is None:
            return
        write_json(ctx.output_dir / "architecture_result.json", arch_out.model_dump())
        write_json(ctx.output_dir / "architecture_gate_report.json", arch_report)
        ctx.arch_out = arch_out
        ctx.bottleneck_state = {
            "primary_impact": arch_out.primary_impact,
            "dominant_root_cause": arch_out.dominant_root_cause,
            "recommended_focus": arch_out.recommended_focus,
            "circuit_recommendations": list(getattr(arch_out, "circuit_recommendations", None) or []),
        }
        self.run_state.context.bottleneck_state = ctx.bottleneck_state
        self._record_evidence(ctx, "architecture", arch_out.model_dump())

    def _run_topology_stage(self, ctx: IterationContext) -> None:
        if ctx.stage_failed:
            return
        if self.analysis_mode == "one-shot":
            write_json(ctx.output_dir / "topology_skipped.json", {
                "stage": "topology", "iteration": ctx.iteration,
                "ran": False, "reason": "one-shot unclassified ablation",
            })
            return
        if not self._scheduled(ctx, "topology"):
            self._carry_forward(ctx, "topology")
            return
        self._bind_ctx(ctx)
        # 流水线模式下拓扑层消费**上一轮**的架构证据：把偏移写进载荷与日志，
        # 使模型知道自己在读旧证据，而不是误以为与硬件现状同步。
        offset = int(ctx.schedule.get("evidence_offsets", {}).get("topology", 0) or 0)
        if offset > 0:
            stale = self._evidence(ctx, "architecture", offset)
            if stale is not None:
                actual_age = ctx.iteration - int(stale.get("_iteration", ctx.iteration - offset))
                payload_state = {k: v for k, v in stale.items() if k != "_iteration"}
                ctx.bottleneck_state = {
                    "primary_impact": payload_state.get("primary_impact"),
                    "dominant_root_cause": payload_state.get("dominant_root_cause"),
                    "recommended_focus": payload_state.get("recommended_focus"),
                    "circuit_recommendations": list(
                        payload_state.get("circuit_recommendations") or []
                    ),
                    "evidence_age": actual_age,
                }
        topo_out, topo_report = self._invoke_stage(
            ctx, self._agents, "topology", ctx.output_dir,
            topology_payload(ctx.bottleneck_state, ctx.current.metrics),
            TopologyOutput,
            {"bottleneck_state": ctx.bottleneck_state, "metrics": ctx.current.metrics,
             "hardware": ctx.current.hardware, "design_space": self.design_space,
             "mode": self._evaluable_mode},
            ctx.rework_state,
        )
        if topo_out is None:
            return
        write_json(ctx.output_dir / "topology_result.json", topo_out.model_dump())
        write_json(ctx.output_dir / "topology_gate_report.json", topo_report)
        ctx.topo_out = topo_out
        ctx.optical_analysis = {
            "loss_budget_diagnosis": topo_out.loss_budget_diagnosis,
            "power_budget_margin": topo_out.power_budget_margin,
            "power_budget": dict(getattr(topo_out, "power_budget", None) or {}),
            "dominant_loss_contributors": topo_out.dominant_loss_contributors,
        }
        self.run_state.context.optical_analysis = ctx.optical_analysis
        self._record_evidence(ctx, "topology", topo_out.model_dump())

    def _run_device_stage(self, ctx: IterationContext) -> None:
        if ctx.stage_failed:
            return
        if self.analysis_mode == "one-shot":
            write_json(ctx.output_dir / "device_skipped.json", {
                "stage": "device", "iteration": ctx.iteration,
                "ran": False, "reason": "one-shot unclassified ablation",
            })
            return
        if not self._scheduled(ctx, "device"):
            self._carry_forward(ctx, "device")
            return
        self._bind_ctx(ctx)
        offset = int(ctx.schedule.get("evidence_offsets", {}).get("device", 0) or 0)
        if offset > 0:
            stale = self._evidence(ctx, "topology", offset)
            if stale is not None:
                actual_age = ctx.iteration - int(stale.get("_iteration", ctx.iteration - offset))
                stale = {k: v for k, v in stale.items() if k != "_iteration"}
                ctx.optical_analysis = {
                    "loss_budget_diagnosis": stale.get("loss_budget_diagnosis"),
                    "power_budget_margin": stale.get("power_budget_margin"),
                    "power_budget": dict(stale.get("power_budget") or {}),
                    "dominant_loss_contributors": stale.get("dominant_loss_contributors"),
                    "evidence_age": actual_age,
                }
        device_out, device_report = self._invoke_stage(
            ctx, self._agents, "device", ctx.output_dir,
            device_payload(ctx.optical_analysis, ctx.current.metrics),
            DeviceOutput,
            {"topology_diagnosis": ctx.optical_analysis,
             "metrics": ctx.current.metrics},
            ctx.rework_state,
        )
        if device_out is None:
            return
        write_json(ctx.output_dir / "device_result.json", device_out.model_dump())
        write_json(ctx.output_dir / "device_gate_report.json", device_report)
        ctx.device_out = device_out
        self._record_evidence(ctx, "device", device_out.model_dump())

    def _recent_attempts(self, ctx: IterationContext, window: int = 3) -> List[Dict[str, Any]]:
        """最近几轮"改了什么、结果如何"，供候选智能体判断步长是否过大。

        诊断一个真实失败模式时发现的缺口：候选智能体每轮只看当前点，看不到
        "前三次都是把面积放大一倍、且都变差"。于是它持续提出跨 5–6 个参数的
        大幅改动，而传统基线只做单参数原子移动——这在乘性目标（面积是因子）
        下代价极大。
        """
        history = list(ctx.state.history)[-window:]
        attempts: List[Dict[str, Any]] = []
        for index, entry in enumerate(history):
            metrics = entry.get("metrics") or {}
            previous = history[index - 1].get("metrics") if index > 0 else None
            record: Dict[str, Any] = {
                "iteration": entry.get("iteration"),
                "accepted": bool(entry.get("is_best")),
                "objective": metrics.get("objective"),
                "latency_ms": metrics.get("latency_ms"),
                "energy_mj": metrics.get("energy_mj"),
                "area_mm2": metrics.get("area_mm2"),
                "applied": entry.get("applied") or entry.get("action"),
                # 无前驱时也保留该键（值为 None），让消费方拿到稳定的 schema。
                "delta_pct_vs_previous": None,
                "log_delta": None,
                "dominant_factor": None,
            }
            if previous:
                record["delta_pct_vs_previous"] = _pct_delta(
                    previous.get("objective"), metrics.get("objective")
                )
                # 目标是对数可加的乘积：Δlog J = Δlog L + Δlog E + Δlog A。
                # 把三项拆给模型看，它才能意识到"面积 −53% 但延时 +107% 是净亏"。
                decomposition = {
                    name: _log_delta(previous.get(name), metrics.get(name))
                    for name in ("latency_ms", "energy_mj", "area_mm2")
                }
                record["log_delta"] = {
                    k: (round(v, 4) if v is not None else None) for k, v in decomposition.items()
                }
                known = {k: v for k, v in decomposition.items() if v is not None}
                if known:
                    driver = max(known, key=lambda k: abs(known[k]))
                    record["dominant_factor"] = driver
            attempts.append(record)
        return attempts

    def _run_candidate_stage(self, ctx: IterationContext) -> None:
        if ctx.stage_failed:
            return
        self._bind_ctx(ctx)
        # 聚焦模式下某些分析层本轮没有运行；流水线模式下三层都会运行，
        # 但拓扑/器件层可能消费带偏移的历史证据。这里容忍缺失并显式携带证据年龄。
        topo_out = ctx.topo_out
        device_out = ctx.device_out
        optical: Dict[str, Any] = {}
        if topo_out is not None:
            optical.update({
                "loss_budget_diagnosis": topo_out.loss_budget_diagnosis,
                "power_budget_margin": topo_out.power_budget_margin,
                "power_budget": dict(getattr(topo_out, "power_budget", None) or {}),
                "dominant_loss_contributors": topo_out.dominant_loss_contributors,
                "recommended_template": topo_out.recommended_template,
                "circuit_recommendations": topo_out.circuit_recommendations,
            })
        if device_out is not None:
            optical["recommended_link_edits"] = device_out.recommended_link_edits
        if ctx.optical_analysis.get("evidence_age") is not None:
            optical["evidence_age"] = ctx.optical_analysis["evidence_age"]
        ctx.optical_analysis = optical
        # 通过框架标准的 Retriever 接口而非 store 私有的 ``search`` 方法检索；
        # 硬件通道按调用提供，因为它每轮迭代都会变化。
        retriever = None
        if self.enable_rag:
            retriever = self.store.as_history_retriever(
                top_k=self.top_k,
                min_score=self.min_retrieval_score,
                hardware_provider=lambda: ctx.current.hardware,
            )
        # 架构层本轮可能被跳过（串行/单层模式），此时用其最近一次输出里的
        # 检索描述；若连历史都没有，退化为瓶颈摘要——绝不把 None 交给检索器。
        query = None
        if ctx.arch_out is not None:
            query = getattr(ctx.arch_out, "retrieval_description", None)
        if not query:
            cached_arch = self._evidence(ctx, "architecture", 1) or {}
            query = cached_arch.get("retrieval_description")
        if not query:
            query = str((ctx.bottleneck_out.summary if ctx.bottleneck_out is not None else "") or "")
        if not self.enable_rag:
            summarized_cases = []
        elif retriever is None:  # pragma: no cover - 域运行中缺少 langchain
            cases = self.store.search(
                query, ctx.current.hardware,
                top_k=self.top_k, min_score=self.min_retrieval_score,
            )
            summarized_cases = summarize_cases(
                cases, improved_delta_pct=self.store.improved_delta_pct,
            )
        else:
            docs = retriever.invoke(query)
            summarized_cases = summarize_documents(
                docs, improved_delta_pct=self.store.improved_delta_pct,
            )
        # ρ 的信号源 = **记忆成熟度**，而不是检索相似度。
        # 依据：项目默认的哈希嵌入下，案例两两余弦实测可达 0.999，任何相似度阈值
        # 都会退化成"库非空"，从而把"检索到相似描述"误当成"有可用先验"。
        # 成熟度同时要求有效结局数量达标、且成功先例覆盖至少两个层级，
        # 避免单一层级上的重复成功被误当成完整跨层先验。
        try:
            precedent_stats = self.store.improved_case_stats()
        except (AttributeError, sqlite3.Error):
            precedent_stats = {"count": 0, "layers": []}
        target = max(int(self.scheduler_config.warm_precedent_target), 1)
        coverage_target = max(int(self.scheduler_config.minimum_mature_layer_coverage), 1)
        improved = int(precedent_stats.get("count", 0))
        improved_layers = list(precedent_stats.get("layers") or [])
        maturity = min(improved / target, len(set(improved_layers)) / coverage_target, 1.0)
        ctx.memory_maturity.append(maturity)
        log.debug(
            "memory maturity iter %d: improved_precedents=%d layers=%s -> %.3f",
            ctx.iteration, improved, improved_layers, maturity,
        )
        self.run_state.context.retrieved_cases = summarized_cases
        self.run_state.context.optical_analysis = ctx.optical_analysis
        # serial 模式允许联合提案；pipeline/focus 只提交本轮被调度的一层。
        if self.analysis_mode == "one-shot":
            ctx.focus_layer = "joint"
        elif ctx.schedule:
            ctx.focus_layer = str(ctx.schedule.get("focus_layer") or "joint")
        else:
            ctx.focus_layer = (
                "joint" if ctx.search_strategy == "joint"
                else _SEQUENTIAL_LAYERS[(ctx.iteration - 1) % len(_SEQUENTIAL_LAYERS)]
            )
        cand_payload = candidate_payload(
            bottleneck_state=ctx.bottleneck_state,
            optical_analysis=ctx.optical_analysis,
            retrieved_cases=summarized_cases,
            metrics=ctx.current.metrics,
            hardware_summary=hardware_summary(ctx.current.hardware),
            search_space=self.search_space,
            search_strategy=ctx.search_strategy,
            focus_layer=ctx.focus_layer,
            recent_attempts=self._recent_attempts(ctx),
        )
        cand_out = self._invoke(self._agents["candidate"], ctx.output_dir, cand_payload, CandidateOutput)
        cand_transcript = self._last_transcript
        # 模型可能返回缺字段的结构化响应（例如只给了 strategy 和 actions，没有
        # rationale）。此时 `_invoke` 返回的是未经校验的占位对象，直接解引用会抛
        # AttributeError，把整次运行截断在第 N 轮。正确语义是：**这一轮作废，
        # 当前点保持不变**，运行继续。
        try:
            _ = cand_out.strategy, cand_out.rationale, cand_out.actions
        except (AttributeError, TypeError) as exc:
            write_json(ctx.output_dir / "candidate_failed.json", {
                "iteration": ctx.iteration,
                "error": f"incomplete structured response: {exc}",
                "last_invoke_error": self._last_invoke_error,
                "note": "iteration discarded; incumbent kept unchanged",
            })
            log.warning(
                "iteration %d candidate output incomplete (%s); discarding this iteration",
                ctx.iteration, exc,
            )
            ctx.stage_failed = True
            if ctx.schedule_history:
                ctx.schedule_history[-1]["accepted"] = False
                ctx.schedule_history[-1]["output_error"] = str(exc)[:200]
            return
        cand_sig = json.dumps(cand_out.model_dump(), sort_keys=True, ensure_ascii=False)
        skip_judge = cand_sig == self._prev_stage_outputs.get("candidate")
        self._prev_stage_outputs["candidate"] = cand_sig
        cand_actions = [a.model_dump() for a in cand_out.actions]
        cand_link_edits = [e.model_dump() for e in cand_out.link_edits]

        focused_actions, focused_link = _filter_edits_for_layer(
            cand_actions, cand_link_edits, ctx.focus_layer, self.design_space
        )
        deferred = (
            []
            if ctx.focus_layer == "joint"
            else [a.get("name") for a in cand_actions + cand_link_edits
                  if _action_layers(a, self.design_space) != {ctx.focus_layer}]
        )
        # 提案粒度约束：一次评估只承载一个编辑假设。
        # 依据——目标是对数可加的乘积，多参数同时改会让各项相互抵消且无法归因；
        # 实测 agent 一次改 3–6 个参数（含把 core 16x16 改到 32x32，面积 ×4），
        # 而基线每次只动一格并直接评估，于是在同预算下 agent 反而更差。
        focused_actions, focused_link, granularity_deferred = _limit_edits(
            focused_actions, focused_link, self.max_edits_per_proposal
        )
        if granularity_deferred:
            deferred = list(deferred) + granularity_deferred

        gate_attempts: List[Dict[str, Any]] = []
        applied: List[str] = []
        failed: List[Dict[str, str]] = []
        updated_hardware = ctx.current.hardware
        for _attempt in range(self.max_gate_retries + 1):
            updated_hardware, applied, failed = self._apply_edits(
                ctx.current.hardware, focused_actions, focused_link, ctx.output_dir
            )
            if not self.enable_gate:
                break
            sol_stage_context = {
                "bottleneck_state": ctx.bottleneck_state,
                "optical_analysis": ctx.optical_analysis,
                "metrics": ctx.current.metrics,
                "current_core_kind": str(
                    (ctx.current.hardware.get("arch", {}).get("topology", {}) or {}).get(
                        "core_kind", "dptc_crossbar"
                    )
                ),
                "declared_edits": [a.get("name") for a in focused_actions] + [e.get("name") for e in focused_link],
                "applied_edits": applied,
                "failed_edits": failed,
                "deferred_edits": deferred,
                "focus_layer": ctx.focus_layer,
                "hardware": updated_hardware,
                "design_space": self.design_space,
                "mode": self._evaluable_mode,
            }
            report = self._run_stage_gate(
                "candidate", cand_out, cand_transcript, ctx.output_dir, sol_stage_context,
                skip_judge=skip_judge or _attempt > 0,
            )
            gate_attempts.append(report)
            decision = (report.get("decision") or {}).get("decision")
            self._run_metrics.record_gate("candidate_agent", decision)
            if decision in ("re-execute", "request-info") and _attempt < self.max_gate_retries:
                ctx.rework_state["count"] += 1
                cand_payload["gate_feedback"] = {
                    "decision": decision,
                    "reasons": (report.get("decision") or {}).get("reasons", []),
                }
                cand_out = self._invoke(
                    self._agents["candidate"], ctx.output_dir, cand_payload, CandidateOutput
                )
                cand_transcript = self._last_transcript
                self._prev_stage_outputs["candidate"] = json.dumps(cand_out.model_dump(), sort_keys=True, ensure_ascii=False)
                cand_actions = [a.model_dump() for a in cand_out.actions]
                cand_link_edits = [e.model_dump() for e in cand_out.link_edits]
                focused_actions, focused_link = _filter_edits_for_layer(
                    cand_actions, cand_link_edits, ctx.focus_layer, self.design_space
                )
                deferred = (
                    []
                    if ctx.focus_layer == "joint"
                    else [a.get("name") for a in cand_actions + cand_link_edits
                          if _action_layers(a, self.design_space) != {ctx.focus_layer}]
                )
                focused_actions, focused_link, granularity_deferred = _limit_edits(
                    focused_actions, focused_link, self.max_edits_per_proposal
                )
                if granularity_deferred:
                    deferred = list(deferred) + granularity_deferred
                continue
            break
        if gate_attempts:
            write_json(ctx.output_dir / "gate_report.json", {
                "attempts": gate_attempts,
                "final": gate_attempts[-1],
                "retried": len(gate_attempts) - 1,
            })

        ctx.rework_total += ctx.rework_state["count"]

        notes = dict(cand_out.notes)
        notes["applied_edits"] = applied
        if failed:
            notes["failed_edits"] = failed
        if deferred:
            notes["deferred_edits"] = deferred
        write_json(ctx.output_dir / "candidate_proposal.json", {
            "strategy": cand_out.strategy,
            "search_strategy": ctx.search_strategy,
            "focus_layer": ctx.focus_layer,
            "actions": [a.get("name") for a in cand_actions],
            "link_edits": cand_link_edits,
            "applied_actions": [a.get("name") for a in focused_actions],
            # 逐参数记录实际被应用/被推迟的编辑，便于审计"粒度约束是否生效"
            # 与后续分析（参数名比动作名细一级）。
            "applied_parameters": [
                str(u.get("parameter"))
                for a in focused_actions
                for u in ((a.get("arguments") or {}).get("updates") or [])
                if isinstance(u, dict)
            ],
            "applied_link_device_types": [
                str((link.get("arguments") or {}).get("device_type"))
                for link in focused_link if isinstance(link, dict)
            ],
            "applied_link_edits": focused_link,
            "deferred_edits": deferred,
            "rationale": cand_out.rationale,
            "coupling_rationale": cand_out.coupling_rationale,
            "notes": notes,
        })

        ctx.cand_out = cand_out
        ctx.cand_actions = cand_actions
        ctx.focused_actions = focused_actions
        ctx.focused_link = focused_link
        ctx.deferred = deferred
        ctx.updated_hardware = updated_hardware
        ctx.applied = applied
        ctx.failed = failed

    def _run_commit_stage(self, ctx: IterationContext) -> None:
        if ctx.stage_failed:
            # 本轮没有合法提案：不评估、不推进，也不消耗仿真预算。
            log.info("iteration %d has no valid proposal; incumbent kept unchanged", ctx.iteration)
            return
        next_hardware = _repair_duplicate(ctx.updated_hardware, ctx.forbidden, self.search_space)
        try:
            next_result = self._evaluate(next_hardware, ctx.iteration)
        except Exception as exc:  # noqa: BLE001 — 仿真器拒绝该点不等于整轮失败
            # 真实 SimPhony 会对物理/后端不可行的提案直接抛错（例如 CACTI 无法
            # 建模的缓存配置）。正确语义是"这个提案被拒绝、当前点保持不变"，
            # 而不是终止整个调优——否则一次坏提案会烧掉整轮实验预算。
            marker = {
                "iteration": ctx.iteration,
                "rejected_hardware": next_hardware,
                "error": str(exc)[:800],
                "note": "proposal rejected by the simulator; incumbent kept unchanged",
            }
            write_json(ctx.output_dir / "simulation_rejected.json", marker)
            log.warning(
                "iteration %d proposal rejected by the simulator: %s",
                ctx.iteration, str(exc).splitlines()[0][:200],
            )
            if ctx.schedule_history:
                ctx.schedule_history[-1]["accepted"] = False
                ctx.schedule_history[-1]["simulation_error"] = str(exc)[:200]
            return
        ctx.pareto.add(
            next_result.metrics,
            self._pareto_meta(next_result.hardware, ctx.iteration, ctx.state_strategy, ctx.search_strategy),
        )
        ctx.forbidden.add(hardware_fingerprint(next_hardware))

        is_best = self._is_better(next_result, ctx.state.best)
        if is_best:
            ctx.state.best = next_result
        # 回填调度历史里的 ``accepted``：下一轮 choose_focus_layer 据此决定
        # "继续挖这条有效的轴"还是轮换到下一层。
        if ctx.schedule_history:
            ctx.schedule_history[-1]["accepted"] = bool(is_best)
        ctx.state.history.append(
            self._history_entry(next_result, is_best=is_best, applied=ctx.applied)
        )

        description = None
        if ctx.arch_out is not None:
            description = getattr(ctx.arch_out, "retrieval_description", None)
        if not description:
            cached_arch = self._evidence(ctx, "architecture", 1) or {}
            description = cached_arch.get("retrieval_description") or ""
        self._commit_case(
            bottleneck_description=description,
            bottleneck_state=ctx.bottleneck_state,
            hardware=ctx.current.hardware,
            strategy=ctx.cand_out.strategy,
            actions=ctx.focused_actions,
            link_edits=ctx.focused_link,
            rationale=ctx.cand_out.rationale,
            before_metrics=ctx.current.metrics,
            after_metrics=next_result.metrics,
        )

        if ctx.state_strategy == "best" and ctx.state.best is not None:
            ctx.current = ctx.state.best
        else:
            ctx.current = next_result

    def _invoke(self, agent: Any, output_dir: Path, payload: Dict[str, Any], output_model: type):
        # 带 ``response_format`` 的原生 LangGraph ReAct 子图：智能体循环调用其工具
        # 收集证据，然后图在 ``structured_response`` 状态键中输出所请求的 Pydantic
        # 模型。此处不做散文到 schema 的强制转换。
        name = getattr(agent, "name", "?")
        cache = getattr(self, "_response_cache", None)
        metrics = getattr(self, "_run_metrics", None)

        # 相同的 (role, payload) 对由进程内缓存直接提供，完全跳过模型调用。
        cache_key = cache.key(name, payload) if cache is not None else None
        if cache_key is not None:
            cached = cache.get(cache_key)
            if cached is not None:
                self._last_transcript = cached["transcript"]
                self._last_invoke_error = cached["error"]
                if metrics is not None:
                    metrics.record_agent_call(name, 0.0, cached["error"], {})
                return cached["out"]

        # 对提示载荷做预算，使模型为工具结果保留余量。
        max_chars = int(getattr(self, "max_payload_chars", 24000))
        content = json.dumps(fit_payload(payload, max_chars), ensure_ascii=False)
        # 有界的工具循环。收集证据的角色在回答前会调用多个报告工具，因此预算须
        # 为这些轮次留出空间；限制过紧会使智能体在回答前耗尽轮次。
        invoke_config = None
        started = time.monotonic()
        try:
            retry_attempts = int(
                getattr(getattr(self, "llm_config", None), "retry_attempts", 0) or 0
            )
            invoke = llm_retry(retry_attempts)(agent.invoke)
            tracer = LLMTraceCollector(
                stream_tokens=bool(getattr(getattr(self, "llm_config", None), "stream", False))
            )
            self._thread_seq = getattr(self, "_thread_seq", 0) + 1
            invoke_config = {
                "recursion_limit": self.recursion_limit,
                "callbacks": [tracer],
                "configurable": {
                    # 每次调用唯一，使检查点器保存全新状态而不是恢复
                    # （从而绝不会把上一阶段的消息历史泄漏到本阶段）。
                    "thread_id": f"{name}:{self._thread_seq}",
                },
            }
            result = invoke(
                {"messages": [HumanMessage(content=content)]},
                config=invoke_config,
            )
            duration = time.monotonic() - started
            log.info("agent %s finished in %.2fs", name, duration)
        except Exception as exc:  # noqa: BLE001
            # GraphRecursionError（或任何运行时失败）必须降级为空记录，使阶段仍可被
            # 门控并返工，而不是中止整个运行。记录完整追踪以供排查。
            log.exception("agent %s invoke failed", name)
            self._last_invoke_error = f"{type(exc).__name__}: {exc}"
            duration = time.monotonic() - started
            # ``create_react_agent(response_format=...)`` may fail only in its
            # final formatting node after all evidence tools have completed.  In
            # that case the checkpoint still contains the useful ReAct transcript;
            # recover it instead of discarding minutes of real tool work.
            recovered_messages = []
            if invoke_config is not None and hasattr(agent, "get_state"):
                try:
                    checkpoint = agent.get_state(invoke_config)
                    recovered_messages = list(
                        ((getattr(checkpoint, "values", None) or {}).get("messages") or [])
                    )
                except Exception:  # noqa: BLE001 — recovery is best-effort
                    recovered_messages = []
            result = {"messages": recovered_messages}
        else:
            self._last_invoke_error = None
        self._last_transcript = result.get("messages", []) or self._last_transcript
        # 持久化每次调用的 LLM 耗时 + token 用量 + 调用树。
        # 在 ``flush`` 清空事件缓冲区之前计算。
        tree = tracer.build_tree()
        tokens = _aggregate_tokens(tracer.records)
        if getattr(self, "_trace_forest", None) is not None:
            self._trace_forest.append({"stage": name, "tree": tree})
        tracer.flush(output_dir / "llm_trace.jsonl")
        if metrics is not None:
            metrics.record_agent_call(name, duration, self._last_invoke_error, tokens)

        out = result.get("structured_response")
        if out is None and self._last_transcript:
            # Some OpenAI-compatible gateways occasionally ignore the forced
            # response-schema tool in LangGraph's final formatting node and emit
            # the name of a previously used evidence tool instead.  Re-run only
            # the formatting step against a model that has *no* evidence tools
            # bound, so the requested Pydantic schema is the sole available tool.
            transcript = []
            for message in self._last_transcript[-24:]:
                transcript.append({
                    "type": getattr(message, "type", type(message).__name__),
                    "content": getattr(message, "content", ""),
                    "tool_calls": getattr(message, "tool_calls", None),
                    "name": getattr(message, "name", None),
                })
            formatter_payload = fit_payload({
                "instruction": (
                    f"Return the final {output_model.__name__} object from the completed "
                    "evidence transcript. Do not request another evidence tool."
                ),
                "original_task": payload,
                "transcript": transcript,
            }, max_chars)
            formatter_tracer = LLMTraceCollector(
                stream_tokens=bool(getattr(getattr(self, "llm_config", None), "stream", False))
            )
            formatter_started = time.monotonic()
            formatter_error = None
            try:
                formatter = self.model.with_structured_output(output_model)
                out = llm_retry(retry_attempts)(formatter.invoke)(
                    [HumanMessage(content=json.dumps(formatter_payload, ensure_ascii=False))],
                    config={"callbacks": [formatter_tracer]},
                )
            except Exception as exc:  # noqa: BLE001 — normal degradation path below
                formatter_error = f"{type(exc).__name__}: {exc}"
                log.exception("agent %s fallback formatter failed", name)
            formatter_duration = time.monotonic() - formatter_started
            formatter_tree = formatter_tracer.build_tree()
            formatter_tokens = _aggregate_tokens(formatter_tracer.records)
            if getattr(self, "_trace_forest", None) is not None:
                self._trace_forest.append({"stage": f"{name}_formatter", "tree": formatter_tree})
            formatter_tracer.flush(output_dir / "llm_trace.jsonl")
            if metrics is not None:
                metrics.record_agent_call(
                    f"{name}_formatter", formatter_duration, formatter_error, formatter_tokens
                )
            if out is not None:
                self._last_invoke_error = None
                log.info("agent %s recovered structured output via fallback formatter", name)
            elif formatter_error:
                self._last_invoke_error = (
                    f"{self._last_invoke_error}; fallback formatter: {formatter_error}"
                    if self._last_invoke_error else f"fallback formatter: {formatter_error}"
                )
        if out is None:
            # 无可用的结果：返回 schema 默认实例，让确定性门控判定阶段失败，
            # 以便返工循环重试。
            self._last_invoke_error = (
                self._last_invoke_error
                or f"agent {name} produced no structured response"
            )
            log.warning("agent %s produced no structured response; gate will rework", name)
            # 占位：跳过验证以便构造，第 5.7 行（以及确定性各行）将判定阶段失败
            # 以便返工。
            out = output_model.model_construct()
        if cache is not None:
            cache.put(cache_key, out, self._last_transcript, self._last_invoke_error)
        return out

    def _apply_edits(
        self,
        hardware: Dict[str, Any],
        actions: List[Dict[str, Any]],
        link_edits: List[Dict[str, Any]],
        output_dir: Path,
    ):
        context = ToolContext(
            output_dir=output_dir,
            current_hardware=materialize_hardware(hardware),
            search_space=self.simulator_schema.get("search_space", {}),
        )
        applied: List[str] = []
        failed: List[Dict[str, str]] = []
        for action in actions:
            name = _edit_tool_name(action)
            if name not in EDIT_TOOLS:
                failed.append({"name": name or "<empty>", "error": "unknown tool"})
                continue
            self._run_edit(self.analysis_toolbox, name, (action or {}).get("arguments", {}), context, applied, failed)
        for edit in link_edits:
            name = _edit_tool_name(edit)
            if name not in LINK_EDIT_TOOLS:
                failed.append({"name": name or "<empty>", "error": "unknown tool"})
                continue
            self._run_edit(self.device_toolbox, name, (edit or {}).get("arguments", {}) or {}, context, applied, failed)
        return materialize_hardware(context.current_hardware), applied, failed

    def _run_edit(self, toolbox, name, arguments, context, applied, failed):
        try:
            toolbox.run(name, context, arguments=arguments or {})
            applied.append(name)
        except (ValueError, KeyError, TypeError) as exc:
            failed.append({"name": name, "error": str(exc)})

    def _invoke_judge(self, role: str, out: Any, output_dir: Path, stage_context: Dict[str, Any]) -> EvaluateOutput:
        """调用只读 evaluate 智能体，用于语义第 2/3/6/7 行。"""
        payload = _json_safe(evaluate_payload(
            role, STAGE_CHECKLISTS.get(role, []), out.model_dump(), stage_context
        ))
        return self._invoke(self._agents["evaluate"], output_dir, payload, EvaluateOutput)

    def _run_stage_gate(
        self,
        role: str,
        out: Any,
        transcript: Optional[List[Any]],
        output_dir: Path,
        stage_context: Dict[str, Any],
        skip_judge: bool = False,
    ) -> Dict[str, Any]:
        """确定性各行 + 结构合法性 + 可选语义评判 + 第 8 行。

        评判失败绝不能使运行崩溃：发生异常时门控记录 ``judge_error`` 检查，
        并让第 8 行优雅降级。
        """
        checks = deterministic_checks(
            role, out, messages=transcript,
            applied_edits=stage_context.get("applied_edits"),
            failed_edits=stage_context.get("failed_edits"),
            declared_edits=stage_context.get("declared_edits"),
        )
        if role in STRUCTURE_LEGALITY_ROLES:
            checks.append(structure_legality_check(role, out, stage_context))
        if role == "architecture":
            checks.append(architecture_legality_check(out, stage_context))
        if role == "candidate":
            checks.append(topology_binding_check(role, out, stage_context))
        checks.append(evidence_completeness_check(role, out, stage_context))
        # 降级的调用（智能体耗尽轮次/抛出异常）必须在审计轨迹中可见，并必须判定
        # 阶段失败，而不是静默产出看似真实答案的默认填充输出。
        if self._last_invoke_error:
            checks.append({
                "row": 5.7,
                "name": "agent_completion",
                "status": "fail",
                "detail": self._last_invoke_error,
            })

        verdict: Optional[EvaluateOutput] = None
        if not skip_judge and all(c.get("status") == "pass" for c in checks):
            try:
                verdict = self._invoke_judge(role, out, output_dir, stage_context)
            except Exception as exc:  # noqa: BLE001 — 评判必须非致命
                checks.append({"row": "judge", "name": "judge_error", "status": "fail", "detail": str(exc)})

        decision = decide(checks, verdict)
        return {
            "checks": checks,
            "judge": verdict.model_dump() if verdict is not None else None,
            "decision": decision,
        }

    def _invoke_stage(
        self,
        ctx: IterationContext,
        agents: Dict[str, Any],
        role: str,
        output_dir: Path,
        payload: Dict[str, Any],
        output_model: type,
        stage_context: Dict[str, Any],
        rework_state: Dict[str, int],
    ):
        """调用一个角色智能体，运行其检查，被拒绝时返工一次。

        强制每阶段预算：每阶段至多一次语义评判（被拒绝的返工仅再次做确定性检查），
        以及每轮返工预算（``rework_state["count"] < self.max_stage_rework``）。
        跳过未变化：与上一迭代相同的输出跳过评判（确定性各行仍会运行——它们开销为零）。
        """
        out = self._invoke(agents[role], output_dir, payload, output_model)
        # 记录阶段执行次数：κ 的分母。用 LLM 调用总数会混入证据门调用，
        # 把返工率稀释一个数量级（实测 52% 被算成 ~5%）。计数存在 ctx 里，
        # 随检查点持久化，恢复运行后 κ 不会跳变。
        ctx.stage_invocations += 1
        transcript = self._last_transcript
        sig = json.dumps(out.model_dump(), sort_keys=True, ensure_ascii=False)
        skip_judge = sig == self._prev_stage_outputs.get(role)
        self._prev_stage_outputs[role] = sig

        report = self._run_stage_gate(role, out, transcript, output_dir, stage_context, skip_judge=skip_judge)
        decision = (report.get("decision") or {}).get("decision")
        self._run_metrics.record_gate(getattr(agents[role], "name", role), decision)
        if decision in ("re-execute", "request-info") and rework_state["count"] < self.max_stage_rework:
            # 在重新调用前保留首次尝试的报告——包括语义评判的结论——使触发本次返工的
            # 评估留在审计轨迹中（最终报告以 skip_judge=True 重跑，否则会丢失评判结论）。
            write_json(output_dir / f"{role}_gate_report_first_attempt.json", report)
            rework_state["count"] += 1
            payload = dict(payload)
            payload["gate_feedback"] = {
                "decision": decision,
                "reasons": (report.get("decision") or {}).get("reasons", []),
            }
            out = self._invoke(agents[role], output_dir, payload, output_model)
            transcript = self._last_transcript
            self._prev_stage_outputs[role] = json.dumps(out.model_dump(), sort_keys=True, ensure_ascii=False)
            report = self._run_stage_gate(role, out, transcript, output_dir, stage_context, skip_judge=True)
            self._run_metrics.record_gate(
                getattr(agents[role], "name", role), (report.get("decision") or {}).get("decision")
            )
        required_fields = [
            field_name
            for field_name, field_info in output_model.model_fields.items()
            if field_info.is_required()
        ]
        missing_fields = [
            field_name for field_name in required_fields if not hasattr(out, field_name)
        ]
        if missing_fields:
            error = (
                f"{role} produced no complete {output_model.__name__}; "
                f"missing required fields: {', '.join(missing_fields)}"
            )
            self._last_invoke_error = self._last_invoke_error or error
            ctx.stage_failed = True
            if ctx.schedule_history:
                ctx.schedule_history[-1]["accepted"] = False
                ctx.schedule_history[-1]["output_error"] = error[:200]
            write_json(output_dir / f"{role}_failed.json", {
                "iteration": ctx.iteration,
                "error": error,
                "last_invoke_error": self._last_invoke_error,
                "note": "iteration discarded; incumbent kept unchanged",
            })
            log.warning("iteration %d %s", ctx.iteration, error)
            return None, report
        return out, report

    def _commit_case(self, bottleneck_description, bottleneck_state, hardware, strategy, actions, link_edits, rationale, before_metrics, after_metrics):
        if not self.enable_rag:
            return
        layers: Set[str] = set()
        for action in actions or []:
            if isinstance(action, dict):
                layers.update(_action_layers(action, self.design_space) & _SCALAR_EDIT_LAYERS)
        if link_edits:
            layers.add("device")
        self.store.add(
            bottleneck_description=bottleneck_description,
            hardware=hardware,
            solution={"strategy": strategy, "actions": actions, "link_edits": link_edits, "rationale": rationale},
            metrics={"before": before_metrics, "after": after_metrics},
            metadata={"iteration": self._iteration_label(), "layers": sorted(layers)},
            bottleneck_state=bottleneck_state,
        )

    def _iteration_label(self) -> int:
        return int(self._current_iteration)

    def _history_entry(
        self,
        result: EvaluationResult,
        is_best: bool,
        applied: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """一条历史记录：这轮评了什么、结果如何、**改了什么**。

        ``applied`` 必须记下来：只给下游"变差了"而不给"因为改了什么"，
        候选智能体就无法从失败里学到任何东西（``recent_attempts.edit`` 曾恒为
        None，正是这个缺口）。
        """
        return {
            "iteration": self._iteration_label(),
            "metrics": result.metrics,
            "hardware_fingerprint": hardware_fingerprint(result.hardware),
            "is_best": is_best,
            "run_dir": os.path.relpath(result.run_dir, self.output_root),
            "applied": list(applied or []),
        }

    def _next_run_dir(self) -> Path:
        prefix = "run_iteration_"
        max_index = 0
        if self.output_root.exists():
            for child in self.output_root.iterdir():
                if child.is_dir() and child.name.startswith(prefix):
                    try:
                        max_index = max(max_index, int(child.name[len(prefix):]))
                    except ValueError:
                        continue
        return self.output_root / f"{prefix}{max_index + 1:03d}"

    def _evaluate(self, hardware: Dict[str, Any], iteration: int) -> EvaluationResult:
        hardware = self.design_space.normalize(hardware)
        result = self.adapter.evaluate(hardware, iteration)
        result.metrics = metrics_with_objective(result.metrics, result.hardware, self.search_space)
        return result

    def _pareto_meta(self, hardware: Dict[str, Any], iteration: int, strategy: str, search_strategy: str = "joint") -> Dict[str, Any]:
        """为归档到 Pareto 前沿的设计点生成范式/策略标签。"""
        topo = (hardware or {}).get("arch", {}).get("topology", {}) or {}
        return {
            "iteration": iteration,
            "core_kind": topo.get("core_kind", "dptc_crossbar"),
            "node_kind": topo.get("node_kind", "dode"),
            "strategy": strategy,
            "search_strategy": search_strategy,
        }

    def _is_better(self, candidate: EvaluationResult, best: Optional[EvaluationResult]) -> bool:
        if best is None:
            return True
        return float(candidate.metrics.get("objective", float("inf"))) < float(
            best.metrics.get("objective", float("inf"))
        )


def _repair_duplicate(
    hardware: Dict[str, Any],
    forbidden_fingerprints: Set[str],
    search_space: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """通过步进一个合法架构参数来逃离重复设计。

    步进顺序覆盖完整的可调架构表面（``ARCH_PARAM_KEYS``）而非手工挑选的子集，
    因此新的搜索空间参数会自动被重复修复回退覆盖。
    """
    candidate = materialize_hardware(hardware)
    if hardware_fingerprint(candidate) not in forbidden_fingerprints:
        return candidate
    for parameter in ARCH_PARAM_KEYS:
        for direction in (1, -1):
            try:
                stepped = step_arch_parameter(candidate, parameter, direction, search_space)
            except (ValueError, KeyError):
                continue
            if hardware_fingerprint(stepped) not in forbidden_fingerprints:
                return stepped
    return candidate
