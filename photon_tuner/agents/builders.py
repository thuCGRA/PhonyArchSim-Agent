"""面向原生 LangGraph 运行时的角色智能体构建器。

每个构建器组装一个已编译的 ReAct 智能体（``langgraph.prebuilt
.create_react_agent``），带有角色 ``system_prompt``、白名单工具面以及原生
结构化输出 ``response_format``。智能体通过工具循环收集证据，随后直接输出
所请求的 Pydantic 模型；编排器从 ``structured_response`` 状态键读取结果。

宿主（``runtime/orchestrator.py``）掌控五个阶段的*顺序*；每个阶段都是一个
已编译的 LangGraph 子图，由编排器确定性调用。
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from photon_tuner.agents.outputs import (
    ArchitectureOutput,
    DeviceOutput,
    EvaluateOutput,
    BottleneckOutput,
    CandidateOutput,
    TopologyOutput,
)
from photon_tuner.agents.prompts import (
    ARCHITECTURE_PROMPT,
    DEVICE_PROMPT,
    EVALUATE_PROMPT,
    BOTTLENECK_PROMPT,
    CANDIDATE_PROMPT,
    TOPOLOGY_PROMPT,
)

# 各角色的工具白名单（按名称）。光学层面划分为 topology 与 device；
# solution 智能体只读，仅提出由编排器确定性应用的编辑。
BOTTLENECK_TOOLS = [
    "summarize_hardware_config",
    "summarize_metrics",
    "summarize_energy_breakdown",
    "summarize_area_breakdown",
    "summarize_insertion_loss",
    "summarize_layer_ranks",
    "summarize_operator_groups",
    "compare_search_states",
    "select_analysis_base",
    "describe_metric_units",
    "summarize_workload_features",
    "summarize_pareto_frontier",
    "summarize_schedule_signals",
    "assess_memory_energy_dominance",
    "identify_latency_hotspot",
    "identify_dominant_energy_device",
    "identify_dominant_area_device",
    "choose_analysis_baseline",
    "assess_objective_improvement",
    "assess_objective_convergence",
]

ARCHITECTURE_TOOLS = [
    "inspect_layer_details",
    "summarize_layer_ranks",
    "summarize_operator_groups",
    "summarize_hardware_config",
    "summarize_metrics",
    "compare_search_states",
    "select_analysis_base",
    # architecture 阶段负责把成本归因到根因（计算/存储/器件），并据此衡量实测
    # 占比，因此需要能量/面积分解——包括能识别实际主导组件的按器件排名。
    "summarize_energy_breakdown",
    "summarize_area_breakdown",
    "summarize_device_energy_breakdown",
    "summarize_device_area_breakdown",
    "summarize_insertion_loss",
    # 部分架构参数在非 crossbar 拓扑下不生效（被 mesh_rank 取代 / 物理固定）。
    # 推荐前先查，避免提出改了等于没改的编辑。
    "searchable_parameters",
    "summarize_search_space",
    "summarize_workload_features",
    "summarize_pareto_frontier",
    "assess_memory_energy_dominance",
    "identify_latency_hotspot",
    "identify_dominant_energy_device",
    "identify_dominant_area_device",
    "choose_analysis_baseline",
    "assess_objective_improvement",
    "assess_parameter_effectiveness",
]

TOPOLOGY_TOOLS = [
    "summarize_optical_structure",
    "summarize_insertion_loss_detail",
    "summarize_insertion_loss",
    "summarize_device_energy_breakdown",
    "summarize_device_area_breakdown",
    "summarize_topology",
    "inspect_light_path",
    "list_topology_templates",
    "inspect_topology_template",
    "validate_coupling",
    # topology 提示词要求检查激光器（功率）与光电探测器（灵敏度）以计算光链路
    # 余量，因此该工具必须在 topology 的工具面内。
    "inspect_device_config",
    # 单次调用打包损耗归因 + 器件变体 + 激光器/PD 电平，作为首选入口。
    "summarize_topology_evidence",
    # 切换模板会改变哪些参数生效（如网格下 mesh_rank 接管矩阵维度），
    # 推荐前需据此核对。
    "searchable_parameters",
    "summarize_device_provenance",
    "assess_optical_power_budget",
    "identify_dominant_optical_loss",
    "assess_parameter_effectiveness",
    "assess_candidate_legality",
]

DEVICE_TOOLS = [
    "list_device_candidates",
    "inspect_device_config",
    "summarize_device_frontier",
    "summarize_device_energy_breakdown",
    "summarize_device_area_breakdown",
    "summarize_insertion_loss_detail",
    "summarize_device_provenance",
    "identify_dominant_energy_device",
    "identify_dominant_area_device",
    "identify_dominant_optical_loss",
    "select_device_frontier_candidate",
]

# solution 智能体只读，主要依据载荷中的上游结论提出 actions / link_edits；
# 它仅需最小工具面在提出前复核当前状态。保持精简可避免工具循环，编排器确定性
# 应用所提编辑（并对单个编辑的失败做容错处理）。
CANDIDATE_TOOLS = [
    "searchable_parameters",
    "summarize_hardware_config",
    "summarize_metrics",
    "compare_search_states",
    "list_topology_templates",
    "inspect_topology_template",
    "validate_coupling",
    "summarize_design_identity",
    "retrieve_similar_cases",
    "summarize_case_memory",
    "summarize_pareto_frontier",
    "assess_objective_improvement",
    "assess_parameter_effectiveness",
    "assess_candidate_legality",
    "assess_design_novelty",
    "assess_case_reusability",
    "assess_memory_maturity",
    "assess_objective_convergence",
]

# evaluate 智能体严格只读：用真实证据核验单个上游阶段的输出，因此拥有宽泛的
# inspect/summarize 工具面，但不含编辑工具（编辑工具本就不暴露给任何智能体循环）。
EVALUATE_TOOLS = [
    "summarize_hardware_config",
    "summarize_metrics",
    "summarize_energy_breakdown",
    "summarize_area_breakdown",
    "summarize_insertion_loss",
    "summarize_insertion_loss_detail",
    "summarize_layer_ranks",
    "summarize_operator_groups",
    "summarize_optical_structure",
    "summarize_device_energy_breakdown",
    "summarize_device_area_breakdown",
    "summarize_topology",
    "inspect_light_path",
    "list_topology_templates",
    "inspect_topology_template",
    "validate_coupling",
    "list_device_candidates",
    "inspect_device_config",
    "compare_search_states",
    "select_analysis_base",
    "summarize_search_space",
    "describe_metric_units",
    "summarize_workload_features",
    "summarize_device_provenance",
    "preview_evaluation_schema",
    "summarize_design_identity",
    "retrieve_similar_cases",
    "summarize_case_memory",
    "summarize_pareto_frontier",
    "summarize_schedule_signals",
    "assess_memory_energy_dominance",
    "identify_latency_hotspot",
    "identify_dominant_energy_device",
    "identify_dominant_area_device",
    "assess_optical_power_budget",
    "identify_dominant_optical_loss",
    "choose_analysis_baseline",
    "assess_objective_improvement",
    "select_device_frontier_candidate",
    "assess_parameter_effectiveness",
    "assess_candidate_legality",
    "assess_design_novelty",
    "assess_case_reusability",
    "assess_memory_maturity",
    "assess_objective_convergence",
]

# Single-agent ablation: one generalist policy replaces all role-specific
# personas.  Keep each stage's original tool interface so the ablation changes
# agent specialization only, rather than confounding it with a much larger
# prompt/tool surface.  Separate response-schema adapters are still
# compiled because LangGraph fixes ``response_format`` at graph construction;
# they are adapters for the same generalist agent, not specialized roles.
GENERALIST_PROMPT = """
You are a single generalist agent for photonic-accelerator design-space
exploration.  There are no specialized bottleneck, architecture, topology,
device, candidate, or evaluator personas.  For every request, inspect the
provided payload, use any available read-only tool needed to ground the answer,
and return exactly the structured schema requested by the caller.  Preserve
explicit architecture/topology/device fields when the schema contains them,
but reason as one unified agent.  Never invent simulator evidence, device
variants, parameter names, or legal values.
""".strip()


def _select(all_tools: Dict[str, BaseTool], names: List[str]) -> List[BaseTool]:
    return [all_tools[name] for name in names if name in all_tools]


def _build(
    model: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: str,
    output_model: type,
    name: str,
    checkpointer: Any = None,
):
    """将一个角色智能体编译为原生 LangGraph ReAct 子图。

    ``response_format`` 使图在工具循环结束后，通过模型自身的
    ``with_structured_output`` 路径，将所请求的 Pydantic 模型输出到
    ``structured_response`` 状态键。

    ``checkpointer``（可选）附加 LangGraph 检查点器，使每次调用的最终状态按
    ``thread_id`` 持久化（见 ``runtime/orchestrator.py::_invoke``）；为 ``None``
    时保持默认的内存图行为。
    """
    kwargs: Dict[str, Any] = {}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    # 将角色提示词作为类型化的 ``SystemMessage`` 传入（而非裸字符串）。不采用
    # ``ChatPromptTemplate``：create_react_agent 会把 BasePromptTemplate 当作完整
    # 消息生成器，而 str/SystemMessage 则作为系统提示词前置。
    system_message = SystemMessage(content=system_prompt)
    return create_react_agent(
        model=model,
        tools=tools,
        prompt=system_message,
        response_format=output_model,
        name=name,
        **kwargs,
    )


def build_bottleneck_agent(model, all_tools, checkpointer=None):
    return _build(model, _select(all_tools, BOTTLENECK_TOOLS), BOTTLENECK_PROMPT, BottleneckOutput, "bottleneck_agent", checkpointer)


def build_architecture_agent(model, all_tools, checkpointer=None):
    return _build(model, _select(all_tools, ARCHITECTURE_TOOLS), ARCHITECTURE_PROMPT, ArchitectureOutput, "architecture_agent", checkpointer)


def build_topology_agent(model, all_tools, checkpointer=None):
    return _build(model, _select(all_tools, TOPOLOGY_TOOLS), TOPOLOGY_PROMPT, TopologyOutput, "topology_agent", checkpointer)


def build_device_agent(model, all_tools, checkpointer=None):
    return _build(model, _select(all_tools, DEVICE_TOOLS), DEVICE_PROMPT, DeviceOutput, "device_agent", checkpointer)


def build_candidate_agent(model, all_tools, checkpointer=None):
    return _build(model, _select(all_tools, CANDIDATE_TOOLS), CANDIDATE_PROMPT, CandidateOutput, "candidate_agent", checkpointer)


def build_evaluate_agent(model, all_tools, checkpointer=None):
    return _build(model, _select(all_tools, EVALUATE_TOOLS), EVALUATE_PROMPT, EvaluateOutput, "evaluate_agent", checkpointer)


def build_generalist_agent_adapters(model, all_tools, checkpointer=None):
    """Build schema adapters for the single-generalist-agent ablation."""
    specs = {
        "bottleneck": (BottleneckOutput, BOTTLENECK_TOOLS),
        "architecture": (ArchitectureOutput, ARCHITECTURE_TOOLS),
        "topology": (TopologyOutput, TOPOLOGY_TOOLS),
        "device": (DeviceOutput, DEVICE_TOOLS),
        "candidate": (CandidateOutput, CANDIDATE_TOOLS),
        "evaluate": (EvaluateOutput, EVALUATE_TOOLS),
    }
    return {
        role: _build(
            model,
            _select(all_tools, tool_names),
            GENERALIST_PROMPT,
            output_model,
            f"single_generalist_{role}_adapter",
            checkpointer,
        )
        for role, (output_model, tool_names) in specs.items()
    }
