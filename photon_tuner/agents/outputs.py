"""面向 LangGraph 角色智能体的 Pydantic 结构化输出模型。

这些是*提取*目标：每个角色智能体先写出纯文本分析，编排器再通过一次
``with_structured_output`` 调用（``orchestrator._invoke``）把该文本强转为其中
一个模型。字段形态与 ``core/schemas.py`` 中的数据类保持一致，以维持领域契约。
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ModelOutput(BaseModel):
    """模型级（整体架构）分析输出。"""

    candidate_layers: List[Dict[str, Any]] = Field(
        description="Bottleneck candidate layers ranked by global objective impact."
    )
    summary: str = Field(description="One-paragraph summary of the analysis.")
    global_findings: List[str] = Field(default_factory=list)
    selected_views: List[str] = Field(default_factory=list)
    analysis_base: Dict[str, Any] = Field(default_factory=dict)
    llm_notes: Dict[str, Any] = Field(default_factory=dict)


class ArchitectureOutput(BaseModel):
    """架构级（逐层根因）瓶颈诊断。"""

    primary_impact: str = Field(description="Dominant impact: latency / energy / area.")
    dominant_root_cause: str = Field(
        description="Dominant root cause: compute / memory / communication / buffer / device."
    )
    layer_diagnoses: List[Dict[str, Any]] = Field(
        description="Per-layer diagnosis with ratios and impact types."
    )
    retrieval_description: str = Field(
        description="Compact description used to retrieve similar historical cases."
    )
    root_cause_summary: Dict[str, float] = Field(
        description="Aggregated root-cause fractions summing to ~1.0, taken from the "
                    "measured ranked energy/area evidence (e.g. the off-chip laser's "
                    "share when mesh insertion loss sets the optical power budget). "
                    "Do not assign a large fraction to a component whose measured "
                    "share is negligible."
    )
    recommended_focus: List[str] = Field(default_factory=list)
    circuit_recommendations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Concrete architecture-level (L2/L3) lever moves, each as "
                    "{'parameter', 'value', 'rationale'} with a parameter drawn from the "
                    "design space (tiles, cores_per_tile, dataflow, precision_in_bit, "
                    "precision_w_bit, precision_out_bit, hbm_bandwidth, glb1_cache_size, "
                    "glb2_cache_size, rf_cache_size, core_height, core_width, "
                    "num_wavelength, work_freq, laser_source, splitter_kind) and 'value' "
                    "one of that parameter's listed candidates. Emit [] only when the "
                    "diagnosis genuinely supports no architecture-level move.",
    )
    llm_notes: Dict[str, Any] = Field(default_factory=dict)


class TopologyOutput(BaseModel):
    """拓扑/电路结构分析：光链路预算。"""

    loss_budget_diagnosis: str = Field(
        description="Where insertion loss concentrates along the longest light paths."
    )
    dominant_loss_contributors: List[Dict[str, Any]] = Field(
        description="Dominant loss contributors, one entry per component, taken from "
        "summarize_insertion_loss_detail's node_path_detail/core_path_detail: "
        "{instance, device_type, device_name, insertion_loss_dB}, ordered by descending dB. "
        "Populate with real values from the tool result — never a placeholder such as "
        "'not exposed'."
    )
    power_budget_margin: str = Field(
        description="Optical link margin as a computed number: laser output power (dBm, from "
        "inspect_device_config) minus the cumulative path insertion loss, compared against the "
        "photodetector sensitivity (dBm). Cite the actual cfg values. Do not report it as "
        "'cannot be stated' and do not confuse it with the objective's insertion-loss budget."
    )
    power_budget: Dict[str, float] = Field(
        default_factory=dict,
        description="Structured power budget in dB. Required keys: laser_output_dbm "
        "(laser 'power' from inspect_device_config), path_loss_db (core_insertion_loss + "
        "node_insertion_loss), pd_sensitivity_dbm (photodetector 'sensitivity'), and "
        "link_margin_db (laser_output_dbm - path_loss_db - pd_sensitivity_dbm). "
        "All four are numbers taken from tool results; leaving them empty or using a "
        "placeholder fails the gate."
    )
    recommended_template: Optional[str] = Field(
        default=None,
        description="Optional structural template recommendation (a legal core_kind name "
        "from list_topology_templates). None/empty means keep the current structure. "
        "The structure-legality gate row 5.5 validates it against the legal, evaluable templates.",
    )
    circuit_recommendations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Concrete circuit-level (L4) lever recommendations, each "
        "{\"parameter\": sharing_policy|temporal_accum|laser_distribution, \"value\": <legal "
        "candidate>, \"rationale\": <why it lowers latency*energy*area>}. These are evaluable "
        "on every executable topology and flow to the solution agent as candidate edits.",
    )
    retrieval_description: str = Field(default="")
    summary: str = Field(default="")


class DeviceOutput(BaseModel):
    """器件级分析：库变体选择与参数调优。"""

    recommended_link_edits: List[Dict[str, Any]] = Field(
        description="Component selections: level, device_type, device_name, reason."
    )
    retrieval_description: str = Field(default="")
    summary: str = Field(default="")


class EditAction(BaseModel):
    """solution 智能体可提出的单个架构/拓扑级编辑。

    ``name`` 受限于宿主实际应用的三个编辑工具，因此格式错误的条目（``name``/
    ``action`` 别名、空 ``{}`` 占位符或杜撰的工具）会被结构化输出模式本身拒绝，
    而不会泄漏到应用环节。
    """

    name: Literal[
        "modify_hardware_parameter",
        "step_hardware_parameter",
        "modify_topology_template",
    ] = Field(description="The edit tool to run.")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool arguments (e.g. {'updates': [...]} or {'parameter': ..., 'direction': ...}).",
    )


class DeviceEditAction(BaseModel):
    """单个器件变体选择编辑（``link_edits`` 列表）。"""

    name: Literal["modify_device_parameter"] = Field(
        description="Always 'modify_device_parameter' for link_edits."
    )
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="{'level': 'core'|'node'|'mesh_core'|'mesh_node', 'device_type': ..., 'device_name': ...}.",
    )


class SolutionOutput(BaseModel):
    """方案生成输出：下一个合法的硬件配置。"""

    strategy: str = Field(description="One-line summary of the proposed move.")
    actions: List[EditAction] = Field(
        description="Architecture/topology edits. Each entry is a legal edit tool "
        "call: modify_hardware_parameter / step_hardware_parameter / "
        "modify_topology_template with its arguments. No empty placeholders."
    )
    link_edits: List[DeviceEditAction] = Field(
        default_factory=list,
        description="Component-selection edits, each a modify_device_parameter call "
        "with {'level', 'device_type', 'device_name'} arguments.",
    )
    rationale: str = Field(description="Why this move lowers the objective.")
    coupling_rationale: str = Field(
        default="",
        description="How the joint point stays within shared cross-layer constraints "
        "(work_freq <= device rate chain, in_bit <= DAC prec, out_bit <= ADC prec); "
        "the coordinator's self-check via validate_coupling.",
    )
    notes: Dict[str, Any] = Field(default_factory=dict)


class EvaluateOutput(BaseModel):
    """各阶段证据评估结论（语义行 2/3/6/7 + 反馈）。

    由只读的 ``evaluate_agent`` 产出：它依据真实证据（工具结果、器件库、SimPhony
    报告）核验单个上游角色的输出，并给四条评判行打分。第 1/4/5/5.5 行保持确定性
    代码，不经过本模型。
    """

    stage: str = Field(description="Which role's output is being reviewed.")
    content_completeness: str = Field(
        description="pass | fail + one-line reason (row 2: required analyses present?)."
    )
    evidence_completeness: str = Field(
        description="pass | fail + one-line reason (row 3: enough evidence gathered?)."
    )
    reasoning_correctness: str = Field(
        description="pass | fail + one-line reason (row 6: reasoning follows evidence?)."
    )
    evidence_applicability: str = Field(
        description="pass | fail + one-line reason (row 7: evidence fits workload/hardware?)."
    )
    feedback: List[str] = Field(
        default_factory=list,
        description="Concrete, actionable rework instructions for the source agent when any row fails.",
    )
    evidence_cited: List[str] = Field(
        default_factory=list,
        description="Evidence paths/tool results actually referenced while judging (traceability).",
    )
