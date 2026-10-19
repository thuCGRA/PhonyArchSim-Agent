"""Layer-level diagnosis agent for photonic DSE."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from photon_tuner.runtime.react_loop import run_react_tool_loop
from photon_tuner.runtime.results import as_dict, as_dict_list, as_str, as_str_list
from photon_tuner.runtime.llm_completion import complete_json_continuing_messages
from photon_tuner.common.io import write_json
from photon_tuner.core.schemas import BottleneckState, EvaluationResult, ModelAnalysisResult
from photon_tuner.llm.clients import LLMClient
from photon_tuner.llm.prompts import LAYER_LEVEL_PROMPT, REACT_TOOL_USE_PROMPT
from photon_tuner.toolbox.analysis_tools import AnalysisToolbox
from photon_tuner.toolbox.common_tool.context import ToolContext


ALLOWED_TOOLS = [
    "inspect_layer_details",
    "summarize_layer_ranks",
    "summarize_operator_groups",
    "summarize_hardware_config",
    "summarize_metrics",
    "compare_search_states",
    "select_analysis_base",
]


class LayerLevelAgent:
    """LLM-driven per-layer bottleneck diagnosis agent."""

    def __init__(self, llm: LLMClient, toolbox: AnalysisToolbox, progress: Any = None) -> None:
        self.llm = llm
        self.toolbox = toolbox
        self.progress = progress

    def analyze(
        self,
        model_result: ModelAnalysisResult,
        evaluation: EvaluationResult,
        output_dir: Path,
        search_state: Optional[Dict[str, Any]] = None,
        evaluation_bases: Optional[Dict[str, EvaluationResult]] = None,
        simulator_schema: Optional[Dict[str, Any]] = None,
    ) -> BottleneckState:
        evaluation_bases = dict(evaluation_bases or {"current": evaluation})
        context = ToolContext(
            output_dir=output_dir,
            evaluation=evaluation,
            model_analysis={"candidate_layers": model_result.candidate_layers},
            current_hardware=evaluation.hardware,
            search_space=(simulator_schema or {}).get("search_space", {}),
            evaluation_bases=evaluation_bases,
            active_base=model_result.analysis_base,
        )
        react_result = run_react_tool_loop(
            llm=self.llm,
            toolbox=self.toolbox,
            system_prompt=f"{LAYER_LEVEL_PROMPT}\n\n{REACT_TOOL_USE_PROMPT}",
            task="react_tool_use",
            agent_name="layer_level",
            agent_goal="Diagnose per-layer cost and root causes for the candidate layers.",
            context_payload={
                "candidate_layers": model_result.candidate_layers,
                "metrics": evaluation.metrics,
            },
            context=context,
            max_steps=8,
            allowed_tool_names=ALLOWED_TOOLS,
            progress=self.progress,
        )

        result = complete_json_continuing_messages(
            self.llm,
            react_result.messages,
            {
                "task": "layer_level_analysis",
                "instruction": (
                    "Produce the final layer-level bottleneck state. Output a JSON "
                    "object with exactly these keys: primary_impact (a string), "
                    "dominant_root_cause (a string), layer_diagnoses (a list), "
                    "retrieval_description (a string), root_cause_summary (an object), "
                    "recommended_focus (a list of strings)."
                ),
            },
            validate_response=_validate_layer_result,
        )

        primary_impact = as_str(result.get("primary_impact")) or "energy"
        dominant_root_cause = as_str(result.get("dominant_root_cause")) or "compute"
        retrieval_description = as_str(result.get("retrieval_description")) or (
            f"{primary_impact} bottleneck driven by {dominant_root_cause}"
        )
        state = BottleneckState(
            primary_impact=primary_impact,
            dominant_root_cause=dominant_root_cause,
            layer_diagnoses=as_dict_list(result.get("layer_diagnoses")),
            retrieval_description=retrieval_description,
            root_cause_summary=as_dict(result.get("root_cause_summary")),
            recommended_focus=as_str_list(result.get("recommended_focus")),
            llm_notes=as_dict(result.get("llm_notes")),
        )
        write_json(output_dir / "layer_level_result.json", _layer_result_dict(state))
        return state


def _validate_layer_result(result: Dict[str, Any]) -> None:
    if not isinstance(result.get("layer_diagnoses"), list):
        raise ValueError(
            "layer_level_analysis must contain a 'layer_diagnoses' list; got keys %s"
            % sorted(result.keys())
        )


def _layer_result_dict(state: BottleneckState) -> Dict[str, Any]:
    return {
        "primary_impact": state.primary_impact,
        "dominant_root_cause": state.dominant_root_cause,
        "layer_diagnoses": state.layer_diagnoses,
        "retrieval_description": state.retrieval_description,
        "root_cause_summary": state.root_cause_summary,
        "recommended_focus": state.recommended_focus,
        "llm_notes": state.llm_notes,
    }
