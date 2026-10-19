"""Model-level (whole-architecture) analysis agent for photonic DSE."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from photon_tuner.runtime.react_loop import run_react_tool_loop
from photon_tuner.runtime.results import as_dict, as_dict_list, as_str, as_str_list
from photon_tuner.runtime.llm_completion import complete_json_continuing_messages
from photon_tuner.common.io import write_json
from photon_tuner.core.schemas import EvaluationResult, ModelAnalysisResult
from photon_tuner.llm.clients import LLMClient
from photon_tuner.llm.prompts import MODEL_LEVEL_PROMPT, REACT_TOOL_USE_PROMPT
from photon_tuner.toolbox.analysis_tools import AnalysisToolbox
from photon_tuner.toolbox.common_tool.context import ToolContext

# Read-only evidence tools for this role.
ALLOWED_TOOLS = [
    "summarize_hardware_config",
    "summarize_metrics",
    "summarize_energy_breakdown",
    "summarize_area_breakdown",
    "summarize_insertion_loss",
    "summarize_layer_ranks",
    "summarize_operator_groups",
    "compare_search_states",
    "select_analysis_base",
]


class ModelLevelAgent:
    """LLM-driven whole-architecture analysis agent."""

    def __init__(self, llm: LLMClient, toolbox: AnalysisToolbox, progress: Any = None) -> None:
        self.llm = llm
        self.toolbox = toolbox
        self.progress = progress

    def analyze(
        self,
        evaluation: EvaluationResult,
        output_dir: Path,
        search_state: Optional[Dict[str, Any]] = None,
        evaluation_bases: Optional[Dict[str, EvaluationResult]] = None,
        simulator_schema: Optional[Dict[str, Any]] = None,
    ) -> ModelAnalysisResult:
        evaluation_bases = dict(evaluation_bases or {"current": evaluation})
        context = ToolContext(
            output_dir=output_dir,
            evaluation=evaluation,
            current_hardware=evaluation.hardware,
            search_space=(simulator_schema or {}).get("search_space", {}),
            evaluation_bases=evaluation_bases,
        )
        react_result = run_react_tool_loop(
            llm=self.llm,
            toolbox=self.toolbox,
            system_prompt=f"{MODEL_LEVEL_PROMPT}\n\n{REACT_TOOL_USE_PROMPT}",
            task="react_tool_use",
            agent_name="model_level",
            agent_goal="Identify whole-model bottleneck layers from global evidence.",
            context_payload={
                "metrics": evaluation.metrics,
                "hardware_summary": evaluation.hardware,
                "search_state_available": search_state is not None,
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
                "task": "model_level_analysis",
                "instruction": (
                    "Produce the final model-level analysis. Candidate layers must be "
                    "ranked by their global impact on the latency*energy*area objective. "
                    "Output a JSON object with exactly these keys: candidate_layers (a "
                    "list of objects, each with layer_id, layer_name, group, rank_metric, "
                    "rank_value, impact_types, root_causes), summary (a string), "
                    "global_findings (a list of strings)."
                ),
            },
            validate_response=_validate_model_result,
        )

        model_result = ModelAnalysisResult(
            metrics=dict(evaluation.metrics),
            candidate_layers=as_dict_list(result.get("candidate_layers")),
            generated_files={},
            summary=as_str(result.get("summary")),
            global_findings=as_str_list(result.get("global_findings")),
            selected_views=as_str_list(result.get("selected_views")),
            analysis_base=as_dict(result.get("analysis_base")),
            llm_notes=as_dict(result.get("llm_notes")),
        )
        write_json(output_dir / "model_level_result.json", _model_result_dict(model_result))
        return model_result


def _validate_model_result(result: Dict[str, Any]) -> None:
    if not isinstance(result.get("candidate_layers"), list):
        raise ValueError(
            "model_level_analysis must contain a 'candidate_layers' list; got keys %s"
            % sorted(result.keys())
        )


def _model_result_dict(result: ModelAnalysisResult) -> Dict[str, Any]:
    return {
        "metrics": result.metrics,
        "candidate_layers": result.candidate_layers,
        "summary": result.summary,
        "global_findings": result.global_findings,
        "selected_views": result.selected_views,
        "analysis_base": result.analysis_base,
        "llm_notes": result.llm_notes,
    }
