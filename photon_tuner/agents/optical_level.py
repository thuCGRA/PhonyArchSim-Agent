"""Optical-link / circuit-level exploration agent for photonic DSE.

This is the photonic-specific agent that extends the model→layer→solution
chain.  It works below the layer level: given the layer-level bottleneck state,
it diagnoses the photonic *optical link* — the insertion-loss budget along the
longest node/core light paths and the optical power budget (laser output vs.
cumulative path loss vs. photodetector sensitivity) — and recommends concrete
library *variant selections* for the link components.

It deliberately does **not** edit device ``cfgs`` values: those override the
fixed device library in physically unsound ways (they let the optimizer mix the
best properties of different devices).  The only legal device edit is selecting
a different library variant via ``modify_device_parameter``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from photon_tuner.runtime.react_loop import run_react_tool_loop
from photon_tuner.runtime.results import as_dict, as_dict_list, as_str
from photon_tuner.runtime.llm_completion import complete_json_continuing_messages
from photon_tuner.common.io import write_json
from photon_tuner.core.schemas import BottleneckState, EvaluationResult, OpticalAnalysisResult
from photon_tuner.llm.clients import LLMClient
from photon_tuner.llm.prompts import OPTICAL_LEVEL_PROMPT, REACT_TOOL_USE_PROMPT
from photon_tuner.toolbox.common_tool.context import ToolContext
from photon_tuner.toolbox.device_tools import DeviceToolbox


ALLOWED_TOOLS = [
    "summarize_optical_structure",
    "summarize_insertion_loss_detail",
    "summarize_device_energy_breakdown",
    "summarize_device_area_breakdown",
    "list_device_candidates",
    "inspect_device_config",
]


class OpticalLevelAgent:
    """LLM-driven optical-link / circuit exploration agent."""

    def __init__(self, llm: LLMClient, toolbox: DeviceToolbox, progress: Any = None) -> None:
        self.llm = llm
        self.toolbox = toolbox
        self.progress = progress

    def analyze(
        self,
        bottleneck_state: BottleneckState,
        evaluation: EvaluationResult,
        output_dir: Path,
        search_state: Optional[Dict[str, Any]] = None,
        simulator_schema: Optional[Dict[str, Any]] = None,
    ) -> OpticalAnalysisResult:
        context = ToolContext(
            output_dir=output_dir,
            evaluation=evaluation,
            bottleneck_state={
                "primary_impact": bottleneck_state.primary_impact,
                "dominant_root_cause": bottleneck_state.dominant_root_cause,
                "recommended_focus": bottleneck_state.recommended_focus,
            },
            current_hardware=evaluation.hardware,
            search_space=(simulator_schema or {}).get("search_space", {}),
        )
        react_result = run_react_tool_loop(
            llm=self.llm,
            toolbox=self.toolbox,
            system_prompt=f"{OPTICAL_LEVEL_PROMPT}\n\n{REACT_TOOL_USE_PROMPT}",
            task="react_tool_use",
            agent_name="optical_level",
            agent_goal="Diagnose the optical link (insertion-loss and power budgets) and recommend component selections.",
            context_payload={
                "bottleneck_state": context.bottleneck_state,
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
                "task": "optical_level_analysis",
                "instruction": (
                    "Produce the final optical-link analysis with recommended component "
                    "selections. Output a JSON object with exactly these keys: "
                    "loss_budget_diagnosis (a string), dominant_loss_contributors (a list "
                    "of objects, each with device_type, device_name, insertion_loss_dB), "
                    "power_budget_margin (a string), recommended_link_edits (a list of "
                    "objects, each with level, device_type, device_name, reason), "
                    "retrieval_description (a string), summary (a string)."
                ),
            },
            validate_response=_validate_optical_result,
        )

        analysis = OpticalAnalysisResult(
            loss_budget_diagnosis=as_str(result.get("loss_budget_diagnosis")),
            dominant_loss_contributors=as_dict_list(result.get("dominant_loss_contributors")),
            power_budget_margin=as_str(result.get("power_budget_margin")),
            recommended_link_edits=as_dict_list(result.get("recommended_link_edits")),
            retrieval_description=as_str(result.get("retrieval_description")),
            summary=as_str(result.get("summary")),
            llm_notes=as_dict(result.get("llm_notes")),
        )
        write_json(output_dir / "optical_level_result.json", _optical_result_dict(analysis))
        return analysis


def _validate_optical_result(result: Dict[str, Any]) -> None:
    if not isinstance(result.get("recommended_link_edits"), list):
        raise ValueError(
            "optical_level_analysis must contain a 'recommended_link_edits' list; got keys %s"
            % sorted(result.keys())
        )
    if not isinstance(result.get("dominant_loss_contributors"), list):
        raise ValueError(
            "optical_level_analysis must contain a 'dominant_loss_contributors' list; got keys %s"
            % sorted(result.keys())
        )


def _optical_result_dict(analysis: OpticalAnalysisResult) -> Dict[str, Any]:
    return {
        "loss_budget_diagnosis": analysis.loss_budget_diagnosis,
        "dominant_loss_contributors": analysis.dominant_loss_contributors,
        "power_budget_margin": analysis.power_budget_margin,
        "recommended_link_edits": analysis.recommended_link_edits,
        "retrieval_description": analysis.retrieval_description,
        "summary": analysis.summary,
        "generated_files": analysis.generated_files,
        "llm_notes": analysis.llm_notes,
    }
