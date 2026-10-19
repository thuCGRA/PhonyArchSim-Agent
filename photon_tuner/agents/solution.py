"""Solution-generation agent for photonic DSE.

This agent combines the model-level, layer-level, and optical-level findings plus
retrieved historical cases into the next legal hardware configuration.  Its
final output carries *both* architecture-level actions and optical-link
component selections, which the framework applies through the two toolboxes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from photon_tuner.runtime.react_loop import run_react_tool_loop
from photon_tuner.runtime.results import as_dict, as_dict_list, as_str
from photon_tuner.runtime.llm_completion import complete_json_continuing_messages
from photon_tuner.common.io import write_json
from photon_tuner.core.schemas import (
    BottleneckState,
    EvaluationResult,
    OpticalAnalysisResult,
    SolutionProposal,
)
from photon_tuner.core.search_space import materialize_hardware
from photon_tuner.llm.clients import LLMClient
from photon_tuner.llm.prompts import REACT_TOOL_USE_PROMPT, SOLUTION_PROMPT
from photon_tuner.rag.vector_store import HistoryVectorStore, summarize_cases
from photon_tuner.toolbox.analysis_tools import AnalysisToolbox
from photon_tuner.toolbox.common_tool.context import ToolContext
from photon_tuner.toolbox.device_tools import DeviceToolbox


READ_ONLY_TOOLS = [
    "summarize_hardware_config",
    "summarize_metrics",
    "summarize_energy_breakdown",
    "summarize_area_breakdown",
    "summarize_insertion_loss",
    "summarize_layer_ranks",
    "summarize_operator_groups",
    "inspect_layer_details",
    "compare_search_states",
    "select_analysis_base",
]

EDIT_TOOLS = {"modify_hardware_parameter", "step_hardware_parameter"}
# The only legal device edit is selecting a library variant.  Device cfgs
# overrides have been removed from the framework entirely.
LINK_EDIT_TOOLS = {"modify_device_parameter"}


class SolutionGenerationAgent:
    """LLM-driven solution generator with RAG historical context."""

    def __init__(
        self,
        llm: LLMClient,
        store: HistoryVectorStore,
        toolbox: AnalysisToolbox,
        device_toolbox: DeviceToolbox,
        simulator_schema: Dict[str, Any],
        top_k: int = 5,
        progress: Any = None,
    ) -> None:
        self.llm = llm
        self.store = store
        self.toolbox = toolbox
        self.device_toolbox = device_toolbox
        self.simulator_schema = simulator_schema
        self.top_k = top_k
        self.progress = progress

    def propose(
        self,
        state: BottleneckState,
        optical_result: OpticalAnalysisResult,
        hardware: Dict[str, Any],
        output_dir: Path,
        evaluation: Optional[EvaluationResult] = None,
        forbidden_hardware_fingerprints: Optional[Set[str]] = None,
        search_state: Optional[Dict[str, Any]] = None,
        evaluation_bases: Optional[Dict[str, EvaluationResult]] = None,
    ) -> SolutionProposal:
        evaluation_bases = dict(evaluation_bases or {})
        if evaluation is not None:
            evaluation_bases.setdefault("current", evaluation)

        state_message = {
            "primary_impact": state.primary_impact,
            "dominant_root_cause": state.dominant_root_cause,
            "retrieval_description": state.retrieval_description,
            "recommended_focus": state.recommended_focus,
        }
        cases = self.store.search(state.retrieval_description, hardware, top_k=self.top_k)
        summarized_cases = summarize_cases(cases)

        context = ToolContext(
            output_dir=output_dir,
            evaluation=evaluation,
            bottleneck_state=state_message,
            optical_analysis={
                "loss_budget_diagnosis": optical_result.loss_budget_diagnosis,
                "power_budget_margin": optical_result.power_budget_margin,
                "recommended_link_edits": optical_result.recommended_link_edits,
            },
            current_hardware=hardware,
            retrieved_cases=summarized_cases,
            search_space=self.simulator_schema.get("search_space", {}),
            evaluation_bases=evaluation_bases,
        )
        react_result = run_react_tool_loop(
            llm=self.llm,
            toolbox=self.toolbox,
            system_prompt=f"{SOLUTION_PROMPT}\n\n{REACT_TOOL_USE_PROMPT}",
            task="react_tool_use",
            agent_name="solution_generation",
            agent_goal="Gather evidence to produce the next legal hardware configuration.",
            context_payload={
                "bottleneck_state": state_message,
                "optical_analysis": context.optical_analysis,
                "retrieved_cases": summarized_cases,
                "metrics": evaluation.metrics if evaluation else None,
            },
            context=context,
            max_steps=8,
            allowed_tool_names=READ_ONLY_TOOLS,
            progress=self.progress,
        )

        result = complete_json_continuing_messages(
            self.llm,
            react_result.messages,
            {
                "task": "solution_generation",
                "instruction": (
                    "Produce the final solution with actions and link_edits. "
                    "actions holds architecture-level tool calls "
                    "(modify_hardware_parameter / step_hardware_parameter). "
                    "link_edits holds component *selection* calls "
                    "(modify_device_parameter with level, device_type, device_name). "
                    "Output a JSON object with exactly these keys: strategy "
                    "(a string), actions (a list of objects, each with name and "
                    "arguments), link_edits (a list of objects, each with name and "
                    "arguments), rationale (a string), notes (an object)."
                ),
            },
            validate_response=_validate_solution_result,
        )

        actions = as_dict_list(result.get("actions"))
        link_edits = as_dict_list(result.get("link_edits"))
        updated_hardware, applied_edits, failed_edits = self._apply_edits(
            hardware, actions, link_edits, output_dir, evaluation, evaluation_bases
        )

        llm_notes = as_dict(result.get("notes"))
        llm_notes["applied_edits"] = applied_edits
        if failed_edits:
            llm_notes["failed_edits"] = failed_edits

        proposal = SolutionProposal(
            strategy=as_str(result.get("strategy")),
            updated_hardware=updated_hardware,
            actions=[as_str(action.get("name")) for action in actions],
            retrieved_cases=summarized_cases,
            rationale=as_str(result.get("rationale")),
            link_edits=link_edits,
            llm_notes=llm_notes,
        )
        write_json(output_dir / "solution_proposal.json", _proposal_dict(proposal))
        return proposal

    # -- Internal ----------------------------------------------------------- #

    def _apply_edits(
        self,
        hardware: Dict[str, Any],
        actions: List[Dict[str, Any]],
        link_edits: List[Dict[str, Any]],
        output_dir: Path,
        evaluation: Optional[EvaluationResult],
        evaluation_bases: Dict[str, EvaluationResult],
    ) -> Tuple[Dict[str, Any], List[str], List[Dict[str, str]]]:
        """Apply arch actions + optical-link selections, tolerating illegal edits.

        A single bad edit must never crash the whole loop; it is skipped and
        reported in the returned ``failed`` list instead.
        """
        apply_context = ToolContext(
            output_dir=output_dir,
            evaluation=evaluation,
            current_hardware=materialize_hardware(hardware),
            search_space=self.simulator_schema.get("search_space", {}),
            evaluation_bases=evaluation_bases,
        )
        applied: List[str] = []
        failed: List[Dict[str, str]] = []
        for action in actions:
            name = str(action.get("name", ""))
            if name not in EDIT_TOOLS:
                failed.append({"name": name or "<empty>", "error": "unknown tool"})
                continue
            self._run_edit(self.toolbox, name, action.get("arguments", {}), apply_context, applied, failed)
        for edit in link_edits:
            name = str(edit.get("name", ""))
            if name not in LINK_EDIT_TOOLS:
                failed.append({"name": name or "<empty>", "error": "unknown tool"})
                continue
            arguments = edit.get("arguments", {}) or {}
            self._run_edit(self.device_toolbox, name, arguments, apply_context, applied, failed)
        return materialize_hardware(apply_context.current_hardware), applied, failed

    def _run_edit(
        self,
        toolbox: Any,
        name: str,
        arguments: Optional[Dict[str, Any]],
        context: ToolContext,
        applied: List[str],
        failed: List[Dict[str, str]],
    ) -> None:
        try:
            toolbox.run(name, context, arguments=arguments or {})
            applied.append(name)
            self._report_edit(name, arguments, ok=True)
        except (ValueError, KeyError, TypeError) as exc:
            failed.append({"name": name, "error": str(exc)})
            self._report_edit(name, arguments, ok=False, error=str(exc))

    def _report_edit(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]],
        *,
        ok: bool,
        error: str = "",
    ) -> None:
        """Surface an edit outcome live; reporting must never break a run."""
        progress = getattr(self, "progress", None)
        if progress is None:
            return
        try:
            progress.edit(
                name=name,
                arguments=arguments or {},
                status="ok" if ok else "error",
                error=error,
            )
        except Exception:
            pass


def _validate_solution_result(result: Dict[str, Any]) -> None:
    if not isinstance(result.get("actions"), list):
        raise ValueError(
            "solution_generation must contain an 'actions' list; got keys %s"
            % sorted(result.keys())
        )
    if not isinstance(result.get("link_edits"), list):
        raise ValueError(
            "solution_generation must contain a 'link_edits' list; got keys %s"
            % sorted(result.keys())
        )


def _proposal_dict(proposal: SolutionProposal) -> Dict[str, Any]:
    return {
        "strategy": proposal.strategy,
        "updated_hardware": proposal.updated_hardware,
        "actions": proposal.actions,
        "link_edits": proposal.link_edits,
        "retrieved_cases": proposal.retrieved_cases,
        "rationale": proposal.rationale,
        "llm_notes": proposal.llm_notes,
    }
