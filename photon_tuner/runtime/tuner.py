"""Imperative multi-agent tuning loop for photonic DSE.

One ReAct evidence pass per agent, then a final JSON completion, run as a plain
dependency-free loop (no LangGraph) so the framework works offline with a mock
LLM.  The four-agent sequence is:

    model_level -> layer_level -> optical_level -> solution_generation

where ``optical_level`` is the photonic-specific agent that diagnoses the
optical link (insertion-loss budget, optical power budget, and library variant
selections) below the layer level.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from photon_tuner.agents.layer_level import LayerLevelAgent
from photon_tuner.agents.model_level import ModelLevelAgent
from photon_tuner.agents.optical_level import OpticalLevelAgent
from photon_tuner.agents.solution import SolutionGenerationAgent
from photon_tuner.common.io import write_json
from photon_tuner.common.paths import MEMORY_DB_PATH
from photon_tuner.core.search_space import build_design_space
from photon_tuner.core.objective import metrics_with_objective
from photon_tuner.common.progress import ProgressReporter
from photon_tuner.core.schemas import EvaluationResult
from photon_tuner.llm.config import LLMConfig
from photon_tuner.core.search_space import (
    discover_device_catalog,
    materialize_hardware,
    normalize_hardware_to_design_space,
    step_arch_parameter,
)
from photon_tuner.llm.clients import LLMClient, create_llm_client
from photon_tuner.llm.tracing import LLMTraceRecorder
from photon_tuner.rag.vector_store import HistoryVectorStore, hardware_fingerprint
from photon_tuner.simulators.base import SimulatorAdapter
from photon_tuner.toolbox.analysis_tools import AnalysisToolbox
from photon_tuner.toolbox.device_tools import DeviceToolbox


@dataclass
class TuningState:
    """Accumulated state for one tuning trajectory."""

    best: Optional[EvaluationResult] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    iteration: int = 0


class MultiAgentTuner:
    """Orchestrates the four photonic agents around a SimPhony adapter."""

    def __init__(
        self,
        adapter: SimulatorAdapter,
        llm_client: LLMClient,
        devicelib_root: str,
        store: Optional[HistoryVectorStore] = None,
        output_root: Optional[Path] = None,
        search_space: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        trace_enabled: bool = False,
    ) -> None:
        self.adapter = adapter
        self.llm = llm_client
        self.devicelib_root = str(devicelib_root)
        # Global, cross-run history store (RAG memory) -- deliberately *not*
        # per-run, so every tuning run reads and appends to the same archive.
        # NOTE: must be an explicit `is not None` check -- an *empty*
        # HistoryVectorStore is falsy (its __len__ is the record count), so
        # `store or ...` would silently fall through to the global path.
        self.store = store if store is not None else HistoryVectorStore(MEMORY_DB_PATH)
        self.output_root = Path(output_root or Path("runs")).resolve()
        self.trace_enabled = trace_enabled
        self.search_space = search_space or {}
        self.top_k = top_k
        self.simulator_schema = adapter.schema()

        # Layered / multi-granularity / adaptive design space with real-mode
        # feasibility (work_freq <= device symbol rate, precision <= device prec).
        self.design_space = build_design_space(
            self.search_space,
            device_catalog=discover_device_catalog(self.devicelib_root),
        )

        self.analysis_toolbox = AnalysisToolbox(search_space=self.search_space)
        self.device_toolbox = DeviceToolbox(
            self.devicelib_root, search_space=self.search_space
        )
        self.model_agent = ModelLevelAgent(self.llm, self.analysis_toolbox)
        self.layer_agent = LayerLevelAgent(self.llm, self.analysis_toolbox)
        self.optical_agent = OpticalLevelAgent(self.llm, self.device_toolbox)
        self.solution_agent = SolutionGenerationAgent(
            llm=self.llm,
            store=self.store,
            toolbox=self.analysis_toolbox,
            device_toolbox=self.device_toolbox,
            simulator_schema=self.simulator_schema,
            top_k=self.top_k,
        )

    def tune(
        self,
        initial_hardware: Dict[str, Any],
        max_iterations: int = 4,
        num_candidates: int = 1,
        state_strategy: str = "best",
        trajectory_logging: bool = True,
        progress: Optional[ProgressReporter] = None,
        il_budget: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run the multi-agent DSE loop and return a tuning summary."""
        if num_candidates != 1:
            raise ValueError("classic loop supports num_candidates=1; use multi-candidate adapter for >1")

        hardware = normalize_hardware_to_design_space(
            materialize_hardware(initial_hardware), self.search_space
        )
        if il_budget is not None:
            self.search_space["il_budget"] = il_budget

        # One run = one packaged directory under output_root, so iter_* /
        # trajectory / trace never scatter across output_root's top level.
        run_dir = self._next_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        self.adapter.output_root = run_dir
        if self.trace_enabled:
            self.llm.recorder = LLMTraceRecorder(run_dir / "trace")

        self._current_iteration = 0
        for agent in (self.model_agent, self.layer_agent, self.optical_agent, self.solution_agent):
            agent.progress = progress

        current = self._evaluate(hardware, 0)
        state = TuningState(best=current, iteration=0)
        state.history.append(self._history_entry(current, is_best=True))
        forbidden: Set[str] = {hardware_fingerprint(hardware)}

        if progress is not None:
            progress.info(
                iteration=0,
                total_iterations=max_iterations,
                component="simulator",
                message=(
                    f"baseline workload={self.simulator_schema.get('workload')} "
                    f"metrics={_compact_metrics(current.metrics)}"
                ),
            )

        for iteration in range(1, max_iterations + 1):
            state.iteration = iteration
            self._current_iteration = iteration
            output_dir = run_dir / f"iter_{iteration:03d}" / "analysis"
            output_dir.mkdir(parents=True, exist_ok=True)

            if progress is not None:
                progress.begin_agent(
                    component="agent:model_level",
                    iteration=iteration,
                    total_iterations=max_iterations,
                    detail=f"analyze from objective={current.metrics.get('objective'):.4g}",
                )
            model_result = self.model_agent.analyze(
                evaluation=current,
                output_dir=output_dir,
                search_state={"iteration": iteration},
                evaluation_bases={"current": current, "best": state.best},
                simulator_schema=self.simulator_schema,
            )
            if progress is not None:
                top = (model_result.candidate_layers[0]["layer_name"]
                       if model_result.candidate_layers else "none")
                progress.end_agent(
                    detail=f"{len(model_result.candidate_layers)} candidate layers, top={top}"
                )
                progress.decision(
                    iteration=iteration,
                    total_iterations=max_iterations,
                    component="agent:model_level",
                    message=model_result.summary or "no summary",
                )

            if progress is not None:
                progress.begin_agent(
                    component="agent:layer_level",
                    iteration=iteration,
                    total_iterations=max_iterations,
                    detail=f"diagnose {len(model_result.candidate_layers)} candidates",
                )
            bottleneck = self.layer_agent.analyze(
                model_result=model_result,
                evaluation=current,
                output_dir=output_dir,
                evaluation_bases={"current": current, "best": state.best},
                simulator_schema=self.simulator_schema,
            )
            if progress is not None:
                progress.end_agent(
                    detail=(
                        f"impact={bottleneck.primary_impact} "
                        f"root_cause={bottleneck.dominant_root_cause}"
                    )
                )
                progress.decision(
                    iteration=iteration,
                    total_iterations=max_iterations,
                    component="agent:layer_level",
                    message=(f"primary_impact={bottleneck.primary_impact}; "
                             f"dominant_root_cause={bottleneck.dominant_root_cause}"),
                )

            if progress is not None:
                progress.begin_agent(
                    component="agent:optical_level",
                    iteration=iteration,
                    total_iterations=max_iterations,
                    detail="diagnose optical link budget",
                )
            optical_result = self.optical_agent.analyze(
                bottleneck_state=bottleneck,
                evaluation=current,
                output_dir=output_dir,
                simulator_schema=self.simulator_schema,
            )
            if progress is not None:
                progress.end_agent(
                    detail=(
                        f"link_loss_diagnosis={(optical_result.loss_budget_diagnosis or '')[:120]}, "
                        f"link_edits={len(optical_result.recommended_link_edits)}"
                    )
                )
                if optical_result.summary:
                    progress.decision(
                        iteration=iteration,
                        total_iterations=max_iterations,
                        component="agent:optical_level",
                        message=optical_result.summary,
                    )

            if progress is not None:
                progress.begin_agent(
                    component="agent:solution_generation",
                    iteration=iteration,
                    total_iterations=max_iterations,
                    detail=f"propose from objective={current.metrics.get('objective'):.4g}",
                )
            proposal = self.solution_agent.propose(
                state=bottleneck,
                optical_result=optical_result,
                hardware=current.hardware,
                output_dir=output_dir,
                evaluation=current,
                forbidden_hardware_fingerprints=forbidden,
                evaluation_bases={"current": current, "best": state.best},
            )
            if progress is not None:
                notes = proposal.llm_notes or {}
                applied = notes.get("applied_edits") or []
                rejected = notes.get("failed_edits") or []
                progress.end_agent(
                    detail=(
                        f"edits applied={len(applied)} rejected={len(rejected)} "
                        f"actions={proposal.actions}"
                    )
                )
                progress.decision(
                    iteration=iteration,
                    total_iterations=max_iterations,
                    component="agent:solution_generation",
                    message=proposal.strategy or "no strategy",
                )

            next_hardware = _repair_duplicate(
                proposal.updated_hardware, forbidden, self.search_space
            )
            next_result = self._evaluate(next_hardware, iteration)
            forbidden.add(hardware_fingerprint(next_hardware))

            if progress is not None:
                progress.metric_delta(
                    iteration=iteration,
                    total_iterations=max_iterations,
                    component="simulator",
                    before=current.metrics,
                    after=next_result.metrics,
                )

            is_best = self._is_better(next_result, state.best)
            if is_best:
                state.best = next_result
            state.history.append(self._history_entry(next_result, is_best=is_best))

            self._commit_case(
                bottleneck=bottleneck,
                hardware=current.hardware,
                proposal=proposal,
                before_metrics=current.metrics,
                after_metrics=next_result.metrics,
            )

            # Keep the best point as the anchor for the next iteration.
            if state_strategy == "best" and state.best is not None:
                current = state.best
            else:
                current = next_result

        if trajectory_logging:
            self._write_trajectory(state, run_dir)

        return {
            "iterations": state.iteration,
            "best": state.best.metrics if state.best else None,
            "best_hardware": state.best.hardware if state.best else None,
            "history": state.history,
            "run_dir": str(run_dir),
            "trace_dir": str(run_dir / "trace") if self.trace_enabled else None,
        }

    # -- Internal ----------------------------------------------------------- #

    def _evaluate(self, hardware: Dict[str, Any], iteration: int) -> EvaluationResult:
        # Enforce real-mode feasibility before evaluation: clamp work_freq to the
        # selected MZM's symbol rate and precision to the DAC/ADC precision limit,
        # mirroring the asserts real SimPhony's energy.py performs.
        hardware = self.design_space.normalize(hardware)
        result = self.adapter.evaluate(hardware, iteration)
        # Recompute the objective against the current search space.
        result.metrics = metrics_with_objective(
            result.metrics, result.hardware, self.search_space
        )
        return result

    def _is_better(self, candidate: EvaluationResult, best: Optional[EvaluationResult]) -> bool:
        if best is None:
            return True
        return float(candidate.metrics.get("objective", float("inf"))) < float(
            best.metrics.get("objective", float("inf"))
        )

    def _commit_case(
        self,
        bottleneck: Any,
        hardware: Dict[str, Any],
        proposal: Any,
        before_metrics: Dict[str, float],
        after_metrics: Dict[str, float],
    ) -> None:
        self.store.add(
            bottleneck_description=bottleneck.retrieval_description,
            hardware=hardware,
            solution={
                "strategy": proposal.strategy,
                "actions": proposal.actions,
                "link_edits": proposal.link_edits,
                "rationale": proposal.rationale,
            },
            metrics={"before": before_metrics, "after": after_metrics},
            metadata={"iteration": self._iteration_label()},
            bottleneck_state={
                "primary_impact": bottleneck.primary_impact,
                "dominant_root_cause": bottleneck.dominant_root_cause,
            },
        )

    def _iteration_label(self) -> int:
        # Kept simple: the loop increments state.iteration before each pass.
        return int(getattr(self, "_current_iteration", 0))

    def _history_entry(self, result: EvaluationResult, is_best: bool) -> Dict[str, Any]:
        return {
            "iteration": self._iteration_label(),
            "metrics": result.metrics,
            "hardware_fingerprint": hardware_fingerprint(result.hardware),
            "is_best": is_best,
            "run_dir": os.path.relpath(result.run_dir, self.output_root),
        }

    def _write_trajectory(self, state: TuningState, run_dir: Path) -> None:
        write_json(run_dir / "trajectory.json", {
            "best": state.best.metrics if state.best else None,
            "best_hardware": state.best.hardware if state.best else None,
            "history": state.history,
        })

    def _next_run_dir(self) -> Path:
        """Return ``output_root/run_iteration_NNN`` with the next free index."""
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


def _compact_metrics(metrics: Dict[str, Any]) -> str:
    keys = ("latency_s", "energy_pj", "area_um2", "objective")
    return ", ".join(
        f"{key}={float(metrics[key]):.4g}" for key in keys if key in metrics
    )


def _repair_duplicate(
    hardware: Dict[str, Any],
    forbidden_fingerprints: Set[str],
    search_space: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Escape a duplicated design by stepping one legal arch parameter."""
    candidate = materialize_hardware(hardware)
    if hardware_fingerprint(candidate) not in forbidden_fingerprints:
        return candidate
    for parameter in ["work_freq", "num_wavelength", "core_width", "core_height", "tiles"]:
        for direction in (1, -1):
            try:
                stepped = step_arch_parameter(candidate, parameter, direction, search_space)
            except (ValueError, KeyError):
                continue
            if hardware_fingerprint(stepped) not in forbidden_fingerprints:
                return stepped
    return candidate
