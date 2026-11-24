"""Deterministic maturity- and convergence-aware workflow scheduling."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

ANALYSIS_LAYERS: Tuple[str, str, str] = ("architecture", "topology", "device")
MODES: Tuple[str, ...] = ("serial", "pipeline", "focus")


@dataclass(frozen=True)
class SchedulerConfig:
    warm_precedent_target: int = 3
    minimum_mature_layer_coverage: int = 3
    convergence_window: int = 10
    convergence_delta: float = 0.10
    # None means that no separate focus-exit threshold is configured; the
    # single convergence threshold governs both entry and subsequent rounds.
    focus_exit_delta: Optional[float] = None
    # 0 disables a hard Focus-duration cap. With the paper default, Focus is
    # left only when a gain exceeds the shared 10% threshold (or evidence must
    # be refreshed before Pipeline can safely resume).
    max_consecutive_focus: int = 0
    max_pipeline_evidence_age: int = 2


@dataclass
class ScheduleSignals:
    iteration: int
    objective_history: List[float] = field(default_factory=list)
    focus_history: List[Tuple[str, bool]] = field(default_factory=list)
    improved_case_count: int = 0
    improved_case_layers: List[str] = field(default_factory=list)
    pipeline_evidence_ages: Dict[str, Optional[int]] = field(default_factory=dict)
    previous_mode: Optional[str] = None
    pipeline_count: int = 0
    mode_history: List[str] = field(default_factory=list)

    def maturity(self, cfg: SchedulerConfig) -> float:
        count_target = max(int(cfg.warm_precedent_target), 1)
        coverage_target = max(int(cfg.minimum_mature_layer_coverage), 1)
        known_layers = {layer for layer in self.improved_case_layers if layer in ANALYSIS_LAYERS}
        return min(
            1.0,
            max(0, int(self.improved_case_count)) / count_target,
            len(known_layers) / coverage_target,
        )

    def improvements(self) -> List[float]:
        vals = [float(x) for x in self.objective_history if float(x) > 0]
        best = vals[0] if vals else None
        out: List[float] = []
        for value in vals[1:]:
            new_best = min(best, value) if best is not None else value
            out.append(max(0.0, (best - new_best) / best) if best else 0.0)
            best = new_best
        return out

    def converged(self, cfg: SchedulerConfig) -> bool:
        w = max(1, int(cfg.convergence_window))
        gains = self.improvements()[-w:]
        threshold = self.convergence_threshold(cfg)
        return len(gains) >= w and all(g <= threshold for g in gains)

    def convergence_threshold(self, cfg: SchedulerConfig) -> float:
        if self.previous_mode == "focus" and cfg.focus_exit_delta is not None:
            return float(cfg.focus_exit_delta)
        return float(cfg.convergence_delta)

    def pipeline_evidence_ready(self, cfg: SchedulerConfig) -> bool:
        required = ANALYSIS_LAYERS
        if any(self.pipeline_evidence_ages.get(stage) is None for stage in required):
            return False
        return all(
            int(self.pipeline_evidence_ages[stage]) <= cfg.max_pipeline_evidence_age
            for stage in required
        )

    def focus_rounds_since_pipeline(self) -> int:
        """Number of focus rounds executed after the latest pipeline round.

        Serial refresh rounds deliberately do not reset this counter: after the
        focus budget is exhausted, a stale-evidence Serial round must be followed
        by Pipeline rather than immediately falling back into Focus again.
        """
        if not self.mode_history:
            return 1 if self.previous_mode == "focus" else 0
        count = 0
        for mode in reversed(self.mode_history):
            if mode == "pipeline":
                break
            if mode == "focus":
                count += 1
        return count

    def as_dict(self, cfg: SchedulerConfig) -> Dict[str, Any]:
        gains = self.improvements()
        return {
            "iteration": self.iteration,
            "rho_memory_maturity": round(self.maturity(cfg), 4),
            "improved_case_count": self.improved_case_count,
            "improved_case_layers": sorted(set(self.improved_case_layers)),
            "best_so_far_relative_improvements": [round(x, 6) for x in gains[-cfg.convergence_window:]],
            "converged": self.converged(cfg),
            "convergence_threshold": self.convergence_threshold(cfg),
            "convergence_enter_delta": cfg.convergence_delta,
            "focus_exit_delta": cfg.focus_exit_delta,
            "previous_mode": self.previous_mode,
            "pipeline_evidence_ages": dict(self.pipeline_evidence_ages),
            "pipeline_evidence_ready": self.pipeline_evidence_ready(cfg),
            "pipeline_rounds_before": self.pipeline_count,
            "focus_rounds_since_pipeline": self.focus_rounds_since_pipeline(),
        }


@dataclass
class ScheduleDecision:
    mode: str
    focus_layer: str
    evidence_offsets: Dict[str, int]
    stages_to_run: List[str]
    reason: str
    signals: Dict[str, Any] = field(default_factory=dict)

    def runs(self, stage: str) -> bool:
        return stage in self.stages_to_run

    def offset_of(self, stage: str) -> int:
        return int(self.evidence_offsets.get(stage, 0))

    def as_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "focus_layer": self.focus_layer,
                "evidence_offsets": dict(self.evidence_offsets),
                "stages_to_run": list(self.stages_to_run), "reason": self.reason,
                "signals": dict(self.signals)}


def choose_focus_layer(signals: ScheduleSignals, cfg: SchedulerConfig) -> str:
    if not signals.focus_history:
        return ANALYSIS_LAYERS[0]
    last, improved = signals.focus_history[-1]
    if last not in ANALYSIS_LAYERS:
        return ANALYSIS_LAYERS[0]
    consecutive = 0
    for layer, _ in reversed(signals.focus_history):
        if layer != last:
            break
        consecutive += 1
    if improved and consecutive < max(1, cfg.max_consecutive_focus):
        return last
    return ANALYSIS_LAYERS[(ANALYSIS_LAYERS.index(last) + 1) % len(ANALYSIS_LAYERS)]


def decide(signals: ScheduleSignals, cfg: Optional[SchedulerConfig] = None) -> ScheduleDecision:
    cfg = cfg or SchedulerConfig()
    rho = signals.maturity(cfg)
    converged = signals.converged(cfg)
    focus = choose_focus_layer(signals, cfg)
    evidence_ready = signals.pipeline_evidence_ready(cfg)
    focus_rounds = signals.focus_rounds_since_pipeline()
    focus_budget_exhausted = (
        int(cfg.max_consecutive_focus) > 0
        and focus_rounds >= int(cfg.max_consecutive_focus)
    )
    if rho < 1.0:
        mode = "serial"
        reason = (
            f"prior is immature (rho={rho:.2f} < 1; "
            f"{signals.improved_case_count} verified cases across "
            f"{len(set(signals.improved_case_layers))} layers); refresh all layers"
        )
        focus_layer, stages, offsets = "joint", list(ANALYSIS_LAYERS), {x: 0 for x in ANALYSIS_LAYERS}
    elif signals.pipeline_count < 1:
        if evidence_ready:
            mode = "pipeline"
            reason = "mature prior and fresh evidence; establish the cross-layer pipeline before entering focus"
            focus_layer = ANALYSIS_LAYERS[0]
            stages = list(ANALYSIS_LAYERS)
            offsets = {"architecture": 0, "topology": 1, "device": 2}
        else:
            mode = "serial"
            reason = "prior is mature but first-pipeline evidence is missing or stale; refresh all layers"
            focus_layer, stages, offsets = "joint", list(ANALYSIS_LAYERS), {x: 0 for x in ANALYSIS_LAYERS}
    elif focus_budget_exhausted:
        if evidence_ready:
            mode = "pipeline"
            reason = (
                f"focus budget exhausted ({focus_rounds}/{cfg.max_consecutive_focus}); "
                "resume the cross-layer pipeline"
            )
            focus_layer = ANALYSIS_LAYERS[max(0, signals.pipeline_count) % len(ANALYSIS_LAYERS)]
            stages = list(ANALYSIS_LAYERS)
            offsets = {"architecture": 0, "topology": 1, "device": 2}
        else:
            mode = "serial"
            reason = (
                f"focus budget exhausted ({focus_rounds}/{cfg.max_consecutive_focus}) and "
                "pipeline evidence is stale; refresh all layers before resuming the pipeline"
            )
            focus_layer, stages, offsets = "joint", list(ANALYSIS_LAYERS), {x: 0 for x in ANALYSIS_LAYERS}
    elif converged:
        mode = "focus"
        threshold = signals.convergence_threshold(cfg)
        reason = f"mature prior and low recent gains (last {cfg.convergence_window} gains <= {threshold:.1%})"
        stages, offsets = [focus], {focus: 0}
        focus_layer = focus
    elif not evidence_ready:
        mode = "serial"
        reason = "pipeline prerequisites are missing or stale; refresh all layers before resuming the pipeline"
        focus_layer, stages, offsets = "joint", list(ANALYSIS_LAYERS), {x: 0 for x in ANALYSIS_LAYERS}
    else:
        mode = "pipeline"
        reason = "mature prior, fresh evidence from all three layers, and objective not converged; run all layers on staggered evidence"
        focus_layer = ANALYSIS_LAYERS[max(0, signals.pipeline_count) % len(ANALYSIS_LAYERS)]
        stages = list(ANALYSIS_LAYERS)
        offsets = {"architecture": 0, "topology": 1, "device": 2}
    return ScheduleDecision(mode, focus_layer, offsets, stages, reason, signals.as_dict(cfg))


def fixed_mode_decision(mode: str, focus: str = "architecture") -> ScheduleDecision:
    if mode not in MODES:
        raise ValueError(f"unknown schedule mode {mode!r}; expected one of {MODES}")
    if mode == "serial":
        stages, offsets, focus_layer = list(ANALYSIS_LAYERS), {x: 0 for x in ANALYSIS_LAYERS}, "joint"
    elif mode == "pipeline":
        focus = focus if focus in ANALYSIS_LAYERS else ANALYSIS_LAYERS[0]
        stages = list(ANALYSIS_LAYERS)
        offsets = {"architecture": 0, "topology": 1, "device": 2}
        focus_layer = focus
    else:
        focus = focus if focus in ANALYSIS_LAYERS else ANALYSIS_LAYERS[0]
        stages, offsets, focus_layer = [focus], {focus: 0}, focus
    return ScheduleDecision(mode, focus_layer, offsets, stages, f"fixed mode {mode} (ablation control)", {})


def summarize_history(decisions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for item in decisions:
        mode = str(item.get("mode", "?")); counts[mode] = counts.get(mode, 0) + 1
    return {"iterations": len(decisions), "mode_counts": counts,
            "mode_sequence": [x.get("mode") for x in decisions],
            "focus_sequence": [x.get("focus_layer") for x in decisions]}
