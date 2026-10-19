"""Data schemas for the photonic multi-agent DSE framework.

These dataclasses form the simulator-independent contract shared by the
SimPhony adapters and every agent.  Alongside the architecture-level result
types there is an *optical-level* result object for the lower-level link/circuit
exploration agent, which has no equivalent in a purely digital accelerator
design space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class EvaluationResult:
    """Simulator-independent description of one evaluated photonic design point.

    ``hardware`` is the full photonic configuration (architecture parameters +
    device choices + device-level ``cfgs`` overrides).  ``metrics`` is a flat
    mapping of scalar summary values (latency_s / energy_pj / area_um2 / ...).
    ``detail_files`` points at fine-grained breakdown artifacts produced by the
    simulator (energy by device, area by device, insertion loss, memory, ...).
    """

    run_dir: Path
    hardware: Dict[str, Any]
    metrics: Dict[str, float]
    detail_files: Dict[str, Path]
    raw_files: Dict[str, Path] = field(default_factory=dict)


@dataclass
class LayerLoad:
    """Per-layer workload / cost evidence used by the layer-level agent."""

    layer_id: int
    layer_name: str
    group: str
    occurrences: int = 0
    latency_s: float = 0.0
    energy_pj: float = 0.0
    area_um2: float = 0.0
    cycles: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "layer_name": self.layer_name,
            "group": self.group,
            "occurrences": self.occurrences,
            "latency_s": self.latency_s,
            "energy_pj": self.energy_pj,
            "area_um2": self.area_um2,
            "cycles": self.cycles,
            "features": self.features,
        }


@dataclass
class ModelAnalysisResult:
    """Output of the model-level (whole-architecture) analysis agent."""

    metrics: Dict[str, float]
    candidate_layers: List[Dict[str, Any]]
    generated_files: Dict[str, str]
    summary: str
    global_findings: List[str] = field(default_factory=list)
    selected_views: List[str] = field(default_factory=list)
    analysis_base: Dict[str, Any] = field(default_factory=dict)
    llm_notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BottleneckState:
    """Output of the layer-level diagnosis agent."""

    primary_impact: str
    dominant_root_cause: str
    layer_diagnoses: List[Dict[str, Any]]
    retrieval_description: str
    root_cause_summary: Dict[str, float]
    recommended_focus: List[str]
    llm_notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpticalAnalysisResult:
    """Output of the optical-level (link/circuit) exploration agent.

    This agent works *below* the layer level: it diagnoses the photonic optical
    link — the insertion-loss budget along the longest node/core paths and the
    optical power budget (laser output vs. cumulative path loss vs.
    photodetector sensitivity) — and proposes concrete *component selections*
    (library variant switches), never fabricated parameter values.
    """

    loss_budget_diagnosis: str
    dominant_loss_contributors: List[Dict[str, Any]]
    power_budget_margin: str
    recommended_link_edits: List[Dict[str, Any]]
    retrieval_description: str
    summary: str
    generated_files: Dict[str, str] = field(default_factory=dict)
    llm_notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolutionProposal:
    """Output of the solution-generation agent.

    ``actions`` lists architecture-level hardware-editing tool calls and
    ``link_edits`` lists optical-link component *selections* (library variant
    switches), both applied by the framework before the next evaluation.
    """

    strategy: str
    updated_hardware: Dict[str, Any]
    actions: List[str]
    retrieved_cases: List[Dict[str, Any]]
    rationale: str
    link_edits: List[Dict[str, Any]] = field(default_factory=list)
    llm_notes: Dict[str, Any] = field(default_factory=dict)
