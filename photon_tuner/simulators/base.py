"""Simulator adapter interface for the photonic DSE framework.

An adapter is the single seam between the simulator-independent agents and a
concrete simulator (the analytic mirror or the real SimPhony engine).  Each
adapter turns a photonic hardware configuration into a normalised
:class:`~photon_tuner.core.schemas.EvaluationResult`, so no agent code ever
imports a specific simulator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from photon_tuner.common.io import load_metrics, read_json
from photon_tuner.core.schemas import EvaluationResult


class SimulatorAdapter(ABC):
    """Contract every simulator adapter must satisfy."""

    @abstractmethod
    def evaluate(self, hardware: Dict[str, Any], iteration: int) -> EvaluationResult:
        """Run one design point and return its normalised artifacts."""

    @abstractmethod
    def schema(self) -> Dict[str, Any]:
        """Return the search-space constraints and required simulator fields."""


class GenericFileEvaluationAdapter(SimulatorAdapter):
    """Replay adapter that reads precomputed evaluation files from a run directory.

    This lets the agents be driven over an already-produced run (for example a
    real SimPhony run) without invoking the simulator again.
    """

    def __init__(
        self,
        run_dir: Path,
        hardware_path: Path,
        metrics_path: Optional[Path] = None,
        report_path: Optional[Path] = None,
        search_space: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.hardware_path = hardware_path.resolve()
        self.metrics_path = (
            metrics_path or _locate(self.run_dir, "metrics.json", "exec_res.csv")
        ).resolve()
        self.report_path = (
            report_path or _locate(self.run_dir, "report.json", "search_report.json")
        ).resolve()
        self.search_space = search_space or {}

    def evaluate(self, hardware: Dict[str, Any], iteration: int = 0) -> EvaluationResult:
        if not self.metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {self.metrics_path}")
        detail_files = {}
        if self.report_path.exists():
            detail_files["report"] = self.report_path
        return EvaluationResult(
            run_dir=self.run_dir,
            hardware=hardware,
            metrics=load_metrics(self.metrics_path),
            detail_files=detail_files,
            raw_files={"metrics": self.metrics_path, "hardware": self.hardware_path},
        )

    def load_hardware(self) -> Dict[str, Any]:
        return read_json(self.hardware_path)

    def schema(self) -> Dict[str, Any]:
        return {
            "adapter": "generic-files",
            "search_space": self.search_space,
            "detail_schema": {
                "report": "Fine-grained report.json with energy/area/insertion-loss/memory breakdowns.",
            },
        }


def _locate(run_dir: Path, preferred: str, fallback: str) -> Path:
    """Pick ``preferred`` when present, otherwise ``fallback``, under ``run_dir``."""
    candidate = run_dir / preferred
    if candidate.exists():
        return candidate
    return run_dir / fallback
