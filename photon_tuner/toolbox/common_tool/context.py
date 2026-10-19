"""Shared context passed to every toolbox call from any agent.

Pulled out of ``analysis_tools.py`` so the device toolbox no longer has to
import a dataclass from the analysis toolbox (which was a reverse dependency).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from photon_tuner.core.schemas import EvaluationResult


@dataclass
class ToolContext:
    """Shared context passed to every toolbox call from any agent."""

    output_dir: Path
    evaluation: Optional[EvaluationResult] = None
    model_analysis: Optional[Dict[str, Any]] = None
    bottleneck_state: Optional[Dict[str, Any]] = None
    optical_analysis: Optional[Dict[str, Any]] = None
    current_hardware: Optional[Dict[str, Any]] = None
    retrieved_cases: List[Dict[str, Any]] = field(default_factory=list)
    search_space: Optional[Dict[str, Any]] = None
    evaluation_bases: Dict[str, EvaluationResult] = field(default_factory=dict)
    active_base: Dict[str, Any] = field(default_factory=dict)
