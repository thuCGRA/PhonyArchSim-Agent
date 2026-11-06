"""任何代理的每次工具箱调用都会接收的共享上下文。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from photon_tuner.core.schemas import EvaluationResult


@dataclass
class ToolContext:
    """任何代理的每次工具箱调用都会接收的共享上下文。"""

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
