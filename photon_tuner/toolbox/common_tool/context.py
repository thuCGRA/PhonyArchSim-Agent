"""任何代理的每次工具箱调用都会接收的共享上下文。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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
    # 以下对象由确定性宿主按轮绑定，使工具可以读取完整的搜索状态，而无需
    # 导入 orchestrator 或依赖全局单例。它们均为可选，领域单元测试仍可只构造
    # 最小 ToolContext。
    design_space: Optional[Any] = None
    case_store: Optional[Any] = None
    pareto_archive: Optional[Any] = None
    schedule_signals: Optional[Any] = None
    schedule_config: Optional[Any] = None
    forbidden_fingerprints: Set[str] = field(default_factory=set)
    simulator_adapter: Optional[Any] = None
    iteration: int = 0
    workload_name: str = ""
