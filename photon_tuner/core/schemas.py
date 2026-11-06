"""光子多智能体 DSE 框架的数据模式。

这些 dataclass 构成与仿真器无关的契约，由 SimPhony 适配器与所有智能体共享。
除架构级结果类型外，还有面向底层链路/电路探索智能体的光层级结果对象，
后者在纯数字加速器设计空间中并无对应物。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class EvaluationResult:
    """与仿真器无关的单个已评估光子设计点描述。

    ``hardware`` 是完整的光子配置（架构参数 + 器件选择 + 器件级 ``cfgs``
    覆盖）。``metrics`` 是标量汇总值的扁平映射
    （latency_s / energy_pj / area_um2 / ...）。``detail_files`` 指向仿真器
    生成的细粒度分解产物（按器件能耗、按器件面积、插入损耗、内存等）。
    """

    run_dir: Path
    hardware: Dict[str, Any]
    metrics: Dict[str, float]
    detail_files: Dict[str, Path]
    raw_files: Dict[str, Path] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON 安全形式，用于 LangGraph 检查点快照。"""
        return {
            "run_dir": str(self.run_dir),
            "hardware": dict(self.hardware),
            "metrics": dict(self.metrics),
            "detail_files": {k: str(v) for k, v in self.detail_files.items()},
            "raw_files": {k: str(v) for k, v in self.raw_files.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """从 ``to_dict`` 快照重建（Path 字段会被重新还原）。"""
        return cls(
            run_dir=Path(data.get("run_dir", ".")),
            hardware=dict(data.get("hardware") or {}),
            metrics=dict(data.get("metrics") or {}),
            detail_files={k: Path(v) for k, v in (data.get("detail_files") or {}).items()},
            raw_files={k: Path(v) for k, v in (data.get("raw_files") or {}).items()},
        )


@dataclass
class LayerLoad:
    """层级智能体使用的逐层工作负载 / 成本证据。"""

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
    """模型级（整架构）分析智能体的输出。"""

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
    """层级诊断智能体的输出。"""

    primary_impact: str
    dominant_root_cause: str
    layer_diagnoses: List[Dict[str, Any]]
    retrieval_description: str
    root_cause_summary: Dict[str, float]
    recommended_focus: List[str]
    llm_notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpticalAnalysisResult:
    """光层级（链路/电路）探索智能体的输出。

    该智能体工作在层级之下：诊断光子光链路——即最长 node/core 路径上的
    插入损耗预算，以及光功率预算（激光输出与累积路径损耗、光电探测器灵敏度
    的比较）——并提出具体的组件选择（库变体切换），从不编造参数值。
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
    """方案生成智能体的输出。

    ``actions`` 列出架构级硬件编辑工具调用，``link_edits`` 列出光链路组件的
    选择（库变体切换），二者均由框架在下一次评估前应用。
    """

    strategy: str
    updated_hardware: Dict[str, Any]
    actions: List[str]
    retrieved_cases: List[Dict[str, Any]]
    rationale: str
    link_edits: List[Dict[str, Any]] = field(default_factory=list)
    llm_notes: Dict[str, Any] = field(default_factory=dict)
