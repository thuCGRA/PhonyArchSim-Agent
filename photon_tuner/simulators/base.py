"""光子 DSE 框架的模拟器适配器接口。

适配器是模拟器无关的智能体与具体模拟器（解析镜像或真实 SimPhony 引擎）之间的唯一
接缝。每个适配器把一份光子学硬件配置转换为标准化的
:class:`~photon_tuner.core.schemas.EvaluationResult`，因此任何智能体代码都无需直接
导入某个具体的模拟器。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from photon_tuner.common.io import load_metrics, read_json
from photon_tuner.core.schemas import EvaluationResult


class SimulatorAdapter(ABC):
    """每个模拟器适配器都必须满足的契约。"""

    @abstractmethod
    def evaluate(self, hardware: Dict[str, Any], iteration: int) -> EvaluationResult:
        """运行一个设计点并返回其标准化产物。"""

    @abstractmethod
    def schema(self) -> Dict[str, Any]:
        """返回搜索空间约束与所需的模拟器字段。"""


class GenericFileEvaluationAdapter(SimulatorAdapter):
    """从运行目录读取预计算求值文件的回放适配器。

    这让智能体可以基于已产出的运行结果（例如一次真实的 SimPhony 运行）被驱动，而无需
    再次调用模拟器。
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
    """在 ``run_dir`` 下，若 ``preferred`` 存在则选它，否则选 ``fallback``。"""
    candidate = run_dir / preferred
    if candidate.exists():
        return candidate
    return run_dir / fallback
