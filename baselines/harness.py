"""与算法无关的公平搜索循环（所有传统基线共用）。

设计要点——**控制器即脚手架，算法只提供决策**：

::

    obs[0] = evaluate(initial_hardware)          # 同一初始点，同一预算口径
    for step in 1..budget:
        next_hw, action = optimizer.propose(obs) # 唯一允许因算法而异的环节
        obs.append(evaluate(next_hw))

于是 RS / SA / BO / GA 与 ``photon_tuner`` 的 LLM 多智能体循环之间的差异**只**来自
``propose`` 这一步。评估器、搜索空间、目标函数、keep-best 规则、产物格式全部一致，
这是对比实验可辩护的基础。

预算口径与 agent 对齐：``photon_tuner --iterations N`` 共产生 ``N + 1`` 次 SimPhony
求值（1 次初始基线 + N 次提案）。本 harness 的 ``budget`` 同样指**提案次数**。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from photon_tuner.common.io import write_json

from baselines.evaluator import EvalRecord, SimphonyBaselineEvaluator


@dataclass
class Observation:
    """一次评估的完整记录（算法可见的全部信息）。"""

    index: int                      # 评估序号
    hardware: Dict[str, Any]
    metrics: Dict[str, Any]
    action: Optional[Dict[str, Any]] = None
    cached: bool = False
    wall_s: float = 0.0

    @property
    def objective(self) -> float:
        return float(self.metrics.get("objective", float("inf")))


class Optimizer(Protocol):
    """传统算法的唯一接口：看历史，产出下一个候选设计点。"""

    name: str

    def propose(self, observations: List[Observation]) -> Optional[Dict[str, Any]]:
        """返回下一个待评估的硬件字典；``None`` 表示放弃（harness 会提前结束）。"""


@dataclass
class SearchResult:
    method: str
    workload: Optional[str]
    seed: int
    budget: int
    initial_metrics: Dict[str, Any]
    observations: List[Observation]
    best_metrics: Dict[str, Any]
    best_hardware: Dict[str, Any]
    n_calls: int
    n_unique: int
    wall_s: float
    options: Dict[str, Any] = field(default_factory=dict)

    # -- 派生量 -------------------------------------------------------------- #

    @property
    def best_objective(self) -> float:
        return float(self.best_metrics.get("objective", float("inf")))

    @property
    def initial_objective(self) -> float:
        return float(self.initial_metrics.get("objective", float("inf")))

    @property
    def improvement(self) -> float:
        """相对初始点的目标下降比例（正数 = 改进）。"""
        initial = self.initial_objective
        if initial <= 0 or initial == float("inf"):
            return 0.0
        return (initial - self.best_objective) / initial

    def curve(self) -> List[float]:
        """逐评估的 best-so-far 目标（横轴 = 求值序号）。"""
        out: List[float] = []
        best = float("inf")
        for obs in self.observations:
            best = min(best, obs.objective)
            out.append(best)
        return out

    def to_trajectory(self) -> Dict[str, Any]:
        """渲染为与 ``photon_tuner`` 运行目录兼容的 ``trajectory.json``。

        键集合是 ``photon_tuner`` 产物的**超集**（额外带 method / workload / seed /
        n_calls / n_unique / curve / options），因此 :mod:`baselines.compare` 可用
        同一解析器同时读取 agent 与基线的运行结果。
        """
        return {
            "best": self.best_metrics,
            "best_hardware": self.best_hardware,
            "history": [
                {
                    "iteration": obs.index,
                    "metrics": obs.metrics,
                    "hardware_fingerprint": obs.metrics.get("hardware_fingerprint", ""),
                    "is_best": obs.hardware == self.best_hardware,
                    "run_dir": f"iter_{obs.index:03d}",
                    "action": obs.action,
                    "cached": obs.cached,
                }
                for obs in self.observations
            ],
            "rework_total": 0,
            "search_strategy": "baseline",
            # 基线特有字段 --------------- #
            "method": self.method,
            "workload": self.workload,
            "seed": self.seed,
            "budget": self.budget,
            "initial_metrics": self.initial_metrics,
            "n_calls": self.n_calls,
            "n_unique": self.n_unique,
            "wall_s": self.wall_s,
            "objective_curve": self.curve(),
            "options": self.options,
        }


def run_search(
    *,
    method: str,
    optimizer: Optimizer,
    initial_hardware: Dict[str, Any],
    budget: int,
    evaluator: SimphonyBaselineEvaluator,
    run_dir: Path,
    state_strategy: str = "best",
    options: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> SearchResult:
    """运行一次完整的公平搜索，并把产物写入 ``run_dir``。

    ``state_strategy`` 与 agent 的 ``--state-strategy`` 同名同义（``best`` 表示每轮
    从历史最优点继续），由各算法自行解释；本函数只负责记录。
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluator.adapter.output_root = run_dir

    started = time.monotonic()

    first: EvalRecord = evaluator.evaluate(initial_hardware, action={"label": "initial"})
    observations: List[Observation] = [
        Observation(index=first.eval_index, hardware=first.hardware, metrics=first.metrics,
                    action={"label": "initial"}, cached=first.cached, wall_s=first.wall_s)
    ]
    best_hw = first.hardware
    best_metrics = first.metrics

    for _ in range(int(budget)):
        proposal = optimizer.propose(observations)
        if proposal is None:
            break
        record = evaluator.evaluate(proposal, action=getattr(optimizer, "last_action", None))
        action = getattr(optimizer, "last_action", None)
        observations.append(
            Observation(index=record.eval_index, hardware=record.hardware,
                        metrics=record.metrics, action=action, cached=record.cached,
                        wall_s=record.wall_s)
        )
        if float(record.metrics.get("objective", float("inf"))) < float(
            best_metrics.get("objective", float("inf"))
        ):
            best_hw, best_metrics = record.hardware, record.metrics
        if verbose:
            print(
                f"  [{method}] eval {record.eval_index:3d}  "
                f"obj={float(record.metrics.get('objective', float('nan'))):.4e}  "
                f"best={float(best_metrics.get('objective', float('nan'))):.4e}"
                + ("  (cache)" if record.cached else "")
            )

    result = SearchResult(
        method=method,
        workload=evaluator.workload_name,
        seed=int((options or {}).get("seed", 0)),
        budget=int(budget),
        initial_metrics=first.metrics,
        observations=observations,
        best_metrics=best_metrics,
        best_hardware=best_hw,
        n_calls=evaluator.n_calls,
        n_unique=evaluator.n_unique,
        wall_s=time.monotonic() - started,
        options=dict(options or {}),
    )
    write_json(run_dir / "trajectory.json", result.to_trajectory())
    write_json(run_dir / "summary.json", {
        "method": method,
        "workload": evaluator.workload_name,
        "seed": result.seed,
        "budget": result.budget,
        "initial_objective": result.initial_objective,
        "best_objective": result.best_objective,
        "improvement": result.improvement,
        "n_calls": result.n_calls,
        "n_unique": result.n_unique,
        "wall_s": result.wall_s,
        "options": result.options,
    })
    return result