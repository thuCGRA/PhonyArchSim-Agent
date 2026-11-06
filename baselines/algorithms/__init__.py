"""传统算法注册表。"""

from __future__ import annotations

from typing import Any, Dict, Type

from baselines.action_space import ActionSpace
from baselines.algorithms.base import BaseOptimizer
from baselines.algorithms.bayesian_optimization import BayesianOptimization
from baselines.algorithms.coordinate_sweep import CoordinateSweep
from baselines.algorithms.genetic_algorithm import GeneticAlgorithm
from baselines.algorithms.random_search import RandomSearch
from baselines.algorithms.simulated_annealing import SimulatedAnnealing

# 名称 -> 算法类。RS/SA/BO/GA 为对比用的四个主基线；sweep 为可选的补充基线
# （确定性的单参数穷举）。
REGISTRY: Dict[str, Type[BaseOptimizer]] = {
    "rs": RandomSearch,
    "sa": SimulatedAnnealing,
    "bo": BayesianOptimization,
    "ga": GeneticAlgorithm,
    "sweep": CoordinateSweep,
}

# 对比主表默认使用的四个基线。
PRIMARY_ALGORITHMS = ("rs", "sa", "bo", "ga")

__all__ = [
    "REGISTRY",
    "PRIMARY_ALGORITHMS",
    "BaseOptimizer",
    "RandomSearch",
    "SimulatedAnnealing",
    "BayesianOptimization",
    "GeneticAlgorithm",
    "CoordinateSweep",
]


def build_optimizer(name: str, action_space: ActionSpace, rng, **options: Any) -> BaseOptimizer:
    """按名称实例化一个优化器。"""
    if name not in REGISTRY:
        raise KeyError(f"unknown algorithm {name!r}; legal: {sorted(REGISTRY)}")
    return REGISTRY[name](action_space, rng=rng, **options)