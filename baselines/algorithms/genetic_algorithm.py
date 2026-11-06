"""遗传算法 (Genetic Algorithm, GA)。

把设计点视为「染色体」，在统一动作空间上进行种群式搜索，采用分组交叉 + 逐基因变异
+ 精英保留。

实现要点：

* **个体** = 一份硬件字典（已经 DesignSpace 规范化，天然合法）；
* **适应度** = ``-log10(objective)``（最大化）；
* **初始化**：用初始点 + 其随机邻居填充种群（个体在轮到它的求值序列里按需评估，
  绝不预支预算）；
* **选择**：三元锦标赛；
* **交叉**：逐参数统一交叉；拓扑 ``core_kind``/``node_kind`` **原子耦合**地取自同一
  亲本（经 ``switch_topology_template`` 保证合法性）；
* **变异**：随机邻居步进（覆盖 architecture / topology / device 三类动作，
  器件层变化由此引入）；
* **替换**：精英保留（elitism=1）+ 子代锦标赛。

``pop_size`` 默认 12，使预算（例如 48 次提案）约跑满 4 代。
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List

from baselines.action_space import canonical_fingerprint
from baselines.algorithms.base import BaseOptimizer
from baselines.harness import Observation

from photon_tuner.core.search_space import switch_topology_template

# 交叉时参与的架构与拓扑参数（按 DesignSpace 声明的顺序，确定性）。
_CROSSOVER_PARAMS = [
    "tiles", "cores_per_tile", "dataflow",
    "precision_in_bit", "precision_w_bit", "precision_out_bit",
    "hbm_bandwidth", "glb1_cache_size", "glb2_cache_size", "rf_cache_size",
    "core_height", "core_width", "num_wavelength", "work_freq",
    "sharing_policy", "temporal_accum", "laser_distribution", "mesh_rank",
]


class GeneticAlgorithm(BaseOptimizer):
    name = "ga"

    def __init__(self, action_space, rng=None, **options: Any) -> None:
        super().__init__(action_space, rng, **options)
        self.pop_size = int(self.option("pop_size", 12))
        self.crossover_rate = float(self.option("crossover_rate", 0.7))
        self.mutation_rate = float(self.option("mutation_rate", 0.3))
        self.elitism = int(self.option("elitism", 1))
        self._population: List[Dict[str, Any]] = []

    # -- 适应度 -------------------------------------------------------------- #

    @staticmethod
    def _fitness(metrics: Dict[str, Any]) -> float:
        objective = float(metrics.get("objective", float("inf")))
        return -math.log10(max(objective, 1e-300))

    # -- 遗传算子 ------------------------------------------------------------ #

    def _crossover(self, parent_a: Dict[str, Any], parent_b: Dict[str, Any]) -> Dict[str, Any]:
        space = self.action_space.space
        child = copy.deepcopy(parent_a)
        # 拓扑原子耦合：core_kind/node_kind 同时取自同一亲本。
        topo_donor = parent_a if self.rng.random() < 0.5 else parent_b
        try:
            child = switch_topology_template(child, str(space.get(topo_donor, "core_kind")))
        except Exception:
            pass

        for param in _CROSSOVER_PARAMS:
            donor = parent_a if self.rng.random() < 0.5 else parent_b
            try:
                current = space.get(donor, param)
            except (KeyError, IndexError):
                continue
            spec = space.param_spec(param)
            if current is None:
                continue
            if spec.get("kind") == "enum":
                space.set(child, param, str(current))
            else:
                space.set(child, param, int(round(float(current))))
        # 器件层变化交给变异算子（随机邻居内含 device 动作）。
        return space.normalize(child)

    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        picked = self.action_space.random_neighbor(individual, self.rng)
        if picked is None:
            return individual
        _action, hw = picked
        return hw

    # -- 提议 ---------------------------------------------------------------- #

    def propose(self, observations: List[Observation]):
        # 1) 建立初始种群（初始点 + 去重的随机邻居）。
        if not self._population:
            seed = observations[0].hardware
            self._population.append(seed)
            seen = {canonical_fingerprint(seed)}
            for _action, hw in self.action_space.neighbors(seed):
                fp = canonical_fingerprint(hw)
                if fp not in seen and len(self._population) < self.pop_size:
                    seen.add(fp)
                    self._population.append(hw)
            while len(self._population) < self.pop_size:
                picked = self.action_space.random_neighbor(seed, self.rng)
                if picked is None:
                    break
                _action, hw = picked
                fp = canonical_fingerprint(hw)
                if fp not in seen:
                    seen.add(fp)
                    self._population.append(hw)

        evaluated = {canonical_fingerprint(o.hardware) for o in observations}

        # 2) 种群中还有未评估个体 → 依序交出。
        pending = [hw for hw in self._population if canonical_fingerprint(hw) not in evaluated]
        if pending:
            self.last_action = {"label": "ga::population"}
            return pending[0]

        # 3) 整个种群评估完毕 → 进化一代并交出一个子代。
        self._population = self._evolve(observations)
        pending = [hw for hw in self._population if canonical_fingerprint(hw) not in evaluated]
        if pending:
            self.last_action = {"label": "ga::offspring"}
            return pending[0]
        return None

    # -- 进化 ---------------------------------------------------------------- #

    def _evolve(self, observations: List[Observation]) -> List[Dict[str, Any]]:
        by_fp = {canonical_fingerprint(o.hardware): o.metrics for o in observations}

        def fitness_of(hw: Dict[str, Any]) -> float:
            return self._fitness(by_fp.get(canonical_fingerprint(hw), {}))

        parents = sorted(self._population, key=fitness_of, reverse=True)
        if not parents:
            return self._population

        offspring: List[Dict[str, Any]] = []
        if self.elitism > 0:
            offspring.append(copy.deepcopy(parents[0]))

        while len(offspring) < self.pop_size:
            pa = self._tournament(parents, fitness_of)
            pb = self._tournament(parents, fitness_of)
            if pa is pb or self.rng.random() < self.crossover_rate:
                child = self._crossover(pa, pb)
            else:
                child = copy.deepcopy(pa)
            if self.rng.random() < self.mutation_rate:
                child = self._mutate(child)
            # 避免种群退化（重复个体）。
            fp = canonical_fingerprint(child)
            existing = {canonical_fingerprint(h) for h in offspring}
            while fp in existing:
                child = self._mutate(child)
                fp = canonical_fingerprint(child)
            offspring.append(child)
        return offspring

    def _tournament(self, parents: List[Dict[str, Any]], fitness_of) -> Dict[str, Any]:
        k = 3
        contestants = [parents[self.rng.randrange(len(parents))] for _ in range(k)]
        return max(contestants, key=fitness_of)