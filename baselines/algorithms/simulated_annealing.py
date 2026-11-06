"""模拟退火 (Simulated Annealing, SA)。

教科书式 Metropolis 退火，作用域是统一动作空间。

状态机（自包含在 ``propose`` 内，与 harness 的单一 ``propose(observations)`` 接口
完全解耦）：

1. 首次调用：链的起点 = 初始点（``observations[0]``），直接按初始温度离开；
2. 此后每次调用：``observations[-1]`` 即上一条提案的评估结果——先做 Metropolis 判据
   决定链是否移动过去，再按当前温度从链上做一次随机邻居提案。

接受判据：``ΔE = log(objective_new) - log(objective_cur)``，以
``exp(-ΔE / T)`` 接收更差解。在 log 空间比较，避免乘积目标跨多个数量级时退火
温度尺度失衡。冷却采用几何冷却 ``T_k = T_0 * alpha^k``（下界 ``t_min``）。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from baselines.algorithms.base import BaseOptimizer
from baselines.harness import Observation


class SimulatedAnnealing(BaseOptimizer):
    name = "sa"

    def __init__(self, action_space, rng=None, **options: Any) -> None:
        super().__init__(action_space, rng, **options)
        self._chain: Optional[Dict[str, Any]] = None
        self._chain_obj: float = float("inf")
        self._k = 0  # 已完成的退火步数（用于温度调度）

    # -- 温度调度 ------------------------------------------------------------ #

    def _temperature(self, k: int) -> float:
        t0 = float(self.option("t0", 0.20))
        alpha = float(self.option("alpha", 0.97))
        floor = float(self.option("t_min", 0.001))
        return max(floor, t0 * (alpha ** k))

    @staticmethod
    def _logobj(objective: float) -> float:
        return math.log(max(objective, 1e-300))

    # -- 提议 ---------------------------------------------------------------- #

    def propose(self, observations: List[Observation]) -> Optional[Dict[str, Any]]:
        if self._chain is None:
            # 起点 = 初始点，尚未有任何退火步。
            self._chain = observations[0].hardware
            self._chain_obj = observations[0].objective

        # 在产生新提议前，先对上一条已评估的提案做接受/拒绝。
        last = observations[-1]
        if last.index > observations[0].index or len(observations) > 1:
            delta = self._logobj(last.objective) - self._logobj(self._chain_obj)
            temperature = self._temperature(self._k)
            if delta <= 0.0 or self.rng.random() < math.exp(-delta / max(temperature, 1e-12)):
                self._chain = last.hardware
                self._chain_obj = last.objective
            self._k += 1

        picked = self.action_space.random_neighbor(self._chain, self.rng)
        if picked is None:
            return None
        action, _next_hw = picked
        return self._apply(self._chain, action)