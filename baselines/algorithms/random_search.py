"""随机搜索 (Random Search, RS)。

每一步从**当前最优点的合法邻域**均匀随机地取一个动作并执行；等价于在统一动作空间
上的随机游走（保留最优点的版本，与 agent 默认的 ``state_strategy=best`` 对齐）。
``base`` 选项可切到 ``latest``（纯随机游走，不保留最优点）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from baselines.algorithms.base import BaseOptimizer
from baselines.harness import Observation


class RandomSearch(BaseOptimizer):
    name = "rs"

    def propose(self, observations: List[Observation]) -> Optional[Dict[str, Any]]:
        base_strategy = self.option("base", "best")
        if base_strategy == "latest":
            base = observations[-1].hardware
        else:
            base = min(observations, key=lambda o: o.objective).hardware
        picked = self.action_space.random_neighbor(base, self.rng)
        if picked is None:
            return None
        action, next_hw = picked
        return self._apply(base, action)