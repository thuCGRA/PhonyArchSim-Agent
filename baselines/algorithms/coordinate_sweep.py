"""坐标扫描 / 单参数网格扫描 (Coordinate Sweep)。

确定性的单参数穷举基线与 SimPhony 自身的单维扫描（#波长、位宽）思路一致。

策略：在当前最优点的邻域内，按确定性参数优先序，逐参数地尝试其候选网格上的**全部**
取值（保持其它参数不变），去重后依序提交。参数顺序为：

    architecture → topology → device(实例开关) → 电路模板

它是**非随机、可复现**的上限型传统基线（比随机搜索强很多，适合凸显 LLM agent 的
决策效率）。
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from baselines.action_space import canonical_fingerprint
from baselines.algorithms.base import BaseOptimizer
from baselines.harness import Observation

from photon_tuner.core.search_space import switch_topology_template


class CoordinateSweep(BaseOptimizer):
    name = "sweep"

    def __init__(self, action_space, rng=None, **options: Any) -> None:
        super().__init__(action_space, rng, **options)
        self._queue: List[Dict[str, Any]] = []
        self._proposed: set = set()
        self._order = self._build_order()

    def _build_order(self) -> List[str]:
        """构造确定性参数优先序。

        不含已废弃的 ``spec.mode`` 过滤（仿真只走 real，所有参数都被消费）；
        ``core_kind`` / ``node_kind`` 由模板切换覆盖，故单列在最后。
        """
        params: List[str] = []
        for layer, name, _spec in self.action_space.space.params():
            if layer in ("architecture", "topology", "device"):
                if name not in ("core_kind", "node_kind"):
                    params.append(name)
        # 模板切换放最后（更激进的结构变更）。
        return params

    def propose(self, observations: List[Observation]) -> Optional[Dict[str, Any]]:
        best = min(observations, key=lambda o: o.objective).hardware
        already = {canonical_fingerprint(o.hardware) for o in observations}

        # 每轮围绕当前最优点重建一轮扫描（去重）。
        if not self._queue:
            self._queue = self._sweep_around(best)

        while True:
            # 弹出首个尚未评估的候选。
            while self._queue:
                candidate = self._queue.pop(0)
                fp = canonical_fingerprint(candidate)
                if fp in already or fp in self._proposed:
                    continue
                self._proposed.add(fp)
                self.last_action = {"label": "sweep::coordinate"}
                return candidate
            # 队列空 → 围绕最新最优重建（允许更细的探索）。
            self._queue = self._sweep_around(best)
            # 若上一轮扫描已经覆盖全部参数且无新值，则换随机邻居收尾。
            if not self._queue:
                picked = self.action_space.random_neighbor(best, self.rng)
                if picked is None:
                    return None
                _action, hw = picked
                fp = canonical_fingerprint(hw)
                if fp not in already and fp not in self._proposed:
                    self._proposed.add(fp)
                    self.last_action = {"label": "sweep::fallback-random"}
                    return hw
                return None

    def _sweep_around(self, base: Dict[str, Any]) -> List[Dict[str, Any]]:
        space = self.action_space.space
        out: List[Dict[str, Any]] = []
        for param in self._order:
            try:
                candidates = space.candidates(param)
                current = space.get(base, param)
            except KeyError:
                continue
            if not candidates:
                continue
            for value in candidates:
                if str(value) == str(current):
                    continue
                candidate = copy.deepcopy(base)
                spec = space.param_spec(param)
                if spec.get("kind") == "enum":
                    space.set(candidate, param, str(value))
                else:
                    space.set(candidate, param, int(round(float(value))))
                out.append(space.normalize(candidate))
        # 模板切换（原子、含 required slots）。
        current_core = str(space.get(base, "core_kind") or "dptc_crossbar")
        for kind in self.action_space._templates:
            if kind == current_core:
                continue
            try:
                out.append(space.normalize(switch_topology_template(base, kind)))
            except ValueError:
                continue
        return out