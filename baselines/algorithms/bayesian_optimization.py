"""贝叶斯优化 (Bayesian Optimization, BO)。

标准的 GP + EI 做法，实现要点：

* 代理模型：对 ``-log10(objective)`` 拟一个 ``Matern(ν=5/2)`` 高斯过程
  （在 log 空间拟合，应对乘积目标跨数量级的动态范围）；
* 采集函数：期望改进 ``EI(x) = (mu - f* - xi) Φ(Z) + sigma φ(Z)``；
* 候选集：**最优点的邻域 ∪ 最新点的邻域**（即统一动作空间的候选），外加少量随机
  邻域以增强多样性，逐点算 EI 取 argmax。

这样 BO 的可达集合与其它基线完全一致，其唯一优势是「用代理模型挑选下一个点」。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from baselines.action_space import canonical_fingerprint
from baselines.algorithms.base import BaseOptimizer
from baselines.encoding import HardwareEncoder
from baselines.harness import Observation

try:
    import numpy as np
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    _HAS_GP = True
except Exception:  # pragma: no cover - 依赖缺失时优雅降级为随机搜索
    _HAS_GP = False


class BayesianOptimization(BaseOptimizer):
    name = "bo"

    def __init__(self, action_space, rng=None, encoder: Optional[HardwareEncoder] = None, **options: Any) -> None:
        super().__init__(action_space, rng, **options)
        self.encoder = encoder or HardwareEncoder(action_space.space, action_space.catalog)
        self._best_hw: Optional[Dict[str, Any]] = None
        self._proposal: Optional[Dict[str, Any]] = None

    # -- 辅助 ---------------------------------------------------------------- #

    def _candidate_set(self, observations: List[Observation]) -> List[Dict[str, Any]]:
        best = min(observations, key=lambda o: o.objective).hardware
        latest = observations[-1].hardware
        seen: Dict[str, Dict[str, Any]] = {}
        for base in (best, latest):
            for _, hw in self.action_space.neighbors(base):
                seen[canonical_fingerprint(hw)] = hw
        # 少量随机邻居增强多样性（尤其是早期观测极少时）。
        n_random = int(self.option("n_random", 5))
        for _ in range(n_random):
            picked = self.action_space.random_neighbor(best, self.rng)
            if picked is None:
                continue
            _, hw = picked
            seen[canonical_fingerprint(hw)] = hw
        return list(seen.values())

    def propose(self, observations: List[Observation]) -> Optional[Dict[str, Any]]:
        if not _HAS_GP:
            # 兜底：无 scikit-learn 时退化为随机搜索（仍有合法动作空间）。
            best = min(observations, key=lambda o: o.objective).hardware
            picked = self.action_space.random_neighbor(best, self.rng)
            if picked is None:
                return None
            action, hw = picked
            return self._apply(best, action)

        candidates = self._candidate_set(observations)
        if not candidates:
            return None

        # 已评估点（按观测顺序）。
        evaluated_fp = {canonical_fingerprint(o.hardware) for o in observations}
        fresh = [hw for hw in candidates if canonical_fingerprint(hw) not in evaluated_fp]
        if not fresh:
            fresh = candidates

        X = np.asarray(self.encoder.encode_matrix([o.hardware for o in observations]), dtype=float)
        # 拟合目标是最大化 -log10(objective)，即最小化 objective。
        y = np.asarray([-math.log10(max(o.objective, 1e-300)) for o in observations], dtype=float)

        # 训练随机浮动：当观测太少时用先验常均值直接算 EI，避免 GP 欠正定。
        try:
            kernel = Matern(nu=2.5, length_scale=1.0, length_scale_bounds="fixed")
            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=0,
                alpha=1e-6,
            ).fit(X, y)
            mu, sigma = gp.predict(np.asarray(self.encoder.encode_matrix(fresh)), return_std=True)
        except Exception:
            mu = np.full(len(fresh), float(np.mean(y)) if len(y) else 0.0)
            sigma = np.full(len(fresh), 1.0)

        best_y = float(np.max(y)) if len(y) else 0.0
        xi = float(self.option("xi", 0.01))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(sigma > 1e-12, (mu - best_y - xi) / sigma, 0.0)
            ei = np.where(
                sigma > 1e-12,
                (mu - best_y - xi) * norm.cdf(z) + sigma * norm.pdf(z),
                0.0,
            )
        chosen = fresh[int(np.argmax(ei))]

        # 记录动作：找出从最优邻域「到达」该点的动作（用于审计）。
        best = min(observations, key=lambda o: o.objective).hardware
        for action, hw in self.action_space.neighbors(best):
            if canonical_fingerprint(hw) == canonical_fingerprint(chosen):
                return self._apply(best, action)
        self.last_action = {"label": "bo::unmatched"}
        return chosen