"""传统优化算法的公共基类。"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from baselines.action_space import Action, ActionSpace, apply_action
from baselines.harness import Observation


class BaseOptimizer:
    """所有传统算法的基类：持有动作空间与随机源，记录最后一次动作。"""

    name = "base"

    def __init__(self, action_space: ActionSpace, rng: Optional[random.Random] = None, **options: Any) -> None:
        self.action_space = action_space
        self.rng = rng or random.Random()
        self.options = options
        self.last_action: Optional[Dict[str, Any]] = None

    def option(self, key: str, default: Any = None):
        return self.options.get(key, default)

    def _apply(self, hardware: Dict[str, Any], action: Action) -> Dict[str, Any]:
        self.last_action = action.as_dict()
        return apply_action(hardware, action, self.action_space.space)

    def propose(self, observations: List[Observation]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError