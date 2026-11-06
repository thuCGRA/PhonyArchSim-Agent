"""光子 DSE 的多目标帕累托档案（与范式无关）。

该档案在一组原始且具有物理意义的指标（能耗 / 面积 / 延迟 / 核心插入损耗）
上维护已评估设计点的非支配前沿，与用于保留最优解的标量 ``objective`` 无关。

它刻意与范式和策略无关：每个点携带 ``meta`` 字典
（``core_kind`` / ``node_kind`` / ``strategy`` / ``iteration``），因此同一档案
无需重写即可支持第二阶段跨范式搜索（crossbar 与 MZI mesh）以及串行与联合
搜索的消融实验。它只存储真正经过评估的仿真指标，从不使用插值或编造数据，
从而在构造上保证前沿的真实性。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

# 前沿进行权衡的原始指标（均为最小化目标）。``objective`` 被刻意排除：
# 它是由这些指标派生的加权标量，而前沿关注的是底层权衡，而非某一种标量化。
# 跨范式比较用 *吞吐归一化* 指标：能耗归一化到每 MAC（pJ_per_mac）、面积归一化到
# 每单位吞吐 MAC/s（area_um2_per_mac_per_s），使并行权重阵列（单输出向量、器件少
# 但吞吐低）与矩阵阵列（器件多但吞吐高）在同一效率轴上可比，而非被原始面积/能耗
# 假性支配。``latency_s`` 与 ``core_insertion_loss`` 保持原始量纲（时间/光学预算）。
DEFAULT_MINIMIZE_KEYS = [
    "pJ_per_mac",
    "area_um2_per_mac_per_s",
    "latency_s",
    "core_insertion_loss",
]


def _numeric_pairs(a: Dict[str, Any], b: Dict[str, Any], keys: List[str]) -> List[str]:
    """在两个指标字典中都存在且为数值型的键（否则不可比较）。"""
    return [
        k for k in keys
        if isinstance(a.get(k), (int, float)) and isinstance(b.get(k), (int, float))
    ]


def dominates(a: Dict[str, Any], b: Dict[str, Any], keys: List[str]) -> bool:
    """若 ``a`` 帕累托支配 ``b`` 则返回 True（所有轴上 <=，且至少一个轴上 <）。

    任一一方缺失的轴会被丢弃，因此某点不会因未上报的指标而受罚——
    这与不编造数据的规则一致。
    """
    numeric = _numeric_pairs(a, b, keys)
    if not numeric:
        return False
    strictly_better = False
    for k in numeric:
        if a[k] > b[k]:
            return False
        if a[k] < b[k]:
            strictly_better = True
    return strictly_better


@dataclass
class ParetoArchive:
    """带逐点元数据的增量式非支配前沿。"""

    minimize_keys: List[str] = field(default_factory=lambda: list(DEFAULT_MINIMIZE_KEYS))
    points: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, metrics: Dict[str, Any], meta: Dict[str, Any] | None = None) -> bool:
        """插入一个已评估的设计点；若进入前沿则返回 True。

        若某点被已有的前沿点支配，则被拒绝；插入成功后，会剪除所有被它
        支配的前沿点。
        """
        point = {"metrics": dict(metrics), "meta": dict(meta or {})}
        if any(dominates(p["metrics"], point["metrics"], self.minimize_keys) for p in self.points):
            return False
        self.points = [p for p in self.points if not dominates(point["metrics"], p["metrics"], self.minimize_keys)]
        self.points.append(point)
        return True

    def frontier(self, tag_key: str | None = None, tag_value: Any = None) -> List[Dict[str, Any]]:
        """前沿点，可按 ``meta`` 标签（strategy/paradigm）过滤。"""
        if tag_key is None:
            return list(self.points)
        return [p for p in self.points if p.get("meta", {}).get(tag_key) == tag_value]

    def to_dict(self) -> Dict[str, Any]:
        return {"minimize_keys": list(self.minimize_keys), "points": list(self.points)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParetoArchive":
        return cls(
            minimize_keys=list(data.get("minimize_keys", DEFAULT_MINIMIZE_KEYS)),
            points=list(data.get("points", [])),
        )
