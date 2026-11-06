"""电路结构拓扑模型与提供者接口。

硬件字典已携带 ``arch.topology``（sharing_policy / temporal_accum /
laser_distribution，见 :mod:`photon_tuner.core.topology`）。本模块是拓扑智能体
所需的更丰富的电路结构层：一条带器件变体标注的有序光链路
（laser -> splitter -> waveguide -> MZM -> MRR rerouter -> coupler ->
photodetector）。

SimPhony 目前未原生暴露链路图，因此我们提供 ``StaticTopologyProvider``，
从硬件的器件选择推导出规范路径。``TopologyProvider`` 协议是预留给未来
SimPhony 扩展（``SimphonyTopologyProvider``）接入的接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

# 光子 MVM 核心的规范光路径，按光的行进顺序排列。
# ``level`` 是硬件 ``devices`` 表（"core" 或 "node"）。
OPTICAL_PATH_ORDER: List[tuple] = [
    ("core", "on_chip_laser"),
    ("core", "laser_splitter"),
    ("core", "waveguide"),
    ("core", "mzm"),
    ("core", "mrr_rerouter"),
    ("node", "coupler"),
    ("node", "photodetector"),
]

# 实例级开关，而非器件类型（DeviceToolbox.INSTANCE_CHOICES）。
_NON_DEVICE_KEYS = frozenset({"laser_source", "splitter_kind"})


@dataclass
class TopologyNode:
    """电路中的一个组件，附带所选变体标注。"""

    device_type: str
    device_name: str
    level: str
    on_optical_path: bool = False


@dataclass
class TopologyGraph:
    """电路结构：所有组件 + 有序光路径。"""

    nodes: List[TopologyNode]
    optical_path: List[str] = field(default_factory=list)
    source: str = "static"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "optical_path": self.optical_path,
            "nodes": [
                {
                    "device_type": n.device_type,
                    "device_name": n.device_name,
                    "level": n.level,
                    "on_optical_path": n.on_optical_path,
                }
                for n in self.nodes
            ],
        }


class TopologyProvider(Protocol):
    """预留接口：返回某硬件点的电路结构图。"""

    def topology(self, hardware: Dict[str, Any]) -> TopologyGraph: ...


class StaticTopologyProvider:
    """占位提供者：从 ``hardware["devices"]`` 推导规范路径。

    未来的 ``SimphonyTopologyProvider`` 可从仿真器读取真实 netlist / 光路径，
    在不改动智能体层的情况下替换本实现。
    """

    def topology(self, hardware: Dict[str, Any]) -> TopologyGraph:
        devices = hardware.get("devices", {}) or {}
        core = devices.get("core", {}) or {}
        node = devices.get("node", {}) or {}

        path_types = {(level, dtype) for level, dtype in OPTICAL_PATH_ORDER}
        nodes: List[TopologyNode] = []
        optical_path: List[str] = []

        # 光路径上的器件，按规范顺序排列。
        for level, dtype in OPTICAL_PATH_ORDER:
            table = core if level == "core" else node
            name = table.get(dtype)
            if name is None:
                continue
            nodes.append(TopologyNode(dtype, str(name), level, on_optical_path=True))
            optical_path.append(dtype)

        # 其余组件（电气 / 非光路径）：DAC、ADC、TIA、
        # 移相器、y 分支、微梳等。
        for level, table in (("core", core), ("node", node)):
            for dtype, name in table.items():
                if (level, dtype) in path_types or dtype in _NON_DEVICE_KEYS or not isinstance(name, str):
                    continue
                nodes.append(TopologyNode(dtype, name, level, on_optical_path=False))

        return TopologyGraph(nodes=nodes, optical_path=optical_path, source="static")
