"""MZI 网格拓扑模型。

为 :mod:`photon_tuner.core.topology_space` 中声明的三种网格范式提供纯几何、
无依赖的计算：

* ``reck``     — Reck 三角网格。
* ``clements`` — Clements 矩形网格。
* ``butterfly``— FFT 式蝶形网格（MZI 数量/深度取自经典 FFT 蝶形结构，见 ``note``）。

此处每个数值都是已发表论文中的 *结构性* 事实（MZI 数量、深度、耦合器/移相器倍数），
而非捏造的器件参数。器件级成本（耦合器插入损耗、移相器 P_pi、占位面积）**不** 在此
来自现有的器件库（经网格节点网表）。

正确性由单元测试锁定：针对具体模式数断言已发表的 MZI 数量与深度。
"""

from __future__ import annotations

import math
from typing import Any, Dict

# 通用网格中每个 2x2 MZI 分解为两个 50:50 定向耦合器加两个移相器
# （一个内部 theta、一个外部 phi）。该倍数是结构性事实，而非器件参数。
COUPLERS_PER_MZI = 2
PHASE_SHIFTERS_PER_MZI = 2

_MESH_SPECS: Dict[str, Dict[str, Any]] = {
    "reck": {
        "universal": True,
        "crossing_free": False,
        "range_forwards": 1,
        "ref": "10.1103/PhysRevLett.73.58",
        "note": "Reck 1994 triangular mesh: N(N-1)/2 MZIs, depth 2N-3.",
    },
    "clements": {
        "universal": True,
        "crossing_free": True,
        "range_forwards": 1,
        "ref": "10.1364/OPTICA.3.001460",
        "note": "Clements 2016 rectangular mesh: N(N-1)/2 MZIs, depth N.",
    },
    "butterfly": {
        "universal": False,
        "crossing_free": False,
        "range_forwards": 1,
        "ref": "10.1090/S0025-5718-1965-0178586-1",
        "note": "Butterfly FFT mesh: M*log2(M)/2 MZIs, depth log2(M); non-universal. "
        "MZI count/depth follow the classical FFT butterfly (Cooley-Tukey 1965, "
        "10.1090/S0025-5718-1965-0178586-1); each butterfly maps to one 2x2 MZI via the "
        "standard coupler+phase-shifter decomposition (Clements/Reck). No single "
        "photonic-mesh DOI: a prior Tang-2021 attribution was retracted "
        "(10.1021/acsphotonics.1c00419 is a 10-port universal processor, not a butterfly); "
        "evaluable stays false pending a dedicated photonic-FFT source.",
    },
}


def n_mzi(kind: str, n_modes: int) -> int:
    """实现一个 ``n_modes`` 模式网格所需的 2x2 MZI 数量。"""
    if kind in ("reck", "clements"):
        return n_modes * (n_modes - 1) // 2
    if kind == "butterfly":
        return n_modes * int(math.ceil(math.log2(n_modes))) // 2
    raise ValueError(f"Unknown mesh kind: {kind!r}")


def depth(kind: str, n_modes: int) -> int:
    """网格的 MZI 层数（光学深度）。"""
    if kind == "reck":
        return 2 * n_modes - 3
    if kind == "clements":
        return n_modes
    if kind == "butterfly":
        return int(math.ceil(math.log2(n_modes)))
    raise ValueError(f"Unknown mesh kind: {kind!r}")


def n_crossing(kind: str, n_modes: int) -> int:
    """crossing-bearing 布局所需的波导交叉数（**结构量级估计**）。

    Clements 矩形网格无交叉（``crossing_free=True``）。Reck 三角网格与
    butterfly/FFT 蝶形网格均含交叉：非平面布局中每个 2x2 MZI 单元约需一次交叉
    布线，故交叉数与 MZI 数同量级。精确数量取决于具体物理版图；此处按拓扑类的
    结构量级建模，属拓扑结构事实（与 ``n_mzi`` 同源），**不是**器件参数。
    """
    if kind == "clements":
        return 0
    return n_mzi(kind, n_modes)


def mesh_topology(kind: str, n_modes: int) -> Dict[str, Any]:
    """网格的结构概要（MZI 数量、深度、器件倍数）。"""
    if kind not in _MESH_SPECS:
        raise ValueError(f"Unknown mesh kind: {kind!r}")
    count = n_mzi(kind, n_modes)
    return {
        "kind": kind,
        "n_modes": n_modes,
        "n_mzi": count,
        "depth": depth(kind, n_modes),
        "n_coupler": count * COUPLERS_PER_MZI,
        "n_phase_shifter": count * PHASE_SHIFTERS_PER_MZI,
        "n_crossing": n_crossing(kind, n_modes),
        "universal": _MESH_SPECS[kind]["universal"],
        "crossing_free": _MESH_SPECS[kind]["crossing_free"],
        "range_forwards": _MESH_SPECS[kind]["range_forwards"],
        "ref": _MESH_SPECS[kind]["ref"],
        "note": _MESH_SPECS[kind]["note"],
    }
