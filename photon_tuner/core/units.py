"""规范单位：把 SimPhony 的 SI 输出换算成光电系统惯例的工程量纲。

## 为什么需要这一层

SimPhony 的物理模型按 SI 工作（秒、焦耳、米²），其原生报告因此给出
``latency ~ 6.6e-3 s``、``energy ~ 2.3e-9 J``、``area ~ 6.7e7 um^2`` 这样的量级。
这些数值在 JSON 里可读性极差，且与光电子/加速器文献的惯例不符（文献里
延时用 ms/µs、能耗用 mJ/pJ、面积用 mm²、能效用 fJ/MAC）。

本模块是**唯一的**单位换算点：仿真边界（:mod:`photon_tuner.simulators.simphony`
的指标组装）与解析镜像（:mod:`photon_tuner.simulators.analytic`）调用
:func:`to_display_metrics`，此后全项目（目标函数、Pareto、工具、智能体载荷、
基线、实验脚本）只处理规范单位。之所以不散落各处，是因为同一个物理量出现
两套单位、却没有一处声明，是这类项目最常见也最难查的一类 bug。

## 规范单位表

===================  ================  ==========  ==================================
规范键                物理含义           单位         典型量级（mobilevit, crossbar）
===================  ================  ==========  ==================================
``latency_ms``       端到端延时          ms          ~6.6
``energy_mj``        单次推理总能耗       mJ          ~2.3
``area_mm2``         芯片总面积          mm²         ~66.5
``fJ_per_mac``       每次 MAC 能耗       fJ/MAC      ~223.5
``area_mm2_per_mac_per_ms``  面积/吞吐    mm²/(MAC/ms)  ~4.7e-2
===================  ================  ==========  ==================================

## 一个有用的性质：目标函数在换算下不变

目标函数是三个指标之积 ``J = latency * energy_per_mac * area``。上述系数之积
正好为 1（``1e3 * 1e3 * 1e-6``），因此 **J 的数值在换算前后完全相同**。
这让单位换算成为纯展示层改动：历史 run、Pareto 前沿、已发表的对比数值全部
保持可比，无需重算，也不存在"换了单位所以变好了"的歧义。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping

#: 规范单位名（出现在 ``metrics.json`` 的 ``units`` 子字典里，供工具与论文引用）。
DISPLAY_UNITS: Dict[str, str] = {
    "latency_ms": "ms",
    "energy_mj": "mJ",
    "energy_compute_mj": "mJ",
    "energy_memory_mj": "mJ",
    "area_mm2": "mm^2",
    "fJ_per_mac": "fJ/MAC",
    "area_mm2_per_mac_per_ms": "mm^2/(MAC/ms)",
    "throughput_mac_per_ms": "MAC/ms",
    "core_insertion_loss": "dB",
}

#: SI -> 规范单位的乘数。键为规范名。
_CONVERSIONS: Dict[str, float] = {
    "latency_ms": 1e3,      # s    -> ms
    "energy_mj": 1e-9,      # pJ   -> mJ   (1 pJ = 1e-9 mJ)
    "energy_compute_mj": 1e-9,
    "energy_memory_mj": 1e-9,
    "area_mm2": 1e-6,       # um^2 -> mm^2
    "fJ_per_mac": 1e3,      # pJ/MAC -> fJ/MAC
    "area_mm2_per_mac_per_ms": 1e-3,  # um^2/(MAC/s) -> mm^2/(MAC/ms)
    "throughput_mac_per_ms": 1e-3,    # MAC/s -> MAC/ms
}

#: 旧键名 -> 规范键名。仅用于读取历史 run 与容忍外部输入。
LEGACY_ALIASES: Dict[str, str] = {
    "latency_s": "latency_ms",
    "energy_pj": "energy_mj",
    "energy_compute_pj": "energy_compute_mj",
    "energy_memory_pj": "energy_memory_mj",
    "area_um2": "area_mm2",
    "pJ_per_mac": "fJ_per_mac",
    "area_um2_per_mac_per_s": "area_mm2_per_mac_per_ms",
    "throughput_mac_per_s": "throughput_mac_per_ms",
}

#: 规范键 -> SI 原始键（反向，用于写 ``si_metrics`` 镜像）。
SI_COUNTERPARTS: Dict[str, str] = {
    canonical: legacy for legacy, canonical in LEGACY_ALIASES.items()
}


def convert(key: str, value: float) -> float:
    """把 ``key`` 的 SI 值换算成对应的规范单位值；非长度/能量键原样返回。"""
    factor = _CONVERSIONS.get(key)
    if factor is None:
        return float(value)
    return float(value) * factor


def to_display_metrics(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """把一份 SI 指标字典换算成规范单位。

    键名按 :data:`LEGACY_ALIASES` 改名，数值按 :data:`_CONVERSIONS` 缩放；
    未在换算表中的键（如 ``total_macs``、``weight_*``、``il_budget_penalty``）
    原样保留，因为它们是无量纲计数或比率。

    ``raw`` 中的 ``throughput_mac_per_s`` / ``area_um2_per_mac_per_s`` 这类派生量
    单独换算，避免与被换算的原始量重复相乘。
    """
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        canonical = LEGACY_ALIASES.get(str(key), str(key))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            out[canonical] = value
            continue
        number = float(value)
        if not math.isfinite(number):
            out[canonical] = value
            continue
        out[canonical] = convert(canonical, number)
    return out


def si_mirror(display: Mapping[str, Any]) -> Dict[str, float]:
    """返回规范单位指标对应的原 SI 值，便于在报告里保留可追溯的原始量。"""
    mirror: Dict[str, float] = {}
    for canonical, si_key in SI_COUNTERPARTS.items():
        if canonical not in display:
            continue
        factor = _CONVERSIONS.get(canonical)
        try:
            number = float(display[canonical])
        except (TypeError, ValueError):
            continue
        if not factor or not math.isfinite(number):
            continue
        mirror[si_key] = number / factor
    return mirror


def describe_units() -> Dict[str, str]:
    """返回规范单位表的副本，供报告 ``units`` 块与智能体提示使用。"""
    return dict(DISPLAY_UNITS)
