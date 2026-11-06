"""真实器件变体的帕累托前沿汇总（仅离散版本）。

该模块基于真实库器件在上报的能耗/面积权衡上的取值，输出其帕累托前沿，
使器件智能体以离散选择覆盖权衡空间。返回的每个数值都是器件 YAML 中已存在的值，
不做任何插值或编造。

支持耦合感知：与当前 ``work_freq`` / 精度不兼容的器件变体会被列入
``excluded_by_coupling``（复用 R3 过滤器），绝不会被静默显示为可选。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from photon_tuner.core.pareto import dominates
from photon_tuner.core.topology_space import coupling_compatible_variants

# 前沿坐标轴，均为最小化目标。第一轴是"能量代价"，其*物理度量按器件类型而定*
# （见 ``_TYPE_ENERGY``）；第二轴 "area" 优先取显式的 ``area``，缺失时回退到
# ``length * width``。因为前沿是逐器件类型独立计算的，同一轴上的所有点必然是
# 同一度量，跨类型绝不会混比。
FRONTIER_AXES = ["power", "area"]


def _num(cfg: Dict[str, Any], key: str) -> Optional[float]:
    v = cfg.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def _speed(cfg: Dict[str, Any]) -> Optional[float]:
    """器件速率上限（GHz，与 DesignSpace 耦合链的语义一致）。"""
    if "modulation_speed" in cfg:
        v = _num(cfg, "modulation_speed")
        if v is not None:
            return v / float(cfg.get("testing_bit", 1) or 1)
    for key in ("sample_rate", "bandwidth"):
        v = _num(cfg, key)
        if v is not None:
            return v
    return None


# 每种器件类型的"能量代价"语义不同，必须按类型取物理上正确的那一列：
#   * ``mzm``           → ``efficiency``（fJ/bit）调制器的切换能量。光子调制器的
#     ``static_power`` 恒为 0，读它会让整类器件退化到"只比面积"。
#   * ``phase_shifter`` → ``P_pi``（mW）相移 π 的功耗。同理其 ``static_power`` 恒为 0。
#   * 其他类型          → ``power`` / ``static_power`` / ``dynamic_power``（mW）。
# 这只是改变*读取哪一列已记录的真实值*，不构造任何数字。
_TYPE_ENERGY: Dict[str, tuple] = {
    "mzm": (("efficiency",), "fJ/bit"),
    "phase_shifter": (("P_pi",), "mW"),
}
_DEFAULT_ENERGY: tuple = (("power", "static_power", "dynamic_power"), "mW")


def _energy(device_type: str, cfg: Dict[str, Any]) -> tuple:
    """返回 ``(数值, 单位, 命中字段)``；无可用能量指标时为 ``(None, "", "")``。"""
    keys, unit = _TYPE_ENERGY.get(device_type, _DEFAULT_ENERGY)
    for key in keys:
        v = _num(cfg, key)
        if v is not None:
            return v, unit, key
    return None, "", ""


def _area(cfg: Dict[str, Any]) -> Optional[float]:
    v = _num(cfg, "area")
    if v is not None:
        return v
    length = _num(cfg, "length")
    width = _num(cfg, "width")
    if length is not None and width is not None:
        return length * width
    return None


def _tradeoff(device_type: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """单个变体上报的权衡指标（仅包含实际存在的字段）。"""
    cfg = (body or {}).get("cfgs", {}) or {}
    out: Dict[str, Any] = {}
    speed = _speed(cfg)
    if speed is not None:
        out["speed_GHz"] = speed
    prec = _num(cfg, "prec")
    if prec is not None:
        out["prec_bit"] = prec
    energy, unit, field = _energy(device_type, cfg)
    if energy is not None:
        out["energy_cost"] = energy
        out["energy_unit"] = unit
        out["energy_field"] = field
    area = _area(cfg)
    if area is not None:
        out["area_um2"] = area
    return out


def _frontier(comparable: Dict[str, Dict[str, float]]) -> tuple:
    """FRONTIER_AXES 上未被支配的器件名（最小化目标）。"""
    names = list(comparable)
    kept = [
        n for n in names
        if not any(dominates(comparable[m], comparable[n], FRONTIER_AXES) for m in names if m != n)
    ]
    dropped = [n for n in names if n not in kept]
    return sorted(kept), sorted(dropped)


def summarize_device_frontier(
    catalog: Dict[str, Dict[str, Any]],
    hardware: Optional[Dict[str, Any]] = None,
    device_type: Optional[str] = None,
) -> Dict[str, Any]:
    """按器件类型汇总耦合兼容真实变体的帕累托前沿。"""
    filt = coupling_compatible_variants(catalog, hardware)
    types = [device_type] if device_type else sorted(catalog.keys())
    result: Dict[str, Any] = {}
    for dt in types:
        variants = catalog.get(dt, {}) or {}
        compatible = [n for n in sorted(variants) if n in filt["compatible"].get(dt, [])]
        excluded = sorted(filt["excluded"].get(dt, []))
        tradeoffs = {n: _tradeoff(dt, variants[n]) for n in sorted(variants)}
        comparable = {
            n: {"power": t.get("energy_cost"), "area": t.get("area_um2")}
            for n, t in tradeoffs.items()
            if n in compatible and t.get("energy_cost") is not None and t.get("area_um2") is not None
        }
        frontier, dominated = _frontier(comparable)
        result[dt] = {
            "compatible": compatible,
            "excluded_by_coupling": excluded,
            "frontier_power_area": frontier,
            "dominated_power_area": dominated,
            "non_comparable": [n for n in compatible if n not in comparable],
            "tradeoffs": {n: tradeoffs[n] for n in compatible},
        }
    return result
