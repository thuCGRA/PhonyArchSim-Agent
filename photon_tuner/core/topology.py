"""L4 拓扑 / 电路搜索空间模板。

SimPhony 的电路级旋钮位于核心 ``netlist`` 中，且基于字符串/表达式
（``scaling_rules``、``temporal_accum_factor``、``laser_power_distribution``）。
这些值绝不能由 LLM 手写，因此本模块将少量白名单化的语义枚举映射为具体的
netlist 值，使搜索空间保持为受控的模板空间，而非任意连接。

暴露的三个枚举如下：

* ``sharing_policy``  -- DAC/ADC/TIA 在 crossbar 上的共享方式。仅 ADC
  （实例 ``i2``）与 TIA（实例 ``i8``）的缩放规则会改变；每个光子实例保持
  ``dptc.yml`` 默认值，使器件数量与 crossbar 拓扑保持耦合。
* ``temporal_accum``   -- 时间（时分复用）累加深度。
* ``laser_distribution`` -- 光功率分配因子（基于 R/H/C/W/N 的白名单表达式）。
"""

from __future__ import annotations

from typing import Any, Dict, List

# --------------------------------------------------------------------------- #
# 候选枚举（与 design_space.py 的 ``topology`` 层保持一致）。
# --------------------------------------------------------------------------- #

SHARING_POLICIES: List[str] = ["no_sharing", "adc_shared", "adc_tia_shared"]

# 每种共享策略对应的 ADC 与 TIA 缩放规则覆盖。DPTC crossbar 将 ADC/TIA 实例
# 命名为 ``i2`` / ``i8``；MZI-mesh netlist 将其命名为 ``i_adc`` / ``i_tia``。
# 同时写入两种命名，使该旋钮在任一核心下都生效
# （未知键对另一种 netlist 而言只是无副作用）。
_SHARING_OVERRIDES: Dict[str, Dict[str, str]] = {
    "no_sharing": {                                          # 每个 node 拥有独立的 ADC+TIA
        "i2": "H*W*R*C", "i8": "H*W*R*C",
        "i_adc": "H*W*R*C", "i_tia": "H*W*R*C",
    },
    "adc_shared": {                                          # LT 默认：ADC 按 tile 共享
        "i2": "(H*W)*R", "i8": "H*W*R*C",
        "i_adc": "(H*W)*R", "i_tia": "H*W*R*C",
    },
    "adc_tia_shared": {                                      # ADC+TIA 按 tile 共享
        "i2": "(H*W)*R", "i8": "(H*W)*R",
        "i_adc": "(H*W)*R", "i_tia": "(H*W)*R",
    },
}

TEMPORAL_ACCUM: List[str] = ["off", "default", "deep"]
_TEMPORAL_DURATIONS: Dict[str, int] = {"off": 1, "default": 3, "deep": 6}

LASER_DISTRIBUTIONS: List[str] = ["H*W", "H*W*N", "H*W*N*C"]

DEFAULT_SHARING_POLICY = "adc_shared"
DEFAULT_TEMPORAL_ACCUM = "default"
DEFAULT_LASER_DISTRIBUTION = "H*W"


def apply_sharing_policy(netlist: Dict[str, Any], policy: str) -> None:
    """按所选共享策略覆盖 ADC/TIA 缩放规则。"""
    if policy not in SHARING_POLICIES:
        raise ValueError(
            f"Unknown sharing policy {policy!r}; expected one of {SHARING_POLICIES}"
        )
    rules = dict(netlist.get("scaling_rules") or {})
    rules.update(_SHARING_OVERRIDES[policy])
    netlist["scaling_rules"] = rules


def apply_temporal_accum(netlist: Dict[str, Any], temporal: str) -> None:
    """设置时间累加时长（1 表示关闭）。"""
    if temporal not in TEMPORAL_ACCUM:
        raise ValueError(
            f"Unknown temporal accumulation {temporal!r}; expected one of {TEMPORAL_ACCUM}"
        )
    cfg = dict(netlist.get("temporal_accum_factor") or {})
    cfg["duration"] = _TEMPORAL_DURATIONS[temporal]
    netlist["temporal_accum_factor"] = cfg


def apply_laser_distribution(netlist: Dict[str, Any], distribution: str) -> None:
    """设置激光功率分配因子。

    ``laser_power_distribution`` 是解析镜像读取的键；
    ``laser_power_distribution_factor`` 是真实 ``energy.py`` 读取的键。
    同时写入两者，使两种模式对光功率预算的解读一致。
    """
    if distribution not in LASER_DISTRIBUTIONS:
        raise ValueError(
            f"Unknown laser distribution {distribution!r}; expected one of {LASER_DISTRIBUTIONS}"
        )
    netlist["laser_power_distribution"] = distribution
    netlist["laser_power_distribution_factor"] = distribution


def apply_topology(
    netlist: Dict[str, Any],
    sharing_policy: str,
    temporal_accum: str,
    laser_distribution: str,
) -> None:
    """将三项 L4 电路级选择应用到核心 netlist。"""
    apply_sharing_policy(netlist, sharing_policy)
    apply_temporal_accum(netlist, temporal_accum)
    apply_laser_distribution(netlist, laser_distribution)
