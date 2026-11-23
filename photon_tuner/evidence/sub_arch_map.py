"""搜索参数到子架构位置的映射。

``build_sub_arch``（位于 :mod:`photon_tuner.simulators.netlist`）是将搜索得到
的硬件值写入 ``ONNArchSimulator`` 实际消费的 ``sub_arch`` 字典的桥梁。本模块
将该映射显式化，使验证框架能够从仿真器读回注入值并确认参数确实生效。

标量参数映射到普通键路径；三个拓扑参数不是标量——它们通过
:mod:`photon_tuner.core.topology` 重塑 ``core.netlist``。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from photon_tuner.core.topology import (
    _SHARING_OVERRIDES,
    _TEMPORAL_DURATIONS,
)

# 标量搜索参数：param -> sub_arch 键路径。
SUB_ARCH_SCALAR_MAP: Dict[str, Tuple[str, ...]] = {
    "tiles": ("tiles",),
    "cores_per_tile": ("cores_per_tile",),
    "dataflow": ("dataflow",),
    "core_height": ("core", "height"),
    "core_width": ("core", "width"),
    "num_wavelength": ("core", "num_wavelength"),
    "work_freq": ("core", "work_freq"),
    "precision_in_bit": ("core", "precision", "in_bit"),
    "precision_w_bit": ("core", "precision", "w_bit"),
    "precision_out_bit": ("core", "precision", "out_bit"),
    "hbm_bandwidth": ("memory", "HBM", "bandwidth"),
    "glb1_cache_size": ("memory", "GLB1", "cache_size"),
    "glb2_cache_size": ("memory", "GLB2", "cache_size"),
    "rf_cache_size": ("memory", "RF", "cache_size"),
}

# 拓扑参数（以网表结构组织，非标量）。
TOPOLOGY_PARAMS: Tuple[str, ...] = (
    "sharing_policy",
    "temporal_accum",
    "laser_distribution",
    "weight_bank_groups",
)

# 记录映射：param -> 证据记录中的 sub_arch 路径。标量参数映射到普通路径；
# 拓扑参数映射到它们所重塑的网表字段。
SUB_ARCH_MAP: Dict[str, Any] = {
    **{p: list(path) for p, path in SUB_ARCH_SCALAR_MAP.items()},
    "laser_distribution": ["core", "netlist", "laser_power_distribution"],
    "temporal_accum": ["core", "netlist", "temporal_accum_factor", "duration"],
    "sharing_policy": ["core", "netlist", "scaling_rules", ["i2", "i8"]],
    "weight_bank_groups": ["core", "weight_bank_groups"],
    # 实例级器件类型切换：切换值是相应网表实例（i4 激光器 / i6 分束器）的
    # 器件类型。
    "laser_source": ["core", "netlist", "instances", "i4"],
    "splitter_kind": ["core", "netlist", "instances", "i6"],
}

# 器件选择以单元素列表 ``[variant_name]`` 形式落在 ``core.devices`` /
# ``core.node.devices`` 中。
DEVICE_READBACK_PATHS: Dict[str, Tuple[str, ...]] = {
    "core": ("core", "devices"),
    "node": ("core", "node", "devices"),
    "mesh_core": ("core", "devices"),
    "mesh_node": ("core", "node", "devices"),
    "weight_array_core": ("core", "devices"),
    "weight_array_node": ("core", "node", "devices"),
}


def _lookup(node: Any, path: Tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def read_sub_arch_param(sub_arch: Dict[str, Any], param: str) -> Any:
    """返回 ``build_sub_arch`` 为 ``param`` 写入的值（若无则返回 None）。"""
    if param in SUB_ARCH_SCALAR_MAP:
        return _lookup(sub_arch, SUB_ARCH_SCALAR_MAP[param])
    if param == "weight_bank_groups":
        return _lookup(sub_arch, ("core", "weight_bank_groups"))
    if param == "laser_source":
        inst = _lookup(sub_arch, ("core", "netlist", "instances", "i4"))
        return inst[0] if isinstance(inst, (list, tuple)) and inst else None
    if param == "splitter_kind":
        inst = _lookup(sub_arch, ("core", "netlist", "instances", "i6"))
        return inst[0] if isinstance(inst, (list, tuple)) and inst else None
    if param == "laser_distribution":
        return _lookup(sub_arch, ("core", "netlist", "laser_power_distribution"))
    if param == "temporal_accum":
        cfg = _lookup(sub_arch, ("core", "netlist", "temporal_accum_factor")) or {}
        return cfg.get("duration")
    if param == "sharing_policy":
        rules = _lookup(sub_arch, ("core", "netlist", "scaling_rules")) or {}
        return {k: rules.get(k) for k in ("i2", "i8")}
    return None


def expected_sub_arch_value(param: str, value: Any) -> Any:
    """参数候选值所期望找到的值。"""
    if param in SUB_ARCH_SCALAR_MAP or param == "weight_bank_groups":
        return value
    if param == "laser_source":
        return "off_chip_laser" if str(value) == "off_chip" else "on_chip_laser"
    if param == "splitter_kind":
        return "mmi" if str(value) == "mmi" else "laser_splitter"
    if param == "laser_distribution":
        return str(value)
    if param == "temporal_accum":
        return _TEMPORAL_DURATIONS[str(value)]
    if param == "sharing_policy":
        return dict(_SHARING_OVERRIDES[str(value)])
    return None


def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


def confirm_injection(
    sub_arch: Dict[str, Any], param: str, value: Any
) -> Tuple[bool, Dict[str, Any]]:
    """返回一个已注入参数的 ``(confirmed, readback_summary)``。"""
    actual = read_sub_arch_param(sub_arch, param)
    expected = expected_sub_arch_value(param, value)
    if param == "weight_bank_groups":
        return _confirm_weight_bank_groups(sub_arch, actual, expected)
    if isinstance(expected, dict):
        # sharing-policy 映射同时携带交叉开关实例名 (i2/i8) 和网格实例名
        # (i_adc/i_tia) 下的 ADC/TIA 缩放规则；仅比较当前网表实际暴露的项。
        relevant = {k: v for k, v in expected.items() if k in (actual or {})}
        confirmed = (
            isinstance(actual, dict)
            and bool(relevant)
            and all(_eq(actual.get(k), v) for k, v in relevant.items())
        )
        expected = relevant or expected
    else:
        confirmed = _eq(actual, expected)
    return confirmed, {
        "param": param,
        "expected": expected,
        "actual": actual,
    }


def _confirm_weight_bank_groups(
    sub_arch: Dict[str, Any], actual: Any, expected: Any,
) -> Tuple[bool, Dict[str, Any]]:
    """Check both the declared grouping and the optical/readout netlist."""
    core = sub_arch.get("core", {}) or {}
    netlist = core.get("netlist", {}) or {}
    instances = netlist.get("instances", {}) or {}
    partition = core.get("weight_bank_partition", {}) or {}
    wavelengths = core.get("num_wavelength")
    group_width = core.get("group_wavelengths")
    chain = ("off_chip_laser", "micro_comb", "node", "photodetector", "tia", "adc")
    by_type = {
        kind: {key for key, spec in instances.items() if spec and spec[0] == kind}
        for kind in (*chain, "dac")
    }
    counts = {kind: len(keys) for kind, keys in by_type.items()}
    edges = {
        (source.partition("(")[0], target.partition("(")[0])
        for source, target in (netlist.get("nets", {}) or {}).values()
    }
    connected = True
    for source_kind, target_kind in (*zip(chain, chain[1:]), ("dac", "node")):
        stage = {(src, dst) for src, dst in edges if src in by_type[source_kind]}
        if (
            {src for src, _ in stage} != by_type[source_kind]
            or {dst for _, dst in stage} != by_type[target_kind]
            or len(stage) != counts[source_kind]
        ):
            connected = False
            break
    confirmed = (
        _eq(actual, expected)
        and expected in (1, 2)
        and isinstance(wavelengths, int)
        and isinstance(group_width, int)
        and group_width > 0
        and group_width * expected == wavelengths
        and all(count == expected for count in counts.values())
        and connected
    )
    if confirmed and expected == 2:
        banks = partition.get("wavelength_indices")
        output_bits = (core.get("precision") or {}).get("out_bit")
        confirmed = (
            isinstance(banks, list)
            and len(banks) == expected
            and all(isinstance(bank, list) and len(bank) == group_width for bank in banks)
            and {index for bank in banks for index in bank} == set(range(wavelengths))
            and set(partition.get("partial_sum_sources") or []) == by_type["adc"]
            and isinstance(output_bits, int)
            and partition.get("accumulator_bits") == output_bits + 1
        )
    return confirmed, {
        "param": "weight_bank_groups",
        "expected": expected,
        "actual": actual,
        "num_wavelength": wavelengths,
        "group_wavelengths": group_width,
        "netlist_instance_counts": counts,
        "independent_readout_paths": connected,
        "partition": partition,
    }


def read_device_choice(sub_arch: Dict[str, Any], level: str, device_type: str) -> Any:
    """返回为 ``devices.<level>.<type>`` 注入的器件变体列表。"""
    path = DEVICE_READBACK_PATHS.get(level)
    if path is None:
        return None
    return _lookup(sub_arch, path + (device_type,))


def confirm_device_injection(
    sub_arch: Dict[str, Any], level: str, device_type: str, device_name: Any
) -> Tuple[bool, Dict[str, Any]]:
    """返回一个已注入器件选择的 ``(confirmed, readback_summary)``。"""
    actual = read_device_choice(sub_arch, level, device_type)
    expected = [str(device_name)]
    confirmed = isinstance(actual, (list, tuple)) and bool(actual) and actual[0] == expected[0]
    return confirmed, {
        "param": f"devices.{level}.{device_type}",
        "expected": expected,
        "actual": actual,
    }
