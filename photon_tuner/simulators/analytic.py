"""无第三方依赖的光子学解析求值器。

本模块是 SimPhony 器件/电路/架构工作流的公式级镜像：读取与 SimPhony 完全相同的器件
与核心/节点网表 YAML，仅用 Python 标准库加 PyYAML 复现其器件计数、功耗初始化、
能耗、面积（包围盒）与插入损耗（最长路径）计算。

它让整个多智能体 DSE 循环可以在离线环境下运行（无需 PyTorch / TorchONN / mmengine /
networkx）。当这些依赖存在时，:class:`photon_tuner.simulators.simphony.SimphonyAdapter`
会改用真实的 ``ONNArchSimulator``；两者产出相同的细粒度报告结构
（见 ``simphony/onnarchsim/report.py``）。

子架构（网表）装配已拆分到 :mod:`photon_tuner.simulators.netlist`——那部分是
**模式无关**的，真实模式同样要读它。本模块为向后兼容继续再导出那几个装配入口。

被评估的 *工作负载* 可命名并加载（见 :mod:`photon_tuner.core.workload` 与
``configs/workloads.json``）。可选的编码器工作负载（``vit_base`` / ``vit_small`` /
``vit_tiny``）的逐层循环元组使用 SimPhony 自身的分区算术计算，因此逐层需求曲线会
随架构编辑而变化。同时上报一个原生的 ``energy.by_layer`` 逐层分解。
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from photon_tuner.core.search_space import materialize_hardware
from photon_tuner.core.workload import (
    DEFAULT_MINIBLOCK,
    build_workload,
    total_macs,
)

# 网表装配已迁往 ``netlist`` 模块；此处再导出以保持既有导入路径可用。
from photon_tuner.simulators.netlist import (  # noqa: F401
    _WEIGHT_ARRAY_SPECS,
    _load_yaml_versioned,
    _tunable_mzi_node,
    _weight_element_node,
    build_mesh_sub_arch,
    build_sub_arch,
    build_weight_array_sub_arch,
    load_device_catalog,
    sub_arch_key,
)


# --------------------------------------------------------------------------- #
# 表达式求值（对应 onnarchsim.workflow.utils.evaluate_factors）。
# --------------------------------------------------------------------------- #

_ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "exp": math.exp,
    "log2": math.log2,
    "log10": math.log10,
    "pow": math.pow,
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "max": max,
    "min": min,
}


def evaluate_factors(
    expression: Any,
    tiles: int,
    cores: int,
    height: int,
    width: int,
    wavelength: int = 1,
) -> int:
    """对 R/H/C/W/N 求值一个 SimPhony 缩放表达式。"""
    if isinstance(expression, int) and not isinstance(expression, bool):
        return expression
    if expression is None:
        return 1
    text = str(expression)
    text = re.sub(r"\bR\b", str(tiles), text)
    text = re.sub(r"\bH\b", str(height), text)
    text = re.sub(r"\bC\b", str(cores), text)
    text = re.sub(r"\bW\b", str(width), text)
    text = re.sub(r"\bN\b", str(wavelength), text)
    try:
        result = eval(text, {"__builtins__": None}, _ALLOWED_FUNCTIONS)  # noqa: S307
    except (SyntaxError, ZeroDivisionError, NameError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid scaling expression {expression!r}: {exc}") from exc
    return int(result)


def _parse_device_name(device: str) -> Tuple[str, str, str, str, str]:
    """把 ``part_type_index_instance-name`` 拆分为五个组成部分。"""
    identifier, name = device.split("-", 1)
    split_right = identifier.rsplit("_", 2)
    if len(split_right) != 3:
        raise ValueError(f"Unexpected device format: {device}")
    part_and_type, index, instance = split_right
    part, type_ = part_and_type.split("_", 1)
    return part, type_, index, instance, name


# --------------------------------------------------------------------------- #
# 用于最长路径 / 拓扑布局的最小有向图。
# --------------------------------------------------------------------------- #

class _DiGraph:
    def __init__(self) -> None:
        self._adj: Dict[str, List[str]] = {}
        self._weights: Dict[Tuple[str, str], float] = {}

    def add_node(self, node: str) -> None:
        self._adj.setdefault(node, [])

    def add_edge(self, src: str, dst: str, weight: float) -> None:
        self.add_node(src)
        self.add_node(dst)
        self._adj[src].append(dst)
        self._weights[(src, dst)] = weight

    def nodes(self) -> List[str]:
        return list(self._adj.keys())

    def neighbors(self, node: str) -> List[str]:
        return self._adj.get(node, [])

    def weight(self, src: str, dst: str) -> float:
        return self._weights.get((src, dst), 0.0)

    def in_degree(self, node: str) -> int:
        return sum(1 for targets in self._adj.values() if node in targets)

    def topological_sort(self) -> List[str]:
        in_degree = {node: self.in_degree(node) for node in self._adj}
        queue = [node for node, degree in in_degree.items() if degree == 0]
        order: List[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in self.neighbors(node):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return order



# --------------------------------------------------------------------------- #
# 器件计数（对应 load_device_and_count）。
# --------------------------------------------------------------------------- #

def _chip_type(instance: str, chip_mapping: Optional[Dict[str, Any]]) -> str:
    if not chip_mapping:
        return "off_chip"
    for chip, instances in chip_mapping.items():
        if instance in instances:
            return chip
    return "off_chip"


def load_required_devices(
    sub_arch: Dict[str, Any],
    device_catalog: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str], Dict[str, int]]:
    """返回 (components, operand_1, operand_2, temporal_shared)。"""
    tiles = sub_arch["tiles"]
    cores = sub_arch["cores_per_tile"]
    core_cfg = sub_arch["core"]
    height = core_cfg["height"]
    width = core_cfg["width"]
    wavelength = core_cfg["num_wavelength"]

    components: Dict[str, Dict[str, Any]] = {}
    operand_1: List[str] = []
    operand_2: List[str] = []
    temporal_shared: Dict[str, int] = {}

    def _load_level(config: Dict[str, Any], prefix: str) -> None:
        devices = config.get("devices", {})
        netlist = config.get("netlist", {})
        instances = netlist.get("instances", {})
        scaling_rules = netlist.get("scaling_rules") or {}
        chip_mapping = netlist.get("chip_mapping") or {}
        operand_1_enc = netlist.get("operand_1") or []
        operand_2_enc = netlist.get("operand_2") or []
        temporal_cfg = netlist.get("temporal_accum_factor") or {}
        temporal_devices = temporal_cfg.get("devices") or []
        temporal_duration = temporal_cfg.get("duration", 1)

        for instance, choice in instances.items():
            device_type, index = choice[0], int(choice[1])
            chip = _chip_type(instance, chip_mapping)
            if device_type in devices:
                device_name = devices[device_type][index]
                final_name = f"{prefix}_{device_type}_{index}_{instance}-{device_name}"
                body = device_catalog.get(device_type, {}).get(device_name, {})
                cfgs = dict(body.get("cfgs", {}))
                static_power = cfgs.get("static_power", 0) or 0
                dynamic_power = cfgs.get("power", cfgs.get("dynamic_power", 0)) or 0
                area = _device_area(cfgs)

                scaling = scaling_rules.get(instance)
                count = _device_count(
                    scaling, tiles, cores, height, width, wavelength
                )

                if final_name not in components:
                    components[final_name] = {
                        "count": count,
                        "static_power": float(static_power),
                        "dynamic_power": float(dynamic_power),
                        "area": float(area),
                        "chip_type": chip,
                        "cfgs": cfgs,
                    }
                else:
                    components[final_name]["count"] += count

                if instance in operand_1_enc:
                    operand_1.append(final_name)
                if instance in operand_2_enc:
                    operand_2.append(final_name)
                if instance in temporal_devices:
                    temporal_shared[final_name] = temporal_duration
            elif prefix == "core" and device_type == "node":
                components["node"] = {
                    "instance": instance,
                    "chip_type": _chip_type(instance, chip_mapping),
                }
            else:
                raise ValueError(
                    f"Device type {device_type} not found in {prefix} config"
                )

    _load_level(core_cfg, "core")
    _load_level(core_cfg.get("node", {}), "node")
    return components, operand_1, operand_2, temporal_shared


def _device_area(cfgs: Dict[str, Any]) -> float:
    if "area" in cfgs and cfgs["area"] is not None:
        return float(cfgs["area"])
    if "width" in cfgs and "length" in cfgs:
        return float(cfgs["width"]) * float(cfgs["length"])
    if "width" in cfgs or "length" in cfgs:
        return 0.0
    raise ValueError("Area not found in device cfgs")


def _device_count(
    scaling: Any,
    tiles: int,
    cores: int,
    height: int,
    width: int,
    wavelength: int,
) -> int:
    if scaling is None:
        return width * height * tiles * cores
    return evaluate_factors(scaling, tiles, cores, height, width, wavelength)


# --------------------------------------------------------------------------- #
# 功耗初始化（对应 initalize_all_devices_power）。
# --------------------------------------------------------------------------- #

def _cal_dac_power(cfgs: Dict[str, Any], in_bit: int, freq: float) -> float:
    prec = cfgs.get("prec", in_bit)
    if cfgs.get("FoM") is not None:
        return 2 * float(cfgs["FoM"]) * in_bit * freq * 1e-3
    base = float(cfgs.get("power", cfgs.get("dynamic_power", 0)) or 0)
    scale = (2 ** in_bit / (in_bit + 1)) / (2 ** int(prec) / (int(prec) + 1))
    return base * scale * freq / float(cfgs.get("sample_rate", 1))


def _cal_adc_power(cfgs: Dict[str, Any], out_bit: int, freq: float) -> float:
    prec = int(cfgs.get("prec", out_bit))
    base = float(cfgs.get("power", cfgs.get("dynamic_power", 0)) or 0)
    ratio = freq / float(cfgs.get("sample_rate", 1))
    if cfgs.get("type") == "flash":
        return base * ratio * ((2 ** out_bit - 1) / (2 ** prec - 1))
    # sar（默认）
    return base * ratio * (out_bit / prec)


def _cal_mzm_power(cfgs: Dict[str, Any], freq: float) -> float:
    return float(cfgs.get("efficiency", 0)) * freq * 1e-3 * int(cfgs.get("testing_bit", 1))


def _cal_laser_power(
    laser_cfgs: Dict[str, Any],
    in_bit: int,
    insertion_loss: float,
    sensitivity: float,
    extinction_ratio: float,
    distribution_factor: float,
) -> float:
    wall_plug_eff = float(laser_cfgs.get("wall_plug_eff", 1.0) or 1.0)
    p_laser_dbm = insertion_loss + sensitivity + 10 * math.log10(max(distribution_factor, 1))
    power = 10 ** (p_laser_dbm / 10) / wall_plug_eff * 2 ** in_bit
    extinction_scale = 1.0 / (1.0 - 0.1 ** (extinction_ratio / 10)) if extinction_ratio > 0 else 1.0
    return power * extinction_scale


def _link_budget(
    sub_arch: Dict[str, Any],
    catalog: Dict[str, Dict[str, Any]],
    core_il_db: float,
) -> Dict[str, Any]:
    """论文 eq.(1) 的链路预算：临界路径 IL → 最低激光功率 + 光信噪比（SNR）。

    论文（arXiv:2411.13715）eq.(1)：
        P_laser = 10^((S+IL)/10) · 2^bin / (η_WPE · (1 − 0.1^(ER/10)))
    其中 S 为 PD 灵敏度、IL 为关键路径插损、ER 为调制消光比、η_WPE 为激光器
    墙插效率。SNR 由 ``2^bin`` 幅度余量隐含：为分辨 bin 位输入电平，接收光功率
    须高出灵敏度 2^bin 倍，即 SNR_db = 10·log10(2^bin) ≈ 3.01·bin（功率域）。
    """
    core = sub_arch["core"]
    sensitivity = -27.0
    er = 0.0
    wpe = 1.0

    def _scan(cfg: Dict[str, Any]) -> None:
        nonlocal sensitivity, er, wpe
        netlist = cfg.get("netlist", {}) or {}
        instances = netlist.get("instances", {}) or {}
        devices = cfg.get("devices", {}) or {}
        for choice in instances.values():
            if not (isinstance(choice, (list, tuple)) and choice):
                continue
            dtype, idx = str(choice[0]), int(choice[1])
            variants = devices.get(dtype) or []
            if idx >= len(variants):
                continue
            cfgs = (catalog.get(dtype, {}).get(variants[idx], {}) or {}).get("cfgs", {}) or {}
            if dtype == "photodetector":
                sensitivity = max(sensitivity, float(cfgs.get("sensitivity", -27) or -27))
            elif dtype == "mzm":
                er = max(er, float(cfgs.get("extinction_ratio", 0) or 0))
            elif "laser" in dtype:
                wpe = float(cfgs.get("wall_plug_eff", 1.0) or 1.0)

    _scan(core)
    node = core.get("node")
    if isinstance(node, dict):
        _scan(node)

    netlist = core.get("netlist", {}) or {}
    tiles = sub_arch["tiles"]
    cores = sub_arch["cores_per_tile"]
    height = core.get("height")
    width = core.get("width")
    wavelength = core.get("num_wavelength", 1)
    in_bit = int((core.get("precision") or {}).get("in_bit", 8) or 8)

    distribution_factor = evaluate_factors(
        netlist.get("laser_power_distribution", 1), tiles, cores, height, width, wavelength
    )
    p_laser_dbm = core_il_db + sensitivity + 10 * math.log10(max(distribution_factor, 1))
    extinction_scale = 1.0 / (1.0 - 0.1 ** (er / 10)) if er > 0 else 1.0
    laser_power_mw = 10 ** (p_laser_dbm / 10) / wpe * 2 ** in_bit * extinction_scale

    return {
        "critical_path_il_db": round(float(core_il_db), 4),
        "receiver_sensitivity_dbm": round(float(sensitivity), 4),
        "input_bit": in_bit,
        "extinction_ratio_db": round(float(er), 4),
        "wall_plug_efficiency": round(float(wpe), 4),
        "required_laser_power_dbm": round(float(p_laser_dbm), 4),
        "required_laser_power_mw": round(float(laser_power_mw), 6),
        "snr_db": round(float(10 * math.log10(2 ** in_bit)), 4),
    }


def initialize_device_power(
    sub_arch: Dict[str, Any],
    components: Dict[str, Dict[str, Any]],
    insertion_loss: float,
) -> Dict[str, Dict[str, Any]]:
    tiles = sub_arch["tiles"]
    cores = sub_arch["cores_per_tile"]
    core_cfg = sub_arch["core"]
    height = core_cfg["height"]
    width = core_cfg["width"]
    wavelength = core_cfg["num_wavelength"]
    freq = core_cfg["work_freq"]
    netlist = core_cfg.get("netlist", {})
    distribution_factor = evaluate_factors(
        netlist.get("laser_power_distribution", 1), tiles, cores, height, width, wavelength
    )
    in_bit = core_cfg.get("precision", {}).get("in_bit", 8)
    out_bit = core_cfg.get("precision", {}).get("out_bit", 8)

    extinction_ratio = 0.0
    sensitivity = -27.0
    laser_cfgs: Dict[str, Any] = {}
    # 网格权重移相器静态保持；DODE 微调器动态调谐。
    ps_static = bool((core_cfg.get("node", {}) or {}).get("weight_stationary_phase_shifter", False))

    for device, device_cfg in components.items():
        if device == "node":
            continue
        _, type_, _, _, _ = _parse_device_name(device)
        cfgs = device_cfg.get("cfgs", {})

        if "adc" in type_:
            device_cfg["dynamic_power"] = _cal_adc_power(cfgs, out_bit, freq)
        elif "dac" in type_:
            device_cfg["dynamic_power"] = _cal_dac_power(cfgs, in_bit, freq)
        elif "mzm" in type_:
            device_cfg["dynamic_power"] = _cal_mzm_power(cfgs, freq)
            er = cfgs.get("extinction_ratio") or 0
            extinction_ratio = max(extinction_ratio, float(er))
        elif "phase_shifter" in type_:
            p_pi = float(cfgs.get("P_pi", 0) or 0)
            if ps_static:
                # 保持权重的移相器：持续静态耗散。
                device_cfg["static_power"] = p_pi
                device_cfg["dynamic_power"] = 0.0
            else:
                device_cfg["dynamic_power"] = p_pi
                device_cfg["tuning_cycles"] = float(cfgs.get("response_time", 0) or 0) * freq
        elif "photodetector" in type_:
            sensitivity = max(sensitivity, float(cfgs.get("sensitivity", -27) or -27))
        elif "on_chip_laser" in type_ or "off_chip_laser" in device:
            laser_cfgs = cfgs
            continue

        device_cfg.setdefault("dynamic_power", 0.0)
        device_cfg.setdefault("static_power", 0.0)
        device_cfg.setdefault("tuning_cycles", 1)
        for key in list(device_cfg.keys()):
            if key not in {"count", "static_power", "dynamic_power", "tuning_cycles", "chip_type"}:
                device_cfg.pop(key, None)
        device_cfg["static_energy"] = device_cfg["static_power"] / freq
        device_cfg["dynamic_energy"] = device_cfg["dynamic_power"] / freq

    if laser_cfgs:
        static_power = _cal_laser_power(
            laser_cfgs, in_bit, insertion_loss, sensitivity, extinction_ratio, distribution_factor
        )
        laser_name = next(
            (name for name, cfg in components.items() if "laser" in name), None
        )
        if laser_name is not None:
            cfg = components[laser_name]
            cfg["static_power"] = static_power
            cfg["dynamic_power"] = 0.0
            cfg["tuning_cycles"] = 1
            cfg["static_energy"] = static_power / freq
            cfg["dynamic_energy"] = 0.0

    return components


# --------------------------------------------------------------------------- #
# 能耗（对应 reuse_determinator + energy_calculator）。
# --------------------------------------------------------------------------- #

def _reuse_determinator(
    dataflow: str,
    iter_m: int,
    iter_n: int,
    iter_d: int,
    max_operand_1_cycles: float,
    max_operand_2_cycles: float,
    max_cycles: float,
    forward_w: int,
    forward_x: int,
) -> Tuple[float, float, float]:
    if dataflow == "weight_stationary":
        operand_1_reuse, operand_2_reuse = 1, iter_m
        max_cycle = (iter_m * iter_n * iter_d) * max_operand_1_cycles + iter_n * iter_d * max_cycles
    elif dataflow == "input_stationary":
        operand_1_reuse, operand_2_reuse = iter_n, 1
        max_cycle = (iter_m * iter_n * iter_d) * max_operand_2_cycles + iter_m * iter_d * max_cycles
    elif dataflow == "output_stationary":
        operand_1_reuse, operand_2_reuse = 1, 1
        max_cycle = (iter_m * iter_n * iter_d) * max_cycles + iter_m * iter_n * max_cycles
    else:
        raise ValueError(f"Unsupported dataflow: {dataflow}")
    operand_2_reuse *= forward_w
    operand_1_reuse *= forward_x
    max_cycle = max_cycle * forward_w * forward_x
    return 1 / operand_2_reuse, 1 / operand_1_reuse, max_cycle


def energy_calculator(
    sub_arch: Dict[str, Any],
    components: Dict[str, Dict[str, Any]],
    operand_1: List[str],
    operand_2: List[str],
    temporal_shared: Dict[str, int],
    cycles: Dict[str, Tuple[int, ...]],
    insertion_loss: float,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float], Dict[str, float]]:
    """逐器件能耗，外加一个 **原生的** 逐层能耗分解。

    累积循环已遍历每一层，因此逐层合计在此处直接采集。这使智能体按层能耗排序从按循环
    数比例估算的 *估计值* 变成真实的仿真量。
    """
    dataflow = sub_arch.get("dataflow", "weight_stationary")
    freq = sub_arch["core"]["work_freq"]
    # 论文 Tab. I 的 I 因子：range-restricted PTC 需多次前向才能得到全值域输出
    # （MRR=2、PCM=4、MZI/butterfly/TeMPO=1）。I 同时放大计算周期与能耗。
    range_forwards = int((sub_arch.get("core", {}) or {}).get("range_forwards") or 1)
    core_cfg = sub_arch.get("core", {}) or {}
    reconfig_time_ns = float(core_cfg.get("reconfig_time_ns", 0.0) or 0.0)
    weight_count = int(core_cfg.get("weight_count", 0) or 0)

    devices = initialize_device_power(sub_arch, components, insertion_loss)

    if not operand_1 or not operand_2:
        raise ValueError("No operand-encoding devices found in the architecture config")

    max_op1_cycles = max(devices[d]["tuning_cycles"] for d in operand_1)
    max_op2_cycles = max(devices[d]["tuning_cycles"] for d in operand_2)
    max_cycles = max(max_op1_cycles, max_op2_cycles)

    latency = {
        "total_tuning_latency": 0.0,
        "total_static_latency": 0.0,
        "total_reconfig_latency": 0.0,
    }
    per_layer_energy: Dict[str, float] = {}
    operand_1_set = set(operand_1)
    operand_2_set = set(operand_2)

    for layer_name, cycle in cycles.items():
        iter_n, iter_d, iter_m, _, _, _, forward_w, forward_x = cycle
        switching_cycles = iter_n * iter_d * iter_m * range_forwards
        layer_energy = 0.0
        op2_duty, op1_duty, max_cycle = _reuse_determinator(
            dataflow, iter_m, iter_n, iter_d, max_op1_cycles, max_op2_cycles, max_cycles,
            forward_w, forward_x,
        )
        max_cycle *= range_forwards
        # 论文重配置延时惩罚：weight-stationary 下换权重受器件重编程速度瓶颈
        # （TO=10µs、PCM=100ns）；cycle penalty = 权重数 × reconfig_time_ns × freq，
        # 且按 I 因子放大（τ_tot = τ_load + τ_write + I(τ_comp + τ_reconfig)）。
        if dataflow == "weight_stationary" and reconfig_time_ns > 0.0:
            reconfig_cycles = int(round(weight_count * reconfig_time_ns * freq * range_forwards))
            latency["total_reconfig_latency"] += reconfig_cycles
        for device, device_cfg in devices.items():
            dynamic = device_cfg["dynamic_energy"] * switching_cycles * device_cfg["count"]
            duration = temporal_shared.get(device)
            if duration:
                dynamic /= duration
            if device in operand_1_set:
                dynamic *= op1_duty
            if device in operand_2_set:
                dynamic *= op2_duty
            static = device_cfg["static_energy"] * max_cycle * device_cfg["count"]
            device_cfg["total_dynamic_energy"] = device_cfg.get("total_dynamic_energy", 0.0) + dynamic
            device_cfg["total_static_energy"] = device_cfg.get("total_static_energy", 0.0) + static
            device_cfg["total_energy"] = (
                device_cfg.get("total_energy", 0.0) + dynamic + static
            )
            layer_energy += dynamic + static
        per_layer_energy[layer_name] = layer_energy
        latency["total_tuning_latency"] += switching_cycles
        latency["total_static_latency"] += max_cycle

    latency["total_tuning_latency"] = latency["total_tuning_latency"] / freq * 1e-9
    latency["total_static_latency"] = latency["total_static_latency"] / freq * 1e-9
    latency["total_reconfig_latency"] = latency["total_reconfig_latency"] / freq * 1e-9
    return devices, latency, per_layer_energy


def chip_energy_calculator(devices: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Dict[str, Any]] = {}
    for device, cfg in devices.items():
        chip = cfg.get("chip_type", "off_chip")
        bucket = result.setdefault(chip, {"devices": [], "total_dynamic_energy": 0.0, "total_static_energy": 0.0})
        bucket["devices"].append(device)
        bucket["total_dynamic_energy"] += cfg.get("total_dynamic_energy", 0.0)
        bucket["total_static_energy"] += cfg.get("total_static_energy", 0.0)
    return result


# --------------------------------------------------------------------------- #
# 面积（对应 initalize_all_devices_area + area_calculator + 包围盒）。
# --------------------------------------------------------------------------- #

def _topo_levels_sorted(
    graph: _DiGraph,
    dims_by_instance: Dict[str, Dict[str, float]],
) -> List[str]:
    """按层级分组的拓扑顺序，每层按器件长度排序。"""
    in_degree = {node: graph.in_degree(node) for node in graph.nodes()}
    queue = [node for node in graph.nodes() if in_degree[node] == 0]
    order: List[str] = []
    while queue:
        level = list(queue)
        queue = []
        for node in level:
            for successor in graph.neighbors(node):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        level.sort(key=lambda n: dims_by_instance.get(n, {}).get("length", 0.0))
        order.extend(level)
    return order


def _node_bounding_box(
    node_cfg: Dict[str, Any],
    catalog: Dict[str, Dict[str, Any]],
    tiles: int,
    cores: int,
    height: int,
    width: int,
) -> Tuple[float, float, float, int]:
    """布局感知的节点包围盒（对应 default_bounding_box_estimator）。"""
    spacing_flag = node_cfg.get("device_spacing_flag", False)
    node_spacing_flag = node_cfg.get("node_spacing_flag", False)
    device_spacing = node_cfg.get("device_spacing") or [5, 5, 5, 5]
    node_spacing = node_cfg.get("node_spacing") or [5, 5, 5, 5]

    dims = (tiles, cores, height, width, 1)
    graph, _ = _construct_graph(node_cfg, catalog, dims)

    netlist = node_cfg.get("netlist", {})
    instances = netlist.get("instances", {})
    devices = node_cfg.get("devices", {})
    dims_by_instance: Dict[str, Dict[str, float]] = {}
    for instance, choice in instances.items():
        device_type, index = choice[0], int(choice[1])
        name = devices[device_type][index]
        cfgs = catalog.get(device_type, {}).get(name, {}).get("cfgs", {})
        dims_by_instance[instance] = {
            "length": float(cfgs.get("length", 0) or 0),
            "width": float(cfgs.get("width", 0) or 0),
        }

    lengths = [v["length"] for v in dims_by_instance.values()]
    if not lengths:
        return 0.0, 0.0, 0.0, width * height * tiles * cores

    max_length = max(lengths) + (device_spacing[2] + device_spacing[3] if spacing_flag else 0)

    levels: List[List[float]] = []
    current: List[float] = []
    current_length = 0.0
    for instance in _topo_levels_sorted(graph, dims_by_instance):
        info = dims_by_instance[instance]
        added = info["length"] + (device_spacing[2] + device_spacing[3] if spacing_flag else 0)
        if current_length + added <= max_length + 1e-9:
            current.append(info["width"])
            current_length += added
        else:
            levels.append(current)
            current = [info["width"]]
            current_length = added
    if current:
        levels.append(current)

    total_height = sum(
        max(level) + (device_spacing[0] + device_spacing[1] if spacing_flag else 0)
        for level in levels
    )
    total_width = max_length
    if node_spacing_flag:
        total_width += node_spacing[2] + node_spacing[3]
        total_height += node_spacing[0] + node_spacing[1]
    area = total_width * total_height
    count = width * height * tiles * cores
    return area, total_width, total_height, count


def area_calculator(
    sub_arch: Dict[str, Any],
    components: Dict[str, Dict[str, Any]],
    catalog: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    core_cfg = sub_arch["core"]
    tiles = sub_arch["tiles"]
    cores = sub_arch["cores_per_tile"]
    height = core_cfg["height"]
    width = core_cfg["width"]

    node_chip = components.pop("node", {}).get("chip_type", "off_chip")
    netlist_ports = (core_cfg.get("netlist") or {}).get("ports_num") or {}
    for device, cfg in components.items():
        _, type_, _, instance, _ = _parse_device_name(device)
        cfgs = cfg.get("cfgs", {})
        cfg["width"] = float(cfgs.get("width", math.sqrt(cfg.get("area", 0) or 0)) or 0)
        cfg["length"] = float(cfgs.get("length", math.sqrt(cfg.get("area", 0) or 0)) or 0)
        if "mmi" in type_:
            # 对应 onnarchsim 的 cal_MMI_area：面积随实例端口数缩放（取自
            # netlist.ports_num，缺失时回退到器件自身的 ports_num）。
            ports = netlist_ports.get(instance) or cfgs.get("ports_num", 10)
            port = evaluate_factors(ports, tiles, cores, height, width, 1)
            port_scaling = port / float(cfgs.get("ports_num", 10))
            cfg["area"] = float(cfg.get("area", 0.0) or 0.0) * port_scaling
        for key in list(cfg.keys()):
            if key not in {"count", "area", "chip_type", "width", "length"}:
                cfg.pop(key, None)

    node_area, node_width, node_length, count = _node_bounding_box(
        core_cfg.get("node", {}), catalog, tiles, cores, height, width
    )
    components["node"] = {
        "area": node_area,
        "width": node_width,
        "length": node_length,
        "count": count,
        "chip_type": node_chip,
    }
    for device, cfg in components.items():
        cfg["total_area"] = cfg.get("area", 0.0) * cfg.get("count", 0)
    return components


def chip_area_calculator(devices: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Dict[str, Any]] = {}
    for device, cfg in devices.items():
        if device != "node" and device.startswith("node"):
            continue
        chip = cfg.get("chip_type", "off_chip")
        bucket = result.setdefault(chip, {"devices": [], "total_area": 0.0})
        bucket["devices"].append(device)
        bucket["total_area"] += cfg.get("total_area", 0.0)
    return result


# --------------------------------------------------------------------------- #
# 插入损耗（对应 architecture_insertion_loss）。
# --------------------------------------------------------------------------- #

def _insertion_loss_of(device_choice: Tuple[str, int], devices: Dict[str, Any], catalog: Dict[str, Dict[str, Any]]) -> float:
    device_type, index = device_choice[0], int(device_choice[1])
    if device_type not in devices:
        raise ValueError(f"Device type {device_type} not found in config")
    device_name = devices[device_type][index]
    cfgs = catalog.get(device_type, {}).get(device_name, {}).get("cfgs", {})
    return float(cfgs.get("insertion_loss", 0) or 0)


def _replication_count(instance: str, replication: Dict[str, Any], dims: Tuple[int, int, int, int, int]) -> int:
    if replication and instance in replication:
        return evaluate_factors(replication[instance], *dims)
    return 1


def _parse_node(node_str: str, replication_counts: Dict[str, int]) -> Tuple[str, str]:
    if "(" in node_str:
        instance_id, idx = node_str.split("(")
        idx = idx.strip(")")
        if idx == "s":
            index = 0
        elif idx == "e":
            index = replication_counts[instance_id] - 1
        else:
            index = int(idx)
        return instance_id, f"{instance_id}_{index}"
    return node_str, node_str


def _construct_graph(
    config: Dict[str, Any],
    catalog: Dict[str, Dict[str, Any]],
    dims: Tuple[int, int, int, int, int],
    node_il: Optional[float] = None,
) -> Tuple[_DiGraph, Dict[str, float]]:
    netlist = config.get("netlist", {})
    instances = netlist.get("instances", {})
    devices = config.get("devices", {})
    nets = netlist.get("nets", {})
    replication = netlist.get("replication") or {}

    instance_weights: Dict[str, float] = {}
    replication_counts: Dict[str, int] = {}
    for instance, choice in instances.items():
        if node_il is not None and choice[0] == "node":
            instance_weights[instance] = node_il
        else:
            instance_weights[instance] = _insertion_loss_of(choice, devices, catalog)
        replication_counts[instance] = _replication_count(instance, replication, dims)

    graph = _DiGraph()
    for instance_id, count in replication_counts.items():
        if count > 1:
            for idx in range(count):
                graph.add_node(f"{instance_id}_{idx}")
        else:
            graph.add_node(instance_id)

    for instance_id, count in replication_counts.items():
        if count > 1:
            for idx in range(count - 1):
                graph.add_edge(f"{instance_id}_{idx}", f"{instance_id}_{idx + 1}", instance_weights[instance_id])

    for _net, (src, dst) in nets.items():
        src_id, src_name = _parse_node(src, replication_counts)
        dst_id, dst_name = _parse_node(dst, replication_counts)
        if src_id in instance_weights and dst_id in instance_weights:
            graph.add_edge(src_name, dst_name, instance_weights[dst_id])
        else:
            raise ValueError(f"Source {src} or destination {dst} not matched")
    return graph, instance_weights


def _longest_path(
    graph: _DiGraph,
    instance_weights: Dict[str, float],
    instances: Dict[str, Tuple[str, int]],
    target_device_type: str,
) -> Tuple[List[str], float]:
    topo = graph.topological_sort()
    dist = {node: float("-inf") for node in graph.nodes()}
    path: Dict[str, List[str]] = {node: [] for node in graph.nodes()}

    for start in [node for node in graph.nodes() if graph.in_degree(node) == 0]:
        node_id = start.split("_")[0] if "_" in start else start
        dist[start] = instance_weights.get(node_id, 0.0)

    for u in topo:
        if dist[u] != float("-inf"):
            for v in graph.neighbors(u):
                weight = graph.weight(u, v)
                if dist[v] < dist[u] + weight:
                    dist[v] = dist[u] + weight
                    path[v] = path[u] + [u]

    targets = [instance for instance, choice in instances.items() if choice[0] == target_device_type]
    best_path: List[str] = []
    best_loss = float("-inf")
    for target in targets:
        target_nodes = [node for node in dist if node == target or node.startswith(f"{target}_")]
        for target_node in target_nodes:
            if dist[target_node] > best_loss:
                best_loss = dist[target_node]
                best_path = path[target_node] + [target_node]
    return best_path, best_loss


def insertion_loss(
    sub_arch: Dict[str, Any],
    catalog: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    core_cfg = sub_arch["core"]
    node_cfg = core_cfg.get("node", {})
    dims = (sub_arch["tiles"], sub_arch["cores_per_tile"], core_cfg["height"], core_cfg["width"], core_cfg["num_wavelength"])

    node_target = node_cfg.get("netlist", {}).get("insertion_loss_final_device", "photodetector")
    node_graph, node_weights = _construct_graph(node_cfg, catalog, dims)
    node_path, node_il = _longest_path(node_graph, node_weights, node_cfg["netlist"]["instances"], node_target)

    core_target = core_cfg.get("netlist", {}).get("insertion_loss_final_device", "node")
    core_graph, core_weights = _construct_graph(core_cfg, catalog, dims, node_il=node_il)
    core_path, core_il = _longest_path(core_graph, core_weights, core_cfg["netlist"]["instances"], core_target)

    return {
        "node_path": node_path,
        "node_insertion_loss": node_il,
        "core_path": core_path,
        "core_insertion_loss": core_il,
        "node_path_detail": _path_detail(
            node_path, node_cfg["netlist"]["instances"], node_cfg["devices"], catalog, node_weights
        ),
        "core_path_detail": _path_detail(
            core_path, core_cfg["netlist"]["instances"], core_cfg["devices"], catalog, core_weights
        ),
    }


def _path_detail(
    path: List[str],
    instances: Dict[str, Tuple[str, int]],
    devices: Dict[str, List[str]],
    catalog: Dict[str, Dict[str, Any]],
    instance_weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    """把插入损耗路径展开为逐器件的明细。

    每个元素给出物理器件名、其库变体以及各自的插入损耗贡献（dB）。这是光学层智能体
    沿光路定位损耗所需的证据，而非只看到整条路径的总损耗。
    """
    detail: List[Dict[str, Any]] = []
    for node in path:
        # 只去掉末尾的复制索引（"i0_3" -> "i0"）。网格实例名为 "i_mzi"/"i_pd"/...，
        # 若用 split("_")[0] 会被截断成 "i" 并静默丢失整个分解。
        instance_id = node
        head, sep, tail = str(node).rpartition("_")
        if sep and head and tail.isdigit():
            instance_id = head
        choice = instances.get(instance_id)
        if choice is None:
            continue
        device_type, index = choice[0], int(choice[1])
        if device_type == "node":
            # 核心网表把整个节点子电路折叠为一个伪器件，其权重为节点总损耗。
            device_name = "node_subcircuit"
        else:
            variants = devices.get(device_type) or []
            device_name = variants[index] if index < len(variants) else None
        loss = float(instance_weights.get(instance_id, 0.0) or 0.0)
        detail.append({
            "instance": node,
            "device_type": device_type,
            "device_name": device_name,
            "insertion_loss_dB": round(loss, 4),
        })
    return detail


# --------------------------------------------------------------------------- #
# 解析内存。
# --------------------------------------------------------------------------- #

def memory_cost(
    sub_arch: Dict[str, Any],
    cycles: Dict[str, Tuple[int, ...]],
    layer_sizes: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Dict[str, float]], float, Dict[str, Any]]:
    """解析的内存时延 / 能耗（无需 CACTI 二进制）。

    使用架构配置中的 HBM 读/写能耗与带宽以及 GLB/RF 缓存大小；在完全离线的前提下
    复现 SimPhony 内存层次报告的结构。
    """
    memory = sub_arch.get("memory", {})
    hbm = memory.get("HBM", {})
    read_energy = float(hbm.get("read_energy", 62.4) or 0)  # pJ/Byte
    write_energy = float(hbm.get("write_energy", 62.4) or 0)  # pJ/Byte
    bandwidth = float(hbm.get("bandwidth", 1024) or 1)  # GB/s

    layer_sizes = layer_sizes if layer_sizes is not None else {}
    latency: Dict[str, Dict[str, float]] = {}
    energy: Dict[str, float] = {}
    for layer, cycle in cycles.items():
        size = layer_sizes.get(layer, {})
        input_bytes = size.get("input_size", [0])[0]
        output_bytes = size.get("output_size", [0])[0]
        weight_bytes = size.get("weight_size", [0])[0]
        # 每 iter_N 次迭代重新读取权重（粗略模型）。
        iter_n = cycle[0]
        total_read = input_bytes + weight_bytes * iter_n
        total_write = output_bytes
        load_latency = (total_read + total_write) / (bandwidth * 1e9)
        load_energy = total_read * read_energy + total_write * write_energy
        latency[layer] = {"input_latency_s": load_latency, "total_size": total_read + total_write}
        energy[layer] = load_energy

    spec = {
        "HBM": {"read_energy": read_energy, "write_energy": write_energy, "bandwidth": bandwidth},
        "caches": {k: v for k, v in memory.items() if k != "HBM"},
    }
    return latency, energy, spec


# --------------------------------------------------------------------------- #
# 顶层求值。
# --------------------------------------------------------------------------- #

def workload_arch_context(sub_arch: Dict[str, Any]) -> Dict[str, Any]:
    """工作负载循环算术所依赖的架构输入。

    ``num_wavelength`` 改变 ``iter_D``；``miniblock`` 与 ``dim_map`` 决定每个矩阵维度
    被拆分的迭代次数。从被评估的子架构读取这些值，使真实工作负载的逐层曲线会
    *随硬件编辑而变化*，而非保持不变。

    ``miniblock`` 采用 ``[tiles, cores_per_tile, matrix_height, matrix_width]``，
    与真实 ONNArchSimulator 的 ``arch_blocks`` 一致：并行权重阵列的 ``matrix_height=1``
    （单输出向量）会放大 ``iter_N``，从而让吞吐归一化区分向量与矩阵阵列。
    """
    core = sub_arch.get("core") or {}
    tiles = int(sub_arch.get("tiles", 1) or 1)
    cores = int(sub_arch.get("cores_per_tile", 1) or 1)
    matrix_h = int(core.get("matrix_height") or core.get("height", 1) or 1)
    matrix_w = int(core.get("matrix_width") or core.get("width", 1) or 1)
    return {
        "num_wavelength": int(core.get("num_wavelength", 1) or 1),
        "dataflow": sub_arch.get("dataflow"),
        "op_type": sub_arch.get("op_type"),
        "miniblock": [tiles, cores, matrix_h, matrix_w],
    }


def evaluate(
    hardware: Dict[str, Any],
    simphony_root: str,
    device_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
    workload: Optional[Dict[str, Any]] = None,
    workload_name: Optional[str] = None,
) -> Dict[str, Any]:
    """求值一个光子设计点并返回细粒度报告。

    Args:
        workload: 预构建的工作负载字典（见
            :func:`photon_tuner.core.workload.build_workload`）。缺省时由
            ``workload_name`` 从 ``configs/workloads.json`` 中选择。
        workload_name: 已注册的工作负载名；默认为 ``configs/workloads.json`` 的
            ``default`` 项。
    """
    hardware = materialize_hardware(hardware)
    catalog = device_catalog or load_device_catalog(f"{simphony_root}/configs/devices")
    core_kind = str((hardware.get("arch", {}).get("topology", {}) or {}).get("core_kind", "dptc_crossbar"))
    if core_kind.startswith("mzi_mesh_"):
        sub_arch = build_mesh_sub_arch(hardware, simphony_root, catalog, core_kind)
    elif core_kind in _WEIGHT_ARRAY_SPECS:
        sub_arch = build_weight_array_sub_arch(hardware, simphony_root, catalog, core_kind)
    elif core_kind == "dptc_crossbar":
        sub_arch = build_sub_arch(hardware, simphony_root, catalog)
    else:
        raise ValueError(
            f"unknown core_kind {core_kind!r}: no netlist builder and no evaluation path; "
            f"refusing to silently fall back to the DPTC crossbar"
        )

    # 报告里的子架构键取实际拓扑名，而不是 YAML 的静态查表名（"LT"），
    # 否则报告会自称仿真的是 crossbar 基线，即使实际结构是 MZI 网格。
    arch_key = sub_arch_key(hardware)
    resolved_workload = workload or build_workload(
        name=workload_name,
        arch=workload_arch_context(sub_arch),
        simphony_root=simphony_root,
    )
    cycles = resolved_workload["cycles"]
    layer_sizes = resolved_workload["layer_sizes"]
    il = insertion_loss(sub_arch, catalog)

    components, operand_1, operand_2, temporal_shared = load_required_devices(sub_arch, catalog)
    components.pop("node", None)
    energy_devices, computation_latency, per_layer_energy = energy_calculator(
        sub_arch, components, operand_1, operand_2, temporal_shared, cycles, il["core_insertion_loss"]
    )
    chip_energy = chip_energy_calculator(energy_devices)
    total_energy = sum(v["total_energy"] for v in energy_devices.values())

    area_components, _, _, _ = load_required_devices(sub_arch, catalog)
    area_devices = area_calculator(sub_arch, area_components, catalog)
    chip_area = chip_area_calculator(area_devices)
    # 对应 SimPhony 的 simu_area 总面积：核心级器件加上布局感知的节点包围盒
    # （节点级器件面积已并入盒内）。
    total_area = sum(v["total_area"] for key, v in area_devices.items() if "node" not in key)
    total_area += area_devices["node"]["total_area"]

    memory_latency, memory_energy, memory_spec = memory_cost(sub_arch, cycles, layer_sizes)
    total_memory_latency = sum(v["input_latency_s"] for v in memory_latency.values())
    total_memory_energy = sum(memory_energy.values())
    total_latency = (
        computation_latency["total_tuning_latency"]
        + computation_latency["total_static_latency"]
        + computation_latency.get("total_reconfig_latency", 0.0)
        + total_memory_latency
    )

    # 吞吐归一化指标（跨范式 Pareto 效率轴）：能耗→每 MAC、面积→每单位吞吐 MAC/s。
    # 并行权重阵列是单输出向量（器件少但吞吐低），按吞吐归一后与矩阵阵列可比。
    macs = total_macs(cycles)
    throughput = macs / total_latency if total_latency > 0 else 0.0
    pj_per_mac = total_energy / macs if macs else total_energy
    area_per_throughput = total_area / throughput if throughput > 0 else total_area

    report = {
        "metadata": {"simulator": "SimPhony-analytic"},
        "hardware": hardware,
        "workload": {
            "name": resolved_workload.get("name"),
            "kind": resolved_workload.get("kind"),
            "partition_cycles": {arch_key: cycles},
            "layer_sizes": layer_sizes,
        },
        "insertion_loss": {arch_key: il},
        "link_budget": {arch_key: _link_budget(sub_arch, catalog, il["core_insertion_loss"])},
        "energy": {
            "by_device": {arch_key: energy_devices},
            "by_chip_type": {arch_key: chip_energy},
            "by_layer": {arch_key: per_layer_energy},
            "total": {arch_key: total_energy},
        },
        "area": {
            "by_device": {arch_key: area_devices},
            "by_chip_type": {arch_key: chip_area},
            "total": {arch_key: total_area},
        },
        "latency": {
            "computation": {arch_key: computation_latency},
            "memory": {arch_key: total_memory_latency},
            "total": {arch_key: total_latency},
        },
        "memory": {
            "latency": {arch_key: memory_latency},
            "energy": {arch_key: memory_energy},
            "spec": memory_spec,
        },
        "metrics": {
            "energy_pj": total_energy,
            "area_um2": total_area,
            "latency_s": total_latency,
            "total_macs": macs,
            "throughput_mac_per_s": throughput,
            "pJ_per_mac": pj_per_mac,
            "area_um2_per_mac_per_s": area_per_throughput,
        },
    }
    return report
