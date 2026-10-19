"""Dependency-free analytic photonic evaluator.

This module is a *formula-level mirror* of SimPhony's device / circuit /
architecture workflows.  It reads the exact same device and core / node netlist
YAMLs that SimPhony consumes and reproduces its device counting, power
initialisation, energy, area (bounding-box), and insertion-loss (longest-path)
calculations using only the Python standard library plus PyYAML.

It exists so the whole multi-agent DSE loop can be exercised *offline*, without
PyTorch / TorchONN / mmengine / networkx.  When those dependencies are present
the :class:`photon_tuner.simulators.simphony.SimphonyAdapter` runs the real
``ONNArchSimulator`` instead; both produce the same fine-grained report schema
(see ``simphony/onnarchsim/report.py``).

The *workload* under evaluation is named and loadable (see
:mod:`photon_tuner.core.workload` and ``configs/workloads.json``).  The legacy
``synthetic`` three-layer baseline is still the default, but real encoder
workloads (``vit_base`` / ``vit_small`` / ``vit_tiny``) can be selected; their
per-layer cycle tuples are computed with SimPhony's own partition arithmetic, so
the per-layer demand profile responds to architecture edits.  A native
``energy.by_layer`` breakdown is reported alongside it.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml

from photon_tuner.core.search_space import materialize_hardware
from photon_tuner.core.topology import (
    DEFAULT_LASER_DISTRIBUTION,
    DEFAULT_SHARING_POLICY,
    DEFAULT_TEMPORAL_ACCUM,
    apply_topology,
)
from photon_tuner.core.workload import (
    DEFAULT_MINIBLOCK,
    build_workload,
    synthetic_cycles,
    synthetic_layer_sizes,
)


# --------------------------------------------------------------------------- #
# Expression evaluation (mirrors onnarchsim.workflow.utils.evaluate_factors).
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
    """Evaluate a SimPhony scaling expression over R/H/C/W/N."""
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
    """Split ``part_type_index_instance-name`` into its five components."""
    identifier, name = device.split("-", 1)
    split_right = identifier.rsplit("_", 2)
    if len(split_right) != 3:
        raise ValueError(f"Unexpected device format: {device}")
    part_and_type, index, instance = split_right
    part, type_ = part_and_type.split("_", 1)
    return part, type_, index, instance, name


# --------------------------------------------------------------------------- #
# Minimal directed graph for longest-path / topological placement.
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
# Device-library loading.
# --------------------------------------------------------------------------- #

def load_device_catalog(devicelib_root: str) -> Dict[str, Dict[str, Any]]:
    """Load every device YAML variant into ``{device_type: {name: body}}``."""
    catalog: Dict[str, Dict[str, Any]] = {}
    import glob
    import os

    for path in sorted(glob.glob(os.path.join(devicelib_root, "*", "*.yml"))):
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        for device_type, variants in data.items():
            bucket = catalog.setdefault(str(device_type), {})
            for name, body in variants.items():
                if isinstance(body, dict):
                    bucket[str(name)] = body
    return catalog


# --------------------------------------------------------------------------- #
# Sub-architecture assembly (arch params + core/node netlists + device choices).
# --------------------------------------------------------------------------- #

def _load_yaml_versioned(path: str, version: str = "v1") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    name = next(iter(data))
    versions = data[name]
    if version not in versions:
        raise ValueError(f"Version {version!r} not found in {path}")
    body = versions[version]
    body.setdefault("name", name)
    return body


def build_sub_arch(
    hardware: Dict[str, Any],
    simphony_root: str,
    device_catalog: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge a photonic ``hardware`` dict with SimPhony's core/node netlists."""
    # SimPhony's CoreLib/NodeLib move the version-level ``devices`` table into
    # the ``core`` / ``node`` sub-dict before simulation, so we mirror that here.
    dptc_body = _load_yaml_versioned(f"{simphony_root}/configs/design/cores/dptc.yml", "v1")
    dode_body = _load_yaml_versioned(f"{simphony_root}/configs/design/nodes/dode.yml", "v1")
    core_cfg = dict(dptc_body["core"])
    core_cfg["devices"] = dict(dptc_body["devices"])
    core_cfg.setdefault("name", dptc_body["name"])
    core_cfg.setdefault("version", "v1")
    node_cfg = dict(dode_body["node"])
    node_cfg["devices"] = dict(dode_body["devices"])
    node_cfg.setdefault("name", dode_body["name"])
    node_cfg.setdefault("version", "v1")

    arch = hardware.get("arch", {})
    system = arch.get("system", {})
    core = arch.get("optical_core", {})
    topology = arch.get("topology", {})
    precision = core.get("precision", {})
    # Start from the LT sub-arch memory defaults (which carry the CACTI
    # ``config_file_name`` / ``output_file_name`` the real workflow needs) and
    # overlay the searched cache / bandwidth values on top.
    lt_body = _load_yaml_versioned(
        f"{simphony_root}/configs/design/sub_architectures/LT.yml", "v1"
    )
    memory = dict(lt_body.get("memory", {}) or {})
    for level, cfg in (arch.get("memory") or {}).items():
        if isinstance(cfg, dict):
            memory.setdefault(level, {}).update(cfg)

    core_devices = dict(hardware.get("devices", {}).get("core", {}))
    node_devices = dict(hardware.get("devices", {}).get("node", {}))

    # Apply device-choice overrides to the netlist device tables.
    for device_type, device_name in core_devices.items():
        if device_type in core_cfg.get("devices", {}):
            core_cfg["devices"][device_type] = [device_name]
    for device_type, device_name in node_devices.items():
        if device_type in node_cfg.get("devices", {}):
            node_cfg["devices"][device_type] = [device_name]

    # Instance-level device-type switches (searchable "instance choices").
    # ``laser_source`` toggles the laser instance i4 between on-chip and
    # off-chip sources; ``splitter_kind`` toggles the fan-out splitter i6
    # between the waveguide laser-splitter and a 1x10 MMI.  The instance id,
    # scaling rule and net position stay identical, so the swap is structural.
    # The real ONNArchSimulator area workflow special-cases device names
    # containing "mmi" via ``netlist.ports_num`` (cal_MMI_area), so the mmi
    # case must also populate that dict.
    netlist = core_cfg.setdefault("netlist", {})
    instances = netlist.setdefault("instances", {})
    laser_source = str(core_devices.get("laser_source", "on_chip"))
    splitter_kind = str(core_devices.get("splitter_kind", "laser_splitter"))
    i4 = instances.get("i4")
    if i4 and i4[0] in ("on_chip_laser", "off_chip_laser"):
        instances["i4"] = [
            "off_chip_laser" if laser_source == "off_chip" else "on_chip_laser",
            0,
        ]
    i6 = instances.get("i6")
    if i6 and i6[0] in ("laser_splitter", "mmi"):
        instances["i6"] = ["mmi" if splitter_kind == "mmi" else "laser_splitter", 0]
    if splitter_kind == "mmi":
        ports = netlist.get("ports_num")
        if not isinstance(ports, dict):
            ports = {}
            netlist["ports_num"] = ports
        ports["i6"] = 10  # matches Customized_1x10_MMI's cfgs.ports_num

    core_cfg["height"] = int(core.get("height", core_cfg.get("height", 12)))
    core_cfg["width"] = int(core.get("width", core_cfg.get("width", 12)))
    core_cfg["num_wavelength"] = int(
        core.get("num_wavelength", core_cfg.get("num_wavelength", 12))
    )
    core_cfg["work_freq"] = float(core.get("work_freq", core_cfg.get("work_freq", 5)))
    core_cfg["precision"] = {
        "in_bit": int(precision.get("in_bit", 4)),
        "w_bit": int(precision.get("w_bit", 4)),
        "out_bit": int(precision.get("out_bit", 4)),
    }
    core_cfg["node"] = node_cfg

    # L4 topology / circuit overrides (sharing policy, temporal accumulation,
    # laser power distribution) map onto the core netlist as controlled values.
    apply_topology(
        core_cfg.setdefault("netlist", {}),
        str(topology.get("sharing_policy", DEFAULT_SHARING_POLICY)),
        str(topology.get("temporal_accum", DEFAULT_TEMPORAL_ACCUM)),
        str(topology.get("laser_distribution", DEFAULT_LASER_DISTRIBUTION)),
    )

    sub_arch = {
        "name": lt_body["name"],
        "version": "v1",
        "tiles": int(system.get("tiles", 4)),
        "cores_per_tile": int(system.get("cores_per_tile", 2)),
        "dataflow": str(system.get("dataflow", "output_stationary")),
        "op_type": str(system.get("op_type", "MVM")),
        "core": core_cfg,
        "memory": memory,
    }
    return sub_arch


# --------------------------------------------------------------------------- #
# Device counting (mirrors load_device_and_count).
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
    """Return (components, operand_1, operand_2, temporal_shared)."""
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
# Power initialisation (mirrors initalize_all_devices_power).
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
    # sar (default)
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
            device_cfg["dynamic_power"] = float(cfgs.get("P_pi", 0) or 0)
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
# Energy (mirrors reuse_determinator + energy_calculator).
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
    """Per-device energy plus a **native** per-layer energy breakdown.

    The accumulation loop already visits every layer, so the per-layer totals are
    captured here instead of being thrown away.  That is what turns the agents'
    per-layer energy ranking from a cycles-proportional *estimate* into a real
    simulated quantity.
    """
    dataflow = sub_arch.get("dataflow", "weight_stationary")
    freq = sub_arch["core"]["work_freq"]

    devices = initialize_device_power(sub_arch, components, insertion_loss)

    if not operand_1 or not operand_2:
        raise ValueError("No operand-encoding devices found in the architecture config")

    max_op1_cycles = max(devices[d]["tuning_cycles"] for d in operand_1)
    max_op2_cycles = max(devices[d]["tuning_cycles"] for d in operand_2)
    max_cycles = max(max_op1_cycles, max_op2_cycles)

    latency = {"total_tuning_latency": 0.0, "total_static_latency": 0.0}
    per_layer_energy: Dict[str, float] = {}
    operand_1_set = set(operand_1)
    operand_2_set = set(operand_2)

    for layer_name, cycle in cycles.items():
        iter_n, iter_d, iter_m, _, _, _, forward_w, forward_x = cycle
        switching_cycles = iter_n * iter_d * iter_m
        layer_energy = 0.0
        op2_duty, op1_duty, max_cycle = _reuse_determinator(
            dataflow, iter_m, iter_n, iter_d, max_op1_cycles, max_op2_cycles, max_cycles,
            forward_w, forward_x,
        )
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
# Area (mirrors initalize_all_devices_area + area_calculator + bounding box).
# --------------------------------------------------------------------------- #

def _topo_levels_sorted(
    graph: _DiGraph,
    dims_by_instance: Dict[str, Dict[str, float]],
) -> List[str]:
    """Topological order grouped by level, each level sorted by device length."""
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
    """Layout-aware node bounding box (mirrors default_bounding_box_estimator)."""
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
            # Mirror onnarchsim's cal_MMI_area: area scales with the instance's
            # port count (from netlist.ports_num, falling back to the device's
            # own ports_num).
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
# Insertion loss (mirrors architecture_insertion_loss).
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
    """Expand an insertion-loss path into per-component detail.

    Each element names the physical component, its library variant, and its
    individual insertion-loss contribution (dB).  This is the evidence the
    optical-level agent needs to localise loss along the light path rather than
    seeing only the aggregate path total.
    """
    detail: List[Dict[str, Any]] = []
    for node in path:
        instance_id = node.split("_")[0] if "_" in node else node
        choice = instances.get(instance_id)
        if choice is None:
            continue
        device_type, index = choice[0], int(choice[1])
        if device_type == "node":
            # The core netlist folds the whole node sub-circuit into a single
            # pseudo-component whose weight is the aggregate node loss.
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
# Synthetic workload + analytic memory.
# --------------------------------------------------------------------------- #

def synthetic_workload() -> Dict[str, Tuple[int, int, int, int, int, int, int, int]]:
    """The legacy fixed three-layer workload (kept for backward compatibility).

    These tuples have the shape ``(iter_N, iter_D, iter_M, N, D, M,
    forward_factor_w, forward_factor_x)`` produced by ``simu_partition_cycles``.

    Real models are now selectable through
    :func:`photon_tuner.core.workload.build_workload` (see
    ``configs/workloads.json``); this baseline is preserved unchanged so the
    original reference numbers still hold.
    """
    return synthetic_cycles()


def _synthetic_layer_sizes() -> Dict[str, Dict[str, Any]]:
    """Approximate layer I/O sizes (bytes) for the analytic memory model."""
    return synthetic_layer_sizes()


def memory_cost(
    sub_arch: Dict[str, Any],
    cycles: Dict[str, Tuple[int, ...]],
    layer_sizes: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Dict[str, float]], float, Dict[str, Any]]:
    """Analytic memory latency / energy (no CACTI binary required).

    Uses the HBM read/write energy and bandwidth plus the GLB/RF cache sizes
    from the architecture config; it mirrors the *shape* of SimPhony's memory
    hierarchy reports while staying fully offline.
    """
    memory = sub_arch.get("memory", {})
    hbm = memory.get("HBM", {})
    read_energy = float(hbm.get("read_energy", 62.4) or 0)  # pJ/Byte
    write_energy = float(hbm.get("write_energy", 62.4) or 0)  # pJ/Byte
    bandwidth = float(hbm.get("bandwidth", 1024) or 1)  # GB/s

    layer_sizes = layer_sizes if layer_sizes is not None else _synthetic_layer_sizes()
    latency: Dict[str, Dict[str, float]] = {}
    energy: Dict[str, float] = {}
    for layer, cycle in cycles.items():
        size = layer_sizes.get(layer, {})
        input_bytes = size.get("input_size", [0])[0]
        output_bytes = size.get("output_size", [0])[0]
        weight_bytes = size.get("weight_size", [0])[0]
        # Re-read weight every iter_N iterations (rough model).
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
# Top-level evaluation.
# --------------------------------------------------------------------------- #

def workload_arch_context(sub_arch: Dict[str, Any]) -> Dict[str, Any]:
    """Architecture inputs that the workload cycle arithmetic depends on.

    ``num_wavelength`` changes ``iter_D``; ``miniblock`` and ``dim_map`` change
    how many iterations each matrix dimension is split into.  Reading them from
    the evaluated sub-architecture is what makes a real workload's per-layer
    profile *respond to hardware edits* instead of staying constant.
    """
    core = sub_arch.get("core") or {}
    return {
        "num_wavelength": int(core.get("num_wavelength", 1) or 1),
        "dataflow": sub_arch.get("dataflow"),
        "op_type": sub_arch.get("op_type"),
        "miniblock": list(DEFAULT_MINIBLOCK),
    }


def evaluate(
    hardware: Dict[str, Any],
    simphony_root: str,
    device_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
    workload: Optional[Dict[str, Any]] = None,
    workload_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate one photonic design point and return the fine-grained report.

    Args:
        workload: A prebuilt workload dict (see
            :func:`photon_tuner.core.workload.build_workload`).  When omitted,
            ``workload_name`` selects one from ``configs/workloads.json``.
        workload_name: Registered workload name; defaults to the config's
            ``default`` entry (``synthetic``), preserving legacy behaviour.
    """
    hardware = materialize_hardware(hardware)
    catalog = device_catalog or load_device_catalog(f"{simphony_root}/configs/devices")
    sub_arch = build_sub_arch(hardware, simphony_root, catalog)

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
    # Mirror SimPhony's simu_area total: core-level devices plus the layout-aware
    # node bounding box (node-level device areas are already folded into the box).
    total_area = sum(v["total_area"] for key, v in area_devices.items() if "node" not in key)
    total_area += area_devices["node"]["total_area"]

    memory_latency, memory_energy, memory_spec = memory_cost(sub_arch, cycles, layer_sizes)
    total_memory_latency = sum(v["input_latency_s"] for v in memory_latency.values())
    total_memory_energy = sum(memory_energy.values())
    total_latency = (
        computation_latency["total_tuning_latency"]
        + computation_latency["total_static_latency"]
        + total_memory_latency
    )

    report = {
        "metadata": {"simulator": "SimPhony-analytic"},
        "hardware": hardware,
        "workload": {
            "name": resolved_workload.get("name"),
            "kind": resolved_workload.get("kind"),
            "partition_cycles": {"LT": cycles},
            "layer_sizes": layer_sizes,
        },
        "insertion_loss": {"LT": il},
        "energy": {
            "by_device": {"LT": energy_devices},
            "by_chip_type": {"LT": chip_energy},
            "by_layer": {"LT": per_layer_energy},
            "total": {"LT": total_energy},
        },
        "area": {
            "by_device": {"LT": area_devices},
            "by_chip_type": {"LT": chip_area},
            "total": {"LT": total_area},
        },
        "latency": {
            "computation": {"LT": computation_latency},
            "memory": {"LT": total_memory_latency},
            "total": {"LT": total_latency},
        },
        "memory": {
            "latency": {"LT": memory_latency},
            "energy": {"LT": memory_energy},
            "spec": memory_spec,
        },
        "metrics": {
            "energy_pj": total_energy,
            "area_um2": total_area,
            "latency_s": total_latency,
        },
    }
    return report
