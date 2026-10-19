"""Photonic design search space and hardware-configuration helpers.

The "hardware" object manipulated by the agents is a plain ``dict`` with two
sections:

``arch``
    Architecture / optical-structure parameters (tiles, cores-per-tile,
    crossbar geometry, wavelength count, clock, precision, dataflow, memory).

``devices``
    Selected device *variant* per device type, split into ``core`` and ``node``
    levels (mirroring SimPhony's core/node config files).

Candidate sets for the ``devices`` section are discovered from SimPhony's
device YAMLs at runtime so the search space stays in sync with the device
library instead of being hard-coded twice.
"""

from __future__ import annotations

import copy
import glob
import os
from typing import Any, Dict, List, Mapping, Optional

import yaml

from photon_tuner.core.topology import (
    DEFAULT_LASER_DISTRIBUTION,
    DEFAULT_SHARING_POLICY,
    DEFAULT_TEMPORAL_ACCUM,
    LASER_DISTRIBUTIONS,
    SHARING_POLICIES,
    TEMPORAL_ACCUM,
)


# --------------------------------------------------------------------------- #
# Architecture-level candidate sets (defaults mirror SimPhony's LT configs).
# --------------------------------------------------------------------------- #

DEFAULT_ARCH_SEARCH_SPACE: Dict[str, Any] = {
    "tiles": [1, 2, 4, 8],
    "cores_per_tile": [1, 2, 4],
    "core_height": [8, 12, 16],
    "core_width": [8, 12, 16],
    "num_wavelength": [8, 12, 16],
    "work_freq": [5, 10, 15, 20, 25],
    "dataflow": ["output_stationary", "weight_stationary", "input_stationary"],
    "precision": {
        "in_bit": [2, 4, 8],
        "w_bit": [2, 4, 8],
        "out_bit": [2, 4, 8],
    },
    "hbm_bandwidth": [512, 1024, 2048],
    "glb1_cache_size": [64, 256, 485, 1024, 4096],
    "glb2_cache_size": [64, 256, 485, 1024, 4096],
    "rf_cache_size": [64, 256, 485, 1024, 4096],
    "sharing_policy": list(SHARING_POLICIES),
    "temporal_accum": list(TEMPORAL_ACCUM),
    "laser_distribution": list(LASER_DISTRIBUTIONS),
}


def _flat_arch_space(search_space: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return the flat ``{param: candidate list}`` arch space for a search-space
    spec, accepting both the layered (``{"layers": {...}}``) and legacy flat
    (``{"arch": {...}}``) formats.  ``precision_*`` params fold into the nested
    ``precision`` table used by the step/modify helpers."""
    spec = search_space or {}
    layers = spec.get("layers")
    if not layers:
        return dict(spec.get("arch") or DEFAULT_ARCH_SEARCH_SPACE)
    flat: Dict[str, Any] = {}
    for layer in layers.values():
        for param, p_spec in (layer.get("params") or {}).items():
            values = list((p_spec or {}).get("values", []))
            if not values:
                continue
            if param.startswith("precision_"):
                flat.setdefault("precision", {})[param.replace("precision_", "")] = values
            else:
                flat[param] = values
    return flat

# Tunable categories exposed to the solution-generation agent.
ARCH_PARAM_KEYS = [
    "tiles",
    "cores_per_tile",
    "core_height",
    "core_width",
    "num_wavelength",
    "work_freq",
    "dataflow",
    "precision_in_bit",
    "precision_w_bit",
    "precision_out_bit",
    "hbm_bandwidth",
    "glb1_cache_size",
    "glb2_cache_size",
    "rf_cache_size",
    "sharing_policy",
    "temporal_accum",
    "laser_distribution",
]

# Memory hierarchy param name -> (memory level, field in that level's dict).
_MEMORY_PARAM_MAP = {
    "hbm_bandwidth": ("HBM", "bandwidth"),
    "glb1_cache_size": ("GLB1", "cache_size"),
    "glb2_cache_size": ("GLB2", "cache_size"),
    "rf_cache_size": ("RF", "cache_size"),
}

# Baseline device choices mirroring configs/design/cores/dptc.yml and
# configs/design/nodes/dode.yml in the copied SimPhony project.
DEFAULT_CORE_DEVICES = {
    "mzm": "LT_MZM",
    "dac": "DAC_2",
    "adc": "ADC_SAR_1",
    "tia": "TIA_1",
    "waveguide": "WG_Single_Strip",
    "mrr_rerouter": "LT_MRR_REROUTER",
    "laser_splitter": "Customized_Laser_Splitter",
    "micro_comb": "LT_Micro_Comb",
    "on_chip_laser": "Customized_Laser",
}

DEFAULT_NODE_DEVICES = {
    "phase_shifter": "MEMS_PS",
    "photodetector": "LT_PD",
    "coupler": "LT_DC_2x2",
    "y_branch": "LT_Y_Branch",
}


def default_hardware() -> Dict[str, Any]:
    """Return the baseline photonic configuration (SimPhony LT design).

    The hardware dict mirrors the layered design space with four clean
    sections: ``arch.system`` (tiling / dataflow), ``arch.optical_core``
    (crossbar geometry / precision / clock), ``arch.topology`` (sharing /
    temporal accumulation / laser distribution) and ``arch.memory``, plus a
    top-level ``devices`` table for library-variant selection.
    """
    return {
        "arch": {
            "system": {
                "tiles": 4,
                "cores_per_tile": 2,
                "dataflow": "output_stationary",
                "op_type": "MVM",
            },
            "optical_core": {
                "type": "Crossbar",
                "coherent": True,
                "universal": True,
                "height": 12,
                "width": 12,
                "num_wavelength": 12,
                "work_freq": 5,
                "precision": {"in_bit": 4, "w_bit": 4, "out_bit": 4},
            },
            "topology": {
                "sharing_policy": DEFAULT_SHARING_POLICY,
                "temporal_accum": DEFAULT_TEMPORAL_ACCUM,
                "laser_distribution": DEFAULT_LASER_DISTRIBUTION,
            },
            "memory": {
                "HBM": {"read_energy": 62.4, "write_energy": 62.4, "bandwidth": 1024},
                "GLB2": {"cache_size": 485, "block_size": 8},
                "GLB1": {"cache_size": 485, "block_size": 8},
                "RF": {"cache_size": 485, "block_size": 8},
            },
        },
        "devices": {
            "core": {
                **dict(DEFAULT_CORE_DEVICES),
                # Instance-level device-type switches (defaults keep the LT
                # baseline: on-chip laser + waveguide laser-splitter).
                "laser_source": "on_chip",
                "splitter_kind": "laser_splitter",
            },
            "node": dict(DEFAULT_NODE_DEVICES),
        },
    }


# --------------------------------------------------------------------------- #
# Device-library discovery (reads SimPhony's device YAMLs).
# --------------------------------------------------------------------------- #


def discover_device_catalog(devicelib_root: str) -> Dict[str, Dict[str, Any]]:
    """Load every device YAML under ``devicelib_root`` into a nested dict.

    Returns ``{device_type: {device_name: {"cfgs": {...}, "type": ...}}}``.
    """
    catalog: Dict[str, Dict[str, Any]] = {}
    files = glob.glob(os.path.join(devicelib_root, "*", "*.yml"))
    for path in sorted(files):
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        for device_type, variants in data.items():
            bucket = catalog.setdefault(str(device_type), {})
            for name, body in variants.items():
                if isinstance(body, dict):
                    bucket[str(name)] = body
    return catalog


def discover_device_variants(devicelib_root: str) -> Dict[str, List[str]]:
    """Return ``{device_type: [variant names]}`` from the device library."""
    catalog = discover_device_catalog(devicelib_root)
    return {
        device_type: sorted(variants.keys())
        for device_type, variants in catalog.items()
    }


# --------------------------------------------------------------------------- #
# Hardware normalisation / stepping helpers.
# --------------------------------------------------------------------------- #


def materialize_hardware(hardware: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return a deep copy with architecture defaults filled in."""
    base = default_hardware()
    if hardware:
        _deep_update(base, dict(hardware))
    return base


def normalize_hardware_to_design_space(
    hardware: Mapping[str, Any],
    search_space: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Clamp scalar architecture values to their nearest legal candidates."""
    arch_space = _flat_arch_space(search_space)
    result = materialize_hardware(hardware)
    arch = result["arch"]
    system = arch["system"]
    oc = arch["optical_core"]
    topo = arch["topology"]

    def _clamp(key: str, candidates: List[Any], target: Dict[str, Any]) -> None:
        if key in target and candidates:
            try:
                current = float(target[key])
            except (TypeError, ValueError):
                return
            target[key] = min(candidates, key=lambda c: abs(float(c) - current))

    _clamp("tiles", arch_space.get("tiles", []), system)
    _clamp("cores_per_tile", arch_space.get("cores_per_tile", []), system)
    _clamp("height", arch_space.get("core_height", []), oc)
    _clamp("width", arch_space.get("core_width", []), oc)
    _clamp("num_wavelength", arch_space.get("num_wavelength", []), oc)
    _clamp("work_freq", arch_space.get("work_freq", []), oc)

    precision_space = arch_space.get("precision", {})
    precision = oc.setdefault("precision", {})
    for bit_key in ("in_bit", "w_bit", "out_bit"):
        candidates = precision_space.get(bit_key, [])
        if bit_key in precision and candidates:
            try:
                current = float(precision[bit_key])
            except (TypeError, ValueError):
                continue
            precision[bit_key] = min(candidates, key=lambda c: abs(float(c) - current))

    memory = arch.setdefault("memory", {})
    for param, (level, field) in _MEMORY_PARAM_MAP.items():
        candidates = arch_space.get(param, [])
        if not candidates:
            continue
        target = memory.setdefault(level, {})
        if field in target:
            try:
                current = float(target[field])
            except (TypeError, ValueError):
                continue
            target[field] = min(candidates, key=lambda c: abs(float(c) - current))

    dataflow = str(system.get("dataflow", ""))
    dataflow_candidates = arch_space.get("dataflow", [])
    if dataflow_candidates and dataflow not in dataflow_candidates:
        system["dataflow"] = dataflow_candidates[0]

    for param in ("sharing_policy", "temporal_accum", "laser_distribution"):
        candidates = arch_space.get(param, [])
        if not candidates:
            continue
        current = str(topo.get(param, ""))
        if current not in [str(c) for c in candidates]:
            topo[param] = candidates[0]

    return result


def step_arch_parameter(
    hardware: Mapping[str, Any],
    parameter: str,
    direction: int,
    search_space: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a new hardware dict with one arch parameter stepped +1 / -1."""
    arch_space = _flat_arch_space(search_space)
    result = materialize_hardware(hardware)
    arch = result["arch"]
    system = arch["system"]
    oc = arch["optical_core"]
    topo = arch["topology"]

    mapping = {
        "tiles": ("tiles", system, arch_space.get("tiles", [])),
        "cores_per_tile": ("cores_per_tile", system, arch_space.get("cores_per_tile", [])),
        "core_height": ("height", oc, arch_space.get("core_height", [])),
        "core_width": ("width", oc, arch_space.get("core_width", [])),
        "num_wavelength": ("num_wavelength", oc, arch_space.get("num_wavelength", [])),
        "work_freq": ("work_freq", oc, arch_space.get("work_freq", [])),
    }
    if parameter in mapping:
        key, target, candidates = mapping[parameter]
        if not candidates:
            return result
        target[key] = _step_value(target.get(key), candidates, direction)
        return result

    precision_map = {
        "precision_in_bit": "in_bit",
        "precision_w_bit": "w_bit",
        "precision_out_bit": "out_bit",
    }
    if parameter in precision_map:
        bit_key = precision_map[parameter]
        candidates = (arch_space.get("precision") or {}).get(bit_key, [])
        precision = oc.setdefault("precision", {})
        if candidates:
            precision[bit_key] = _step_value(precision.get(bit_key), candidates, direction)
        return result

    if parameter in _MEMORY_PARAM_MAP:
        level, field = _MEMORY_PARAM_MAP[parameter]
        candidates = arch_space.get(parameter, [])
        memory = arch.setdefault("memory", {})
        target = memory.setdefault(level, {})
        if candidates:
            target[field] = _step_value(target.get(field), candidates, direction)
        return result

    if parameter == "dataflow":
        candidates = list(arch_space.get("dataflow", []))
        if candidates:
            current = str(system.get("dataflow", ""))
            if current not in [str(c) for c in candidates]:
                system["dataflow"] = candidates[0]
            else:
                index = [str(c) for c in candidates].index(current)
                index = max(0, min(len(candidates) - 1, index + direction))
                system["dataflow"] = candidates[index]
        return result

    if parameter in {"sharing_policy", "temporal_accum", "laser_distribution"}:
        candidates = list(arch_space.get(parameter, []))
        if candidates:
            current = str(topo.get(parameter, ""))
            if current not in [str(c) for c in candidates]:
                topo[parameter] = candidates[0]
            else:
                index = [str(c) for c in candidates].index(current)
                index = max(0, min(len(candidates) - 1, index + direction))
                topo[parameter] = candidates[index]
        return result

    raise ValueError(f"Unknown architecture parameter: {parameter}")


def modify_arch_parameter(
    hardware: Mapping[str, Any],
    updates: List[Dict[str, Any]],
    search_space: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Apply explicit architecture-parameter updates and clamp to candidates."""
    result = materialize_hardware(hardware)
    arch = result["arch"]
    system = arch["system"]
    oc = arch["optical_core"]
    topo = arch["topology"]
    precision = oc.setdefault("precision", {})

    for update in updates:
        parameter = str(update.get("parameter", ""))
        value = update.get("value")
        if parameter == "dataflow":
            system["dataflow"] = str(value)
        elif parameter == "tiles":
            system["tiles"] = _as_int(value)
        elif parameter == "cores_per_tile":
            system["cores_per_tile"] = _as_int(value)
        elif parameter == "core_height":
            oc["height"] = _as_int(value)
        elif parameter == "core_width":
            oc["width"] = _as_int(value)
        elif parameter == "num_wavelength":
            oc["num_wavelength"] = _as_int(value)
        elif parameter == "work_freq":
            oc["work_freq"] = _as_int(value)
        elif parameter in {"precision_in_bit", "precision_w_bit", "precision_out_bit"}:
            precision[parameter.replace("precision_", "")] = _as_int(value)
        elif parameter in _MEMORY_PARAM_MAP:
            level, field = _MEMORY_PARAM_MAP[parameter]
            memory = arch.setdefault("memory", {})
            memory.setdefault(level, {})[field] = _as_int(value)
        elif parameter in {"sharing_policy", "temporal_accum", "laser_distribution"}:
            topo[parameter] = str(value)
        else:
            raise ValueError(f"Unknown architecture parameter: {parameter}")

    return normalize_hardware_to_design_space(result, search_space)


def set_device_choice(
    hardware: Mapping[str, Any],
    level: str,
    device_type: str,
    device_name: str,
) -> Dict[str, Any]:
    """Set a device variant for a device type at the ``core`` or ``node`` level."""
    if level not in {"core", "node"}:
        raise ValueError(f"level must be 'core' or 'node', got {level!r}")
    result = materialize_hardware(hardware)
    result["devices"][level][device_type] = device_name
    return result


def hardware_summary(hardware: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact, LLM-friendly summary of a photonic hardware configuration."""
    hw = materialize_hardware(hardware)
    arch = hw["arch"]
    system = arch["system"]
    oc = arch["optical_core"]
    topo = arch["topology"]
    return {
        "arch": {
            "tiles": system.get("tiles"),
            "cores_per_tile": system.get("cores_per_tile"),
            "core_height": oc.get("height"),
            "core_width": oc.get("width"),
            "num_wavelength": oc.get("num_wavelength"),
            "work_freq": oc.get("work_freq"),
            "precision": oc.get("precision"),
            "sharing_policy": topo.get("sharing_policy"),
            "temporal_accum": topo.get("temporal_accum"),
            "laser_distribution": topo.get("laser_distribution"),
            "dataflow": system.get("dataflow"),
            "memory": {level: dict(cfg) for level, cfg in arch.get("memory", {}).items()},
        },
        "core_devices": dict(hw.get("devices", {}).get("core", {})),
        "node_devices": dict(hw.get("devices", {}).get("node", {})),
    }


# --------------------------------------------------------------------------- #
# Internal helpers.
# --------------------------------------------------------------------------- #


def _step_value(current: Any, candidates: List[Any], direction: int) -> Any:
    ordered = sorted(set(candidates), key=lambda v: float(v))
    try:
        current_value = float(current)
    except (TypeError, ValueError):
        return ordered[0]
    if current_value not in [float(v) for v in ordered]:
        return min(ordered, key=lambda v: abs(float(v) - current_value))
    index = [float(v) for v in ordered].index(current_value)
    index = max(0, min(len(ordered) - 1, index + direction))
    return ordered[index]


def _as_int(value: Any) -> int:
    return int(round(float(value)))


def _deep_update(target: Dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
