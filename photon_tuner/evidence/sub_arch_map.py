"""Search parameter -> sub-architecture location mapping.

``build_sub_arch`` (in :mod:`photon_tuner.simulators.analytic`) is the bridge
that writes searched hardware values into the ``sub_arch`` dict that
``ONNArchSimulator`` actually consumes.  That mapping is currently *implicit*
(hand-written field assignments).  This module makes it explicit so the
verification harness can read the injected value back out of the simulator and
confirm the parameter truly landed.

Scalar parameters map to a plain key path; the three topology parameters are
not scalars -- they reshape ``core.netlist`` via :mod:`photon_tuner.core.topology`.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from photon_tuner.core.topology import (
    _SHARING_OVERRIDES,
    _TEMPORAL_DURATIONS,
)

# Scalar search parameters: param -> sub_arch key path.
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

# Topology parameters (netlist-shaped, not scalar).
TOPOLOGY_PARAMS: Tuple[str, ...] = (
    "sharing_policy",
    "temporal_accum",
    "laser_distribution",
)

# Recording map: param -> sub_arch path(s) for the evidence record.  Scalar
# params map to a plain path; topology params map to the netlist field(s) they
# reshape.
SUB_ARCH_MAP: Dict[str, Any] = {
    **{p: list(path) for p, path in SUB_ARCH_SCALAR_MAP.items()},
    "laser_distribution": ["core", "netlist", "laser_power_distribution"],
    "temporal_accum": ["core", "netlist", "temporal_accum_factor", "duration"],
    "sharing_policy": ["core", "netlist", "scaling_rules", ["i2", "i8"]],
    # Instance-level device-type switches: the switched value is the device
    # type of the corresponding netlist instance (i4 laser / i6 splitter).
    "laser_source": ["core", "netlist", "instances", "i4"],
    "splitter_kind": ["core", "netlist", "instances", "i6"],
}

# Device choices land in ``core.devices`` / ``core.node.devices`` as a
# one-element list ``[variant_name]``.
DEVICE_READBACK_PATHS: Dict[str, Tuple[str, ...]] = {
    "core": ("core", "devices"),
    "node": ("core", "node", "devices"),
}


def _lookup(node: Any, path: Tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def read_sub_arch_param(sub_arch: Dict[str, Any], param: str) -> Any:
    """Return the value(s) ``build_sub_arch`` placed for ``param`` (or None)."""
    if param in SUB_ARCH_SCALAR_MAP:
        return _lookup(sub_arch, SUB_ARCH_SCALAR_MAP[param])
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
    """The value(s) we expect to find for a parameter's candidate value."""
    if param in SUB_ARCH_SCALAR_MAP:
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
    """Return ``(confirmed, readback_summary)`` for one injected parameter."""
    actual = read_sub_arch_param(sub_arch, param)
    expected = expected_sub_arch_value(param, value)
    if isinstance(expected, dict):
        confirmed = isinstance(actual, dict) and all(
            _eq(actual.get(k), v) for k, v in expected.items()
        )
    else:
        confirmed = _eq(actual, expected)
    return confirmed, {
        "param": param,
        "expected": expected,
        "actual": actual,
    }


def read_device_choice(sub_arch: Dict[str, Any], level: str, device_type: str) -> Any:
    """Return the device-variant list injected for ``devices.<level>.<type>``."""
    path = DEVICE_READBACK_PATHS.get(level)
    if path is None:
        return None
    return _lookup(sub_arch, path + (device_type,))


def confirm_device_injection(
    sub_arch: Dict[str, Any], level: str, device_type: str, device_name: Any
) -> Tuple[bool, Dict[str, Any]]:
    """Return ``(confirmed, readback_summary)`` for one injected device choice."""
    actual = read_device_choice(sub_arch, level, device_type)
    expected = [str(device_name)]
    confirmed = isinstance(actual, (list, tuple)) and bool(actual) and actual[0] == expected[0]
    return confirmed, {
        "param": f"devices.{level}.{device_type}",
        "expected": expected,
        "actual": actual,
    }
