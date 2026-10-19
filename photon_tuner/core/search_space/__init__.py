"""Photonic design search space: layered model + hardware ops.

This package groups the two sides of the search space that used to live in
separate top-level modules:

- :mod:`.design_space` -- the layered ``DesignSpace`` model (real-mode
  oriented), with refine / violations / normalize.
- :mod:`.hardware_ops` -- the plain-``dict`` hardware helpers the agents call
  (materialize / step / modify / summary / device discovery) plus the legacy
  flat ``DEFAULT_ARCH_SEARCH_SPACE`` it falls back to.

The public names of both are re-exported here so existing imports such as
``from photon_tuner.core.search_space import materialize_hardware`` keep
working unchanged.
"""

from __future__ import annotations

from .design_space import LAYERED_DESIGN_SPACE, DesignSpace, build_design_space
from .hardware_ops import (
    ARCH_PARAM_KEYS,
    DEFAULT_ARCH_SEARCH_SPACE,
    DEFAULT_CORE_DEVICES,
    DEFAULT_NODE_DEVICES,
    default_hardware,
    discover_device_catalog,
    discover_device_variants,
    hardware_summary,
    materialize_hardware,
    modify_arch_parameter,
    normalize_hardware_to_design_space,
    set_device_choice,
    step_arch_parameter,
)

__all__ = [
    "ARCH_PARAM_KEYS",
    "DEFAULT_ARCH_SEARCH_SPACE",
    "DEFAULT_CORE_DEVICES",
    "DEFAULT_NODE_DEVICES",
    "LAYERED_DESIGN_SPACE",
    "DesignSpace",
    "build_design_space",
    "default_hardware",
    "discover_device_catalog",
    "discover_device_variants",
    "hardware_summary",
    "materialize_hardware",
    "modify_arch_parameter",
    "normalize_hardware_to_design_space",
    "set_device_choice",
    "step_arch_parameter",
]
