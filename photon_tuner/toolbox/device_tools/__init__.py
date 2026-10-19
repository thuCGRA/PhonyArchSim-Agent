"""Device / circuit-level toolbox for the photonic DSE framework.

This toolbox backs the *optical-level* agent that diagnoses the optical link
below the layer level: photonic device variants, their configurable ``cfgs``
parameters, and the optical structure (crossbar geometry, wavelength count,
precision, dataflow).  It is deliberately separate from
:class:`~photon_tuner.toolbox.analysis_tools.AnalysisToolbox` so each agent's
tool surface is explicit.

Tools are grouped one-file-per-category (catalog / metrics / edits);
:class:`DeviceToolbox` assembles them through a ``name -> (spec, handler)``
registry.

Used by: optical_level, solution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from photon_tuner.common.tool_types import ToolResult, ToolSpec
from photon_tuner.core.search_space import (
    discover_device_catalog,
    discover_device_variants,
)
from photon_tuner.toolbox.common_tool.context import ToolContext

from . import catalog, edits, metrics

_REGISTRY: Dict[str, Tuple[ToolSpec, Any]] = {}
for _mod in (catalog, metrics, edits):
    _REGISTRY.update(_mod.TOOLS)


class DeviceToolbox:
    """Device-level read-and-edit toolbox."""

    # Instance-level device-type switches (not library types).  Choosing one
    # rewrites the corresponding netlist instance in ``build_sub_arch``.
    INSTANCE_CHOICES: Dict[str, List[str]] = {
        "laser_source": ["on_chip", "off_chip"],
        "splitter_kind": ["laser_splitter", "mmi"],
    }

    def __init__(self, devicelib_root: str, search_space: Optional[Dict[str, Any]] = None) -> None:
        self.devicelib_root = str(devicelib_root)
        self.search_space = search_space or {}
        self.catalog = discover_device_catalog(self.devicelib_root)
        self.variants = discover_device_variants(self.devicelib_root)

    def specs(self) -> List[ToolSpec]:
        return [spec for spec, _ in _REGISTRY.values()]

    def run(
        self,
        name: str,
        context: ToolContext,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        arguments = dict(arguments or {})
        entry = _REGISTRY.get(name)
        if entry is None:
            raise ValueError(f"Unknown device tool: {name}")
        return entry[1](self, context, arguments)
