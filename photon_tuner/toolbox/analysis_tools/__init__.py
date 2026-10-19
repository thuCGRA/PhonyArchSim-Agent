"""Architecture / layer-level toolbox for the photonic DSE agents.

The toolbox is the *only* way an agent reads simulator evidence or edits the
design.  Each agent receives an explicit ``allowed_tool_names`` allow-list (see
``photon_tuner.agents``), so the model-level, layer-level, and
solution-generation roles only see the tools relevant to them.

Tools are grouped one-file-per-category (hardware metrics / layer analysis /
search state / edits); :class:`AnalysisToolbox` assembles them through a
``name -> (spec, handler)`` registry.

Used by: model_level, layer_level, solution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from photon_tuner.common.tool_types import ToolResult, ToolSpec
from photon_tuner.toolbox.common_tool.context import ToolContext

from . import edits, hardware_metrics, layer_analysis, search_state

_REGISTRY: Dict[str, Tuple[ToolSpec, Any]] = {}
for _mod in (hardware_metrics, layer_analysis, search_state, edits):
    _REGISTRY.update(_mod.TOOLS)


class AnalysisToolbox:
    """Architecture / layer-level read-and-edit toolbox."""

    def __init__(self, search_space: Optional[Dict[str, Any]] = None) -> None:
        self.search_space = search_space or {}

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
            raise ValueError(f"Unknown analysis tool: {name}")
        return entry[1](self, context, arguments)
