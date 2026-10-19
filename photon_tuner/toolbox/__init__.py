"""Agent toolboxes (functions the agents call), distinct from the root
``tools/`` directory which holds CLI driver scripts.

Agent -> toolbox mapping
------------------------
Each agent only sees its ``allowed_tool_names`` allow-list (declared in
``photon_tuner.agents``).  Toolboxes are shared across agents, not owned by a
single one:

model_level
    ``analysis_tools``: hardware_metrics (all 5), layer_analysis
    (layer_ranks / operator_groups), search_state (both).
layer_level
    ``analysis_tools``: layer_analysis (all 3), hardware_metrics
    (hardware_config / metrics), search_state (both).
optical_level
    ``device_tools``: catalog (both), metrics (all 4).
solution
    ``analysis_tools``: hardware_metrics (all 5), layer_analysis (all 3),
    search_state (both), edits (both); ``device_tools``: edits
    (modify_device_parameter).

Shared primitives (``ToolContext`` and report helpers) live in
``photon_tuner.toolbox.common_tool``.
"""
