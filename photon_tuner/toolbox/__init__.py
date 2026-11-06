"""Agent 工具箱：agent 调用的、领域安全的处理器。

本包（``analysis_tools`` / ``device_tools`` / ``common_tool`` /
``tool_cache``）是纯领域逻辑，必须能在不依赖 langchain 的情况下被 import，
以便仅领域测试套件可直接在纯 ``python3`` 上运行。把这些处理器包装成
``BaseTool`` 的 LangChain 绑定层位于同级子包
:mod:`photon_tuner.toolbox.bindings`（此处有意不导入）。

Agent -> 工具箱映射
--------------------
每个 agent 只能看到自己的 ``allowed_tool_names`` 白名单（在
``photon_tuner.agents`` 中声明）。工具箱在 agent 之间共享，而非由某个
agent 独占：

model_level
    ``analysis_tools``：hardware_metrics（全部 5 个）、layer_analysis
    （layer_ranks / operator_groups）、search_state（两个）。
layer_level
    ``analysis_tools``：layer_analysis（全部 3 个）、hardware_metrics
    （hardware_config / metrics）、search_state（两个）。
optical_level
    ``device_tools``：catalog（两个）、metrics（全部 4 个）。
solution
    ``analysis_tools``：hardware_metrics（全部 5 个）、layer_analysis
    （全部 3 个）、search_state（两个）、edits（两个）；``device_tools``：
    edits（modify_device_parameter）。

共享原语（``ToolContext`` 和报告辅助函数）位于
``photon_tuner.toolbox.common_tool``。
"""
