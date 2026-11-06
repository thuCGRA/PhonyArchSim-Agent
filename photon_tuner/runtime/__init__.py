"""基于 LangGraph 的光子 DSE 框架运行时。

编排逻辑位于 :mod:`photon_tuner.runtime.orchestrator`（``DeepAgentTuner``），
它在 SimPhony 适配器外围按固定顺序
（model -> architecture -> topology -> device -> solution）运行五个深度角色智能体，
并保持 keep-best / 禁用手纹 / 打包运行目录的可复现性。
"""
