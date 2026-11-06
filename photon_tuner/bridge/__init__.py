"""LangGraph 运行时的进程外桥接层。

真实的 SimPhony 模拟器依赖 torch 1.12.1 / Python <=3.10，而 LangGraph
运行时需要 Python >=3.11，两者无法共用同一个解释器。本包承载 MCP 桥接，
使二者处于同一编排进程内、却运行在不同的解释器中。
"""
