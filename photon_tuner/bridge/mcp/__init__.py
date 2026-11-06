"""连接 LangGraph 运行时与真实 SimPhony 模拟器的 MCP 桥接层。

- ``simphony_server.py`` 运行在 Python-3.10 虚拟环境下，暴露一个由
  ``SimphonyAdapter(mode="real")`` 支撑的 ``evaluate(hardware)`` 工具。
- ``simphony_tool.py`` 运行在 LangGraph 虚拟环境下，将该服务器封装为一个
  普通同步客户端，对外提供相同的 ``evaluate`` 签名。
"""
