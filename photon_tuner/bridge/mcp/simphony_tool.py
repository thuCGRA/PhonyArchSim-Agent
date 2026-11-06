"""SimPhony 服务器的 MCP 客户端（LangGraph 侧，Python >=3.11）。

将 Python-3.10 的 ``simphony_server`` 通过 stdio 封装为一个普通同步对象，其
``evaluate`` 签名与进程内的 :meth:`SimphonyAdapter.evaluate` 一致。服务器子进程
在客户端生命周期内保持存活，使 DSE 循环只需支付一次 torch/ONN 导入开销。

典型用法::

    client = SimphonyMCPClient.default()
    data = client.evaluate(hardware, iteration=0)   # -> {"metrics":..., "hardware":...}
    client.close()
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient


def _unwrap_text(out: Any) -> str:
    """把 langchain-mcp-adapters 工具结果还原为文本载荷。

    ``MultiServerMCPClient`` 工具返回 LangChain 内容块，单个文本结果通常是
    ``[{"type": "text", "text": "..."}]``。
    """
    if hasattr(out, "content"):
        out = out.content
    if isinstance(out, list):
        parts: List[str] = []
        for block in out:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "text" in block:
                    parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    if isinstance(out, dict):
        if "text" in out:
            return str(out["text"])
        return json.dumps(out)
    if isinstance(out, str):
        return out
    return str(out)


class SimphonyMCPClient:
    """基于持久 stdio MCP 连接的同步客户端。"""

    def __init__(
        self,
        server_command: str,
        server_args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 3600.0,
    ) -> None:
        self._timeout = timeout
        self._config = {
            "simphony": {
                "command": server_command,
                "args": list(server_args or []),
                "env": dict(env or {}),
                "transport": "stdio",
            }
        }
        self._client = MultiServerMCPClient(self._config)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._evaluate_tool: Any = None

    # -- 构造辅助方法 -------------------------------------------------------- #

    @classmethod
    def default(
        cls,
        project_root: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 3600.0,
    ) -> "SimphonyMCPClient":
        root = (project_root or Path(__file__).resolve().parents[3]).resolve()
        server = root / "photon_tuner" / "bridge" / "mcp" / "simphony_server.py"
        python = root / ".venv-simulator" / "bin" / "python"
        full_env = {
            "MPLCONFIGDIR": str(root / ".tools" / "mplconfig"),
            "PYTHONUNBUFFERED": "1",
        }
        full_env.update(env or {})
        return cls(str(python), [str(server)], env=full_env, timeout=timeout)

    # -- 客户端接口 ---------------------------------------------------------- #

    def _tool(self) -> Any:
        if self._evaluate_tool is None:
            future = asyncio.run_coroutine_threadsafe(self._client.get_tools(), self._loop)
            tools = future.result(timeout=120)
            by_name = {t.name: t for t in tools}
            self._evaluate_tool = by_name.get("evaluate") or by_name.get("simphony_evaluate")
            if self._evaluate_tool is None:
                raise RuntimeError(f"MCP evaluate tool not found; available: {sorted(by_name)}")
        return self._evaluate_tool

    def evaluate(self, hardware: Dict[str, Any], iteration: int = 0) -> Dict[str, Any]:
        tool = self._tool()
        args = {"hardware": json.dumps(hardware), "iteration": int(iteration)}
        future = asyncio.run_coroutine_threadsafe(tool.ainvoke(args), self._loop)
        out = future.result(timeout=self._timeout)
        return json.loads(_unwrap_text(out))

    def ping(self) -> str:
        """往返调用服务器的 ``ping`` 工具（真正的存活检查）。"""
        future = asyncio.run_coroutine_threadsafe(self._client.get_tools(), self._loop)
        tools = future.result(timeout=120)
        ping_tool = next((t for t in tools if t.name == "ping"), None)
        if ping_tool is None:
            # 服务器未提供显式 ping 工具时，get_tools() 已证明会话可用。
            return "pong"
        out = asyncio.run_coroutine_threadsafe(ping_tool.ainvoke({}), self._loop).result(timeout=30)
        return str(_unwrap_text(out))

    def close(self) -> None:
        try:
            future = asyncio.run_coroutine_threadsafe(self._client.close(), self._loop)
            future.result(timeout=30)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
