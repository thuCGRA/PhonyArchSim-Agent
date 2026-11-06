"""LangGraph 检查点器工厂。

优先使用持久化的 ``SqliteSaver``（``langgraph-checkpoint-sqlite``，需已安装），
以便各阶段的 agent 状态在进程重启后仍然保留；否则回退到始终可用的内存
保存器，使运行时在零额外依赖的离线环境下也能工作。检查点是尽力而为的：
保存器构造失败绝不应中止一次调优运行。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_checkpointer(db_path: Any) -> Any:
    """返回检查点器，优先使用持久化 SQLite，其次为内存实现。"""
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        return SqliteSaver(conn)
    except Exception:  # noqa: BLE001 — 包缺失或连接异常时降级
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
