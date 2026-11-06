"""光子 DSE 运行时的轻量 LLM 调用跟踪。

借助无依赖的 ``BaseCallbackHandler``，为每次 LLM 调用记录模型名称、prompt
数量、墙钟耗时和 token 用量，并将其追加到运行目录下的 JSONL 文件。无需
LangSmith API 密钥；若要切换到 LangSmith，设置 ``LANGCHAIN_TRACING_V2=1``
与 ``LANGCHAIN_API_KEY``，LangChain 便会将同样的事件发送到托管跟踪器。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class LLMTraceCollector(BaseCallbackHandler):
    """将 LLM 与链的开始/结束事件收集为可感知树结构的内存日志。

    每个事件都携带 ``run_id`` 和 ``parent_run_id``，因此扁平化的 JSONL 可以
    重新组装为调用层级（某次 LLM 调用属于哪个 stage/agent）——
    ``id / parent_id / children`` 结构——而非未嵌套的事件列表。
    """

    def __init__(self, stream_tokens: bool = False, include_chain_events: bool = True) -> None:
        self.records: List[Dict[str, Any]] = []
        self._starts: Dict[str, float] = {}
        self.stream_tokens = stream_tokens
        self.include_chain_events = include_chain_events

    @staticmethod
    def _name(serialized: Dict[str, Any], fallback: str) -> str:
        return (serialized or {}).get("name") or (serialized or {}).get("id") or fallback

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """将流式 token 转发到 stderr。"""
        if self.stream_tokens:
            sys.stderr.write(token)
            sys.stderr.flush()

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if not self.include_chain_events:
            return
        rid = str(run_id)
        self._starts[rid] = time.monotonic()
        self.records.append({
            "event": "chain_start",
            "run_id": rid,
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "name": self._name(serialized, "chain"),
            "timestamp": time.time(),
        })

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if not self.include_chain_events:
            return
        rid = str(run_id)
        started = self._starts.pop(rid, None)
        self.records.append({
            "event": "chain_end",
            "run_id": rid,
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "duration_s": round(time.monotonic() - started, 4) if started is not None else None,
        })

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        self._starts[rid] = time.monotonic()
        self.records.append({
            "event": "llm_start",
            "run_id": rid,
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "name": self._name(serialized, "llm"),
            "n_prompts": len(prompts) if isinstance(prompts, (list, tuple)) else 1,
            "timestamp": time.time(),
        })

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        started = self._starts.pop(rid, None)
        llm_output = getattr(response, "llm_output", None) or {}
        self.records.append({
            "event": "llm_end",
            "run_id": rid,
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "duration_s": round(time.monotonic() - started, 4) if started is not None else None,
            "token_usage": llm_output.get("token_usage") or {},
        })

    def build_tree(self) -> Dict[str, Any]:
        """将 ``records`` 组装为嵌套的 ``{node, children: [...]}`` 树。

        开始事件是规范节点；结束事件仅贡献计时信息，通过 ``run_id`` 匹配回
        对应的开始记录。
        """
        starts: Dict[str, Dict[str, Any]] = {}
        for record in self.records:
            if record["event"] in ("chain_start", "llm_start"):
                starts[record["run_id"]] = {
                    "run_id": record["run_id"],
                    "name": record.get("name") or record["event"].split("_")[0],
                    "kind": record["event"].split("_")[0],
                    "children": [],
                }
        for record in self.records:
            if record["event"] in ("chain_end", "llm_end"):
                node = starts.get(record["run_id"])
                if node is not None:
                    node["duration_s"] = record.get("duration_s")
                    if record.get("token_usage"):
                        node["token_usage"] = record["token_usage"]

        roots: List[Dict[str, Any]] = []
        nodes = list(starts.values())
        for node in nodes:
            parent_id = next(
                (r["parent_run_id"] for r in self.records
                 if r["run_id"] == node["run_id"] and "parent_run_id" in r),
                None,
            )
            if parent_id and parent_id in starts:
                starts[parent_id]["children"].append(node)
            else:
                roots.append(node)
        return {"roots": roots}

    def flush(self, path: Path) -> None:
        """将收集到的事件追加到 ``path``（JSONL）并清空缓冲区。"""
        if not self.records:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records.clear()
