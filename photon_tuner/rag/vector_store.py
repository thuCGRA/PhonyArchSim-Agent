"""基于 SQLite 的历史案例存储（RAG 记忆），用于光子 DSE。

每个提交的案例记录其瓶颈描述、完整硬件配置、提出的解决方案、前后指标，以及
一个两部分的检索向量：一个通道编码瓶颈语义，另一个通道编码硬件状态，使检索
同时以二者为条件。
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.vectorstores import VectorStore
except ImportError:  # pragma: no cover - 域测试在无 langchain 环境下运行
    CallbackManagerForRetrieverRun = None  # type: ignore[assignment]
    Document = None  # type: ignore[assignment]
    BaseRetriever = None  # type: ignore[assignment]
    VectorStore = None  # type: ignore[assignment]

from photon_tuner.core.search_space import hardware_summary
from photon_tuner.rag.embeddings import HashingEmbeddingModel, Vector


BOTTLENECK_VECTOR_WEIGHT = 0.75
HARDWARE_VECTOR_WEIGHT = 0.25


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hardware_fingerprint(hardware: Dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(hardware).encode("utf-8")).digest()
    return digest[:8].hex()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = math.fsum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(math.fsum(x * x for x in a))
    norm_b = math.sqrt(math.fsum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def summarize_hardware(hardware: Dict[str, Any]) -> Dict[str, Any]:
    """紧凑、适合 LLM 的光子硬件摘要（用于检索文本）。"""
    return hardware_summary(hardware)


@dataclass
class HistoryRecord:
    record_id: str
    vector: Vector
    bottleneck_description: str
    hardware: Dict[str, Any]
    solution: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    bottleneck_state: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class RetrievedCase:
    score: float
    record: HistoryRecord


def build_case_vector(embedder: Any, bottleneck_text: str, hardware: Dict[str, Any]) -> Vector:
    """双通道向量：瓶颈语义 + 硬件状态。"""
    v_bottleneck = embedder.embed(bottleneck_text)
    v_hardware = embedder.embed(canonical_json(summarize_hardware(hardware)))
    dimension = len(v_bottleneck)
    return [
        BOTTLENECK_VECTOR_WEIGHT * v_bottleneck[i] + HARDWARE_VECTOR_WEIGHT * v_hardware[i]
        for i in range(dimension)
    ]


def summarize_cases(cases: List[RetrievedCase], limit: int = 5) -> List[Dict[str, Any]]:
    summaries = []
    for case in cases[:limit]:
        record = case.record
        summaries.append({
            "score": round(case.score, 4),
            "hardware": summarize_hardware(record.hardware),
            "metrics": record.metrics,
            "solution": record.solution,
        })
    return summaries


def summarize_documents(docs: List[Any], limit: int = 5) -> List[Dict[str, Any]]:
    """从 Retriever 的 ``Document`` 列表重建面向 LLM 的摘要。

    ``HistoryRetriever`` 将完整的 :class:`HistoryRecord` 编码进每个
    ``Document.metadata``，使框架标准的 ``retriever.invoke(query)`` 路径保持
    无损；本函数是该编码的逆操作。
    """
    summaries = []
    for doc in docs[:limit]:
        meta = (getattr(doc, "metadata", None) or {})
        if "record_id" not in meta:
            continue
        summaries.append({
            "score": round(float(meta.get("score", 0.0)), 4),
            "hardware": summarize_hardware(meta.get("hardware", {})),
            "metrics": meta.get("metrics", {}),
            "solution": meta.get("solution", {}),
        })
    return summaries


class _EmbeddingAdapter:
    """将 LangChain ``Embeddings`` 对象封装为存储的 ``embed(text)`` API。"""

    def __init__(self, embeddings: Any) -> None:
        self.embeddings = embeddings

    def embed(self, text: str) -> Vector:
        return list(self.embeddings.embed_query(text))


class HistoryVectorStore(VectorStore if VectorStore is not None else object):
    """SQLite 向量存储，默认使用轻量哈希嵌入器。

    在同一 SQLite 后端之上实现 LangChain
    :class:`~langchain_core.vectorstores.VectorStore` 接口
    （``add_texts`` / ``similarity_search`` / ``from_texts`` / ``delete``），
    同时保留领域专用的 ``add`` / ``search``（双通道瓶颈+硬件检索）。
    当 ``langchain_core`` 不可用（域测试场景）时，退化为一个具有相同
    ``add`` / ``search`` 行为的普通对象。
    """

    def __init__(
        self,
        db_path: Path,
        embedder: Optional[Any] = None,
        max_records: Optional[int] = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or HashingEmbeddingModel()
        # 有界存储：设置后，``add`` 会在超过上限时清除最旧的记录。
        self.max_records = int(max_records) if max_records is not None else None
        # ``timeout`` + ``busy_timeout`` 使繁忙的数据库在并发访问下短暂等待，
        # 而不是立即抛错。该存储目前是单线程的，但这样更安全。
        self.conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        self.conn.execute("PRAGMA busy_timeout = 30000")
        self._init_schema()
        # 查询结果缓存，键为 (瓶颈描述, 硬件指纹, top_k)：相同查询可跳过
        # 全表扫描和余弦相似度重算。
        self._search_cache: Dict[tuple, List[RetrievedCase]] = {}
        self._cache_limit = 128

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                record_id TEXT PRIMARY KEY,
                vector_json TEXT NOT NULL,
                bottleneck_description TEXT NOT NULL,
                hardware_json TEXT NOT NULL,
                solution_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                bottleneck_state_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def add(
        self,
        bottleneck_description: str,
        hardware: Dict[str, Any],
        solution: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        bottleneck_state: Optional[Dict[str, Any]] = None,
    ) -> HistoryRecord:
        vector = build_case_vector(self.embedder, bottleneck_description, hardware)
        record_id = hashlib.sha256(
            (hardware_fingerprint(hardware) + bottleneck_description).encode("utf-8")
        ).hexdigest()[:24]
        record = HistoryRecord(
            record_id=record_id,
            vector=vector,
            bottleneck_description=bottleneck_description,
            hardware=hardware,
            solution=solution,
            metrics=metrics or {},
            metadata=metadata or {},
            bottleneck_state=bottleneck_state or {},
        )
        self.conn.execute(
            """
            INSERT OR REPLACE INTO history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.record_id,
                json.dumps(vector),
                bottleneck_description,
                json.dumps(hardware, ensure_ascii=False),
                json.dumps(solution, ensure_ascii=False),
                json.dumps(metrics or {}, ensure_ascii=False),
                json.dumps(metadata or {}, ensure_ascii=False),
                json.dumps(bottleneck_state or {}, ensure_ascii=False),
                record.created_at,
            ),
        )
        self.conn.commit()
        # 新数据使缓存的搜索结果失效。
        self._search_cache.clear()
        self._enforce_capacity()
        return record

    def _enforce_capacity(self) -> None:
        if not self.max_records:
            return
        count = int(self.conn.execute("SELECT COUNT(*) FROM history").fetchone()[0])
        if count > self.max_records:
            excess = count - self.max_records
            self.conn.execute(
                "DELETE FROM history WHERE record_id IN ("
                "  SELECT record_id FROM history ORDER BY created_at ASC LIMIT ?"
                ")",
                (excess,),
            )
            self.conn.commit()

    def search(
        self,
        bottleneck_description: str,
        hardware: Dict[str, Any],
        top_k: int = 5,
        bottleneck_state: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[RetrievedCase]:
        cache_key = (
            bottleneck_description, hardware_fingerprint(hardware),
            int(top_k), float(min_score),
        )
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        query_vector = build_case_vector(self.embedder, bottleneck_description, hardware)
        rows = self.conn.execute("SELECT * FROM history").fetchall()
        scored: List[RetrievedCase] = []
        for row in rows:
            vector = json.loads(row[1])
            score = cosine_similarity(query_vector, vector)
            record = HistoryRecord(
                record_id=row[0],
                vector=vector,
                bottleneck_description=row[2],
                hardware=json.loads(row[3]),
                solution=json.loads(row[4]),
                metrics=json.loads(row[5]),
                metadata=json.loads(row[6]),
                bottleneck_state=json.loads(row[7]),
                created_at=row[8],
            )
            scored.append(RetrievedCase(score=score, record=record))
        # 排序前先剔除低于阈值（如负相关）的案例。
        scored = [case for case in scored if case.score >= min_score]
        scored.sort(key=lambda case: case.score, reverse=True)
        result = scored[:top_k]
        self._cache_result(cache_key, result)
        return list(result)

    def _cache_result(self, cache_key: tuple, result: List[RetrievedCase]) -> None:
        if len(self._search_cache) >= self._cache_limit:
            # 丢弃最旧条目（按字典插入顺序）以限制内存占用。
            self._search_cache.pop(next(iter(self._search_cache)))
        self._search_cache[cache_key] = result

    def __len__(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM history").fetchone()[0])

    def __enter__(self) -> "HistoryVectorStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001 — 幂等关闭
            pass

    # -- LangChain VectorStore 接口 ----------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        texts = list(texts)
        metadatas = metadatas or [{} for _ in texts]
        record_ids: List[str] = []
        for text, meta in zip(texts, metadatas):
            record = self.add(
                bottleneck_description=text,
                hardware=meta.get("hardware", {}),
                solution={},
                metrics=meta.get("metrics", {}),
                metadata=meta,
            )
            record_ids.append(record.record_id)
        return record_ids

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        cases = self.search(query, {}, top_k=k)
        return [
            Document(
                page_content=case.record.bottleneck_description,
                metadata={"record_id": case.record.record_id, "score": round(case.score, 4)},
            )
            for case in cases
        ]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Any,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        db_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> "HistoryVectorStore":
        embedder = embedding if hasattr(embedding, "embed") else _EmbeddingAdapter(embedding)
        store = cls(db_path=db_path, embedder=embedder, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if ids:
            self.conn.executemany(
                "DELETE FROM history WHERE record_id = ?", [(record_id,) for record_id in ids]
            )
            self.conn.commit()
            self._search_cache.clear()
        return None

    def as_history_retriever(
        self,
        top_k: int = 5,
        min_score: float = 0.0,
        hardware_provider: Optional[Any] = None,
    ) -> Any:
        """返回基于双通道搜索的 LangChain ``BaseRetriever``。

        检索器的 ``query`` 是瓶颈描述；当前硬件状态在调用时由
        ``hardware_provider``（零参数可调用对象）提供，因为它每次迭代都会变化。
        这使存储通过框架标准的检索器接口对外暴露，而非私有的 ``search`` 方法。
        """
        if BaseRetriever is None:  # pragma: no cover - 无 langchain 的域测试
            return None
        return HistoryRetriever(
            store=self,
            top_k=top_k,
            min_score=min_score,
            hardware_provider=hardware_provider or (lambda: {}),
        )


class HistoryRetriever(BaseRetriever if BaseRetriever is not None else object):
    """将瓶颈查询映射到双通道存储的 ``BaseRetriever``。

    每个返回的 ``Document`` 在其 metadata 中携带完整的 ``HistoryRecord``
    （record_id / score / hardware / solution / metrics），使
    :func:`summarize_documents` 无需二次往返即可重建面向 LLM 的摘要。
    """

    def __init__(
        self,
        store: "HistoryVectorStore",
        top_k: int = 5,
        min_score: float = 0.0,
        hardware_provider: Optional[Any] = None,
    ) -> None:
        super().__init__()
        # ``BaseRetriever`` 是 Pydantic 模型；store / provider 是运行时协作者而非
        # schema 字段，因此作为普通属性附加。
        object.__setattr__(self, "store", store)
        object.__setattr__(self, "top_k", int(top_k))
        object.__setattr__(self, "min_score", float(min_score))
        object.__setattr__(self, "hardware_provider", hardware_provider or (lambda: {}))

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,  # CallbackManagerForRetrieverRun 类型
    ) -> List[Any]:
        hardware = self.hardware_provider() or {}
        cases = self.store.search(
            query, hardware, top_k=self.top_k, min_score=self.min_score
        )
        docs = []
        for case in cases:
            record = case.record
            docs.append(
                Document(
                    page_content=record.bottleneck_description,
                    metadata={
                        "record_id": record.record_id,
                        "score": round(case.score, 4),
                        "hardware": record.hardware,
                        "solution": record.solution,
                        "metrics": record.metrics,
                        "bottleneck_state": record.bottleneck_state,
                    },
                )
            )
        return docs
