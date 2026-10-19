"""SQLite-backed historical case store (RAG memory) for photonic DSE.

Every committed case records its bottleneck description, the full hardware
configuration, the proposed solution, before/after metrics, and a two-part
retrieval vector: one channel encodes the bottleneck semantics and the other
encodes the hardware state, so retrieval is conditioned on both.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
    """Compact, LLM-friendly photonic hardware summary (for retrieval text)."""
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
    """Dual-channel vector: bottleneck semantics + hardware state."""
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


class HistoryVectorStore:
    """SQLite vector store keyed by a lightweight hashing embedder by default."""

    def __init__(self, db_path: Path, embedder: Optional[Any] = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or HashingEmbeddingModel()
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

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
        return record

    def search(
        self,
        bottleneck_description: str,
        hardware: Dict[str, Any],
        top_k: int = 5,
        bottleneck_state: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedCase]:
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
        scored.sort(key=lambda case: case.score, reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM history").fetchone()[0])

    def close(self) -> None:
        self.conn.close()
