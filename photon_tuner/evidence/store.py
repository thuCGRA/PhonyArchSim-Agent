"""SQLite-backed evidence store for search-space ablation records.

Each row is one single-variable perturbation: one parameter, one candidate
value, one workload.  The store is write-only from the deterministic probe and
read by the :class:`~photon_tuner.agents.evidence.EvidenceAnalystAgent` and by
future RAG enrichment.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ablation (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  ts            TEXT,
  param         TEXT NOT NULL,
  layer         TEXT NOT NULL,
  workload      TEXT NOT NULL,
  workload_features TEXT NOT NULL,
  old_value     TEXT,
  new_value     TEXT,
  hardware_map  TEXT NOT NULL,
  sub_arch_map  TEXT NOT NULL,
  constraint_result TEXT NOT NULL,
  clamp_before  TEXT,
  injection_confirmed INTEGER NOT NULL,
  injection_readback TEXT,
  baseline_energy_pj REAL, baseline_area_um2 REAL,
  baseline_latency_s REAL, baseline_objective REAL,
  after_energy_pj REAL, after_area_um2 REAL,
  after_latency_s REAL, after_objective REAL,
  memory_energy_delta_pct REAL,
  metric_deltas  TEXT NOT NULL,
  affected_metrics TEXT NOT NULL,
  is_null        INTEGER NOT NULL,
  trustworthy    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_param ON ablation(param);
CREATE INDEX IF NOT EXISTS idx_workload ON ablation(workload);
CREATE INDEX IF NOT EXISTS idx_param_workload ON ablation(param, workload);
"""

# (column, key) pairs in INSERT order.
_COLUMNS: List[str] = [
    "ts", "param", "layer", "workload", "workload_features", "old_value",
    "new_value", "hardware_map", "sub_arch_map", "constraint_result",
    "clamp_before", "injection_confirmed", "injection_readback",
    "baseline_energy_pj", "baseline_area_um2", "baseline_latency_s",
    "baseline_objective", "after_energy_pj", "after_area_um2",
    "after_latency_s", "after_objective", "memory_energy_delta_pct",
    "metric_deltas", "affected_metrics", "is_null", "trustworthy",
]

_JSON_KEYS = {
    "workload_features", "hardware_map", "sub_arch_map", "injection_readback",
    "metric_deltas", "affected_metrics",
}


class EvidenceStore:
    """Persist ablation evidence to ``data/evidence.db`` (independent of runs)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def record(self, record: Dict[str, Any]) -> int:
        """Insert one ablation record. JSON-serialises structured fields."""
        record = dict(record)
        record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
        values = []
        for col in _COLUMNS:
            v = record.get(col)
            if col in _JSON_KEYS:
                v = json.dumps(v if v is not None else {}, ensure_ascii=False)
            values.append(v)
        placeholders = ", ".join("?" for _ in _COLUMNS)
        cur = self.conn.execute(
            f"INSERT INTO ablation ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
            values,
        )
        self.conn.commit()
        return cur.lastrowid

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM ablation").fetchone()[0]

    def query(
        self,
        param: Optional[str] = None,
        workload: Optional[str] = None,
        is_null: Optional[bool] = None,
        injection_confirmed: Optional[bool] = None,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        args: List[Any] = []
        if param is not None:
            clauses.append("param = ?")
            args.append(param)
        if workload is not None:
            clauses.append("workload = ?")
            args.append(workload)
        if is_null is not None:
            clauses.append("is_null = ?")
            args.append(int(is_null))
        if injection_confirmed is not None:
            clauses.append("injection_confirmed = ?")
            args.append(int(injection_confirmed))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            f"SELECT * FROM ablation {where} ORDER BY id LIMIT ?", [*args, limit]
        ).fetchall()
        return [self._decode(dict(r)) for r in rows]

    def _decode(self, row: Dict[str, Any]) -> Dict[str, Any]:
        for key in _JSON_KEYS:
            if row.get(key):
                try:
                    row[key] = json.loads(row[key])
                except (TypeError, json.JSONDecodeError):
                    pass
        return row

    def export(self, path: str | Path | None = None) -> Path:
        """Dump all records as JSONL (one JSON object per line)."""
        out = Path(path) if path else self.path.with_suffix(".jsonl")
        rows = self.conn.execute("SELECT * FROM ablation ORDER BY id").fetchall()
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(self._decode(dict(r)), ensure_ascii=False) + "\n")
        return out

    def close(self) -> None:
        self.conn.close()
