"""EvidenceAnalystAgent -- interprets deterministic ablation evidence (offline).

The measurement layer (:class:`~photon_tuner.evidence.probe.SearchSpaceProbe`) is
deterministic; this agent is the intelligence that reads the raw evidence,
cross-analyses ``param x workload x mapping x effect``, flags anomalies, and
produces actionable conclusions.  Its findings feed RAG enrichment and the next
search run's prompt (verified-knowledge section) -- offline, not in the
per-iteration loop.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

_FINDINGS_SYSTEM = """You are an evidence analyst for a photonic DSE system. You are given raw,
deterministically-measured single-variable ablation records. Each record is one
parameter, one candidate value, one workload, with:

- injection_confirmed: whether the injected sub-architecture readback matched.
- metric_deltas: energy/area/latency/objective percentage changes.
- affected_metrics / is_null: which metrics moved (null = none moved).
- constraint_result: pass / violated / clamped.
- memory_energy_delta_pct: memory-energy change (NOT in the objective).

Produce STRICT JSON with exactly these keys:
{
  "verified_levers": [{"param","workload","old_value","new_value","objective_pct","affected_metrics","confidence","evidence_ids"}],
  "null_results":    [{"param","workload","new_value","explanation","evidence_ids"}],
  "anomalies":       [{"param","workload","claim","evidence","verdict"}],
  "search_policy":   ["..."]   // short actionable recommendations
}

Rules:
- Only report conclusions the evidence supports. Do not invent effects.
- confidence "high" only when injection_confirmed is true AND the delta is
  clearly non-zero. Otherwise "low".
- For null results, prefer a workload-bound explanation (compute vs memory).
- anomalies: flag conflicts between what a delta shows and what a search agent
  would naively expect (e.g. a device swap that regresses).
- search_policy: concrete next-search guidance (e.g. "prioritise work_freq for
  compute-bound workloads", "do not swap mzm to SL_MZM for attenuation").
"""


class EvidenceAnalystAgent:
    """LLM agent that reads ablation evidence and produces structured findings."""

    def __init__(self, llm: Any, store: Any) -> None:
        self.llm = llm
        self.store = store

    @staticmethod
    def _compact(record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": record.get("id"),
            "param": record.get("param"),
            "workload": record.get("workload"),
            "old_value": record.get("old_value"),
            "new_value": record.get("new_value"),
            "constraint_result": record.get("constraint_result"),
            "injection_confirmed": bool(record.get("injection_confirmed")),
            "metric_deltas": record.get("metric_deltas", {}),
            "affected_metrics": record.get("affected_metrics", []),
            "is_null": bool(record.get("is_null")),
            "memory_energy_delta_pct": record.get("memory_energy_delta_pct"),
        }

    def analyze(
        self,
        workloads: Optional[List[str]] = None,
        params: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Query the store, then ask the LLM to interpret the evidence."""
        records = self.store.query(limit=100000)
        if workloads:
            records = [r for r in records if r["workload"] in workloads]
        if params:
            records = [
                r for r in records
                if r["param"] in params or r["param"].startswith("devices.")
            ]

        payload: Dict[str, Any] = {
            "record_count": len(records),
            "records": [self._compact(r) for r in records],
        }
        findings = self.llm.complete_json(_FINDINGS_SYSTEM, payload)
        for key in ("verified_levers", "null_results", "anomalies", "search_policy"):
            findings.setdefault(key, [] if key != "search_policy" else [])
        return findings

    def analyze_and_save(
        self,
        path: Any,
        workloads: Optional[List[str]] = None,
        params: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        findings = self.analyze(workloads=workloads, params=params)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return findings
