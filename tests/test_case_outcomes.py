"""Case qualification must agree across memory statistics and agent evidence."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from photon_tuner.rag.embeddings import HashingEmbeddingModel
from photon_tuner.rag.vector_store import (
    HistoryVectorStore, case_outcome, summarize_cases, summarize_documents,
)
from photon_tuner.runtime.scheduler import ScheduleSignals, decide


class CaseOutcomeTest(unittest.TestCase):
    def test_default_five_percent_boundary(self):
        for after, expected in ((95.1, "unchanged"), (95.0, "improved"), (94.9, "improved")):
            with self.subTest(after=after):
                outcome = case_outcome({"before": {"objective": 100}, "after": {"objective": after}})
                self.assertEqual(outcome["label"], expected)

    def test_custom_threshold_and_existing_symmetric_classification(self):
        metrics = {"before": {"objective": 100}, "after": {"objective": 80}}
        self.assertEqual(case_outcome(metrics)["label"], "improved")
        self.assertEqual(case_outcome(metrics, improved_delta_pct=25)["label"], "unchanged")
        self.assertEqual(case_outcome(metrics, improved_delta_pct=15)["label"], "improved")
        for after, expected in ((104.9, "unchanged"), (105, "worse"), (105.1, "worse")):
            with self.subTest(after=after):
                self.assertEqual(case_outcome({
                    "before": {"objective": 100}, "after": {"objective": after},
                })["label"], expected)

    def test_memory_and_retrieval_use_the_same_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "cases.db"
            with HistoryVectorStore(db, embedder=HashingEmbeddingModel(dimension=32)) as store:
                records = []
                for layer, after in (("device", 95.1), ("architecture", 95.0), ("topology", 94.0)):
                    records.append(store.add(
                        "power budget", {"case": layer}, solution={},
                        metrics={"before": {"objective": 100}, "after": {"objective": after}},
                        metadata={"layers": [layer]},
                    ))
                self.assertEqual(store.improved_case_count(), 2)
                stats = store.improved_case_stats()
                self.assertEqual(stats, {"count": 2, "layers": ["architecture", "topology"]})
                cases = store.search("power budget", {}, top_k=3)
                docs = [SimpleNamespace(metadata={
                    "record_id": c.record.record_id, "metrics": c.record.metrics,
                    "hardware": c.record.hardware, "solution": c.record.solution, "score": c.score,
                }) for c in cases]
                case_labels = [s["outcome"]["label"] for s in summarize_cases(cases)]
                doc_labels = [s["outcome"]["label"] for s in summarize_documents(docs)]
                self.assertEqual(case_labels, doc_labels)
                self.assertEqual(case_labels.count("improved"), 2)
                self.assertEqual(case_labels.count("unchanged"), 1)
                decision = decide(ScheduleSignals(
                    iteration=4, improved_case_count=stats["count"], improved_case_layers=stats["layers"],
                    objective_history=[100, 80, 64, 51.2],
                    pipeline_evidence_ages={"architecture": 1, "topology": 2, "device": 1},
                ))
                self.assertEqual(decision.mode, "serial")  # Two qualifying layers are still immature.
            # Qualification is recomputed from stored metrics when memory is reopened.
            with HistoryVectorStore(db, improved_delta_pct=50) as stricter_store:
                self.assertEqual(stricter_store.improved_case_stats(), {"count": 0, "layers": []})


if __name__ == "__main__":
    unittest.main()
