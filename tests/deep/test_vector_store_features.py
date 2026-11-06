"""RAG 存储新特性的测试。

覆盖 LangChain ``VectorStore`` 接口对齐，以及检索相似度下限与有界存储容量。
"""

import tempfile
import unittest
from pathlib import Path

from photon_tuner.rag.embeddings import HashingEmbeddingModel
from photon_tuner.rag.vector_store import HistoryVectorStore, summarize_documents


class TestVectorStoreFeatures(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "store.db"
        self.store = HistoryVectorStore(self.db, embedder=HashingEmbeddingModel(dimension=32))

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_similarity_search_returns_documents(self):
        self.store.add_texts(["conv1 energy bottleneck", "memory bandwidth issue"])
        docs = self.store.similarity_search("conv1 energy", k=2)
        self.assertGreaterEqual(len(docs), 1)
        self.assertIn("conv1", docs[0].page_content)

    def test_add_texts_and_delete_roundtrip(self):
        ids = self.store.add_texts(["case a", "case b"])
        self.assertEqual(len(ids), 2)
        self.store.delete(ids=[ids[0]])
        self.assertEqual(len(self.store), 1)

    def test_min_score_excludes_below_threshold(self):
        self.store.add("laser power", {"arch": {"x": 1}}, solution={}, metrics={})
        # 相同查询得分为 1.0；高于 1.0 的下限会过滤掉一切。
        all_cases = self.store.search("laser power", {"arch": {"x": 1}}, top_k=5, min_score=0.0)
        self.assertGreaterEqual(len(all_cases), 1)
        none = self.store.search("laser power", {"arch": {"x": 1}}, top_k=5, min_score=1.01)
        self.assertEqual(len(none), 0)

    def test_max_records_prunes_oldest(self):
        db = Path(self._tmp.name) / "cap.db"
        store = HistoryVectorStore(db, embedder=HashingEmbeddingModel(dimension=32), max_records=2)
        try:
            for i in range(5):
                store.add(f"case {i}", {"arch": {"i": i}}, solution={}, metrics={})
            self.assertEqual(len(store), 2)
        finally:
            store.close()

    def test_as_history_retriever_roundtrip(self):
        # 框架标准 retriever 接口须把已存案例往返成 Documents，
        # 使 summarize_documents 能无损重建。
        self.store.add(
            "laser power too low",
            {"arch": {"power": 5}},
            solution={"fix": "raise power"},
            metrics={"loss": 1.0},
        )
        retriever = self.store.as_history_retriever(
            top_k=5, min_score=0.0, hardware_provider=lambda: {"arch": {"power": 5}}
        )
        self.assertIsNotNone(retriever)
        docs = retriever.invoke("laser power too low")
        self.assertGreaterEqual(len(docs), 1)
        self.assertIn("laser power", docs[0].page_content)

        summaries = summarize_documents(docs, limit=5)
        self.assertGreaterEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["solution"], {"fix": "raise power"})
        self.assertAlmostEqual(summaries[0]["score"], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
