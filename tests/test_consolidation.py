"""Tests for memory consolidation system."""
import json
import tempfile
import time
import unittest
from pathlib import Path

from memory.consolidation import (
    MemoryEntry, MemoryConsolidator, ConsolidationResult,
    NegativeExampleStore,
)


class TestMemoryEntry(unittest.TestCase):
    def test_age_days(self):
        entry = MemoryEntry(
            id="1", content="test", memory_type="pattern",
            created_at=time.time() - 86400 * 3,
        )
        self.assertAlmostEqual(entry.age_days, 3.0, delta=0.01)

    def test_effective_confidence_fresh(self):
        entry = MemoryEntry(
            id="1", content="test", memory_type="pattern",
            confidence=0.8, access_count=0,
        )
        # Fresh entry, no decay, no access boost
        self.assertAlmostEqual(entry.effective_confidence, 0.8, delta=0.05)

    def test_effective_confidence_with_access_boost(self):
        entry = MemoryEntry(
            id="1", content="test", memory_type="pattern",
            confidence=0.5, access_count=4,
        )
        # 0.5 + 4*0.05 = 0.7
        self.assertAlmostEqual(entry.effective_confidence, 0.7, delta=0.05)

    def test_effective_confidence_capped(self):
        entry = MemoryEntry(
            id="1", content="test", memory_type="pattern",
            confidence=0.9, access_count=20,
        )
        self.assertLessEqual(entry.effective_confidence, 1.0)

    def test_retention_score(self):
        entry = MemoryEntry(
            id="1", content="test", memory_type="pattern",
            confidence=0.8, access_count=5,
        )
        score = entry.retention_score
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_negative_sentiment_boosts_retention(self):
        neg = MemoryEntry(
            id="1", content="failed", memory_type="error",
            confidence=0.5, sentiment=-0.8,
        )
        pos = MemoryEntry(
            id="2", content="worked", memory_type="decision",
            confidence=0.5, sentiment=0.5,
        )
        # Negative examples get higher sentiment weight
        self.assertGreater(neg.retention_score, pos.retention_score - 0.1)


class TestMemoryConsolidator(unittest.TestCase):
    def _make_entries(self, n: int, **kwargs) -> list[MemoryEntry]:
        return [
            MemoryEntry(id=str(i), content=f"memory {i}",
                        memory_type="decision", **kwargs)
            for i in range(n)
        ]

    def test_prune_removes_low_retention(self):
        consolidator = MemoryConsolidator(retention_threshold=0.5)
        entries = [
            MemoryEntry(id="1", content="good", memory_type="pattern",
                        confidence=0.9, access_count=5),
            MemoryEntry(id="2", content="bad", memory_type="decision",
                        confidence=0.0, access_count=0),
        ]
        retained, result = consolidator.consolidate(entries)
        self.assertLess(len(retained), len(entries))
        self.assertGreater(result.pruned, 0)

    def test_merge_duplicates(self):
        consolidator = MemoryConsolidator()
        entries = [
            MemoryEntry(id="1", content="hello world", memory_type="pattern",
                        confidence=0.8, access_count=2),
            MemoryEntry(id="2", content="Hello World", memory_type="pattern",
                        confidence=0.6, access_count=3),
        ]
        retained, result = consolidator.consolidate(entries)
        self.assertEqual(result.merged, 1)
        # Merged entry has higher confidence and combined access
        merged = retained[0]
        self.assertEqual(merged.confidence, 0.8)
        self.assertEqual(merged.access_count, 5)

    def test_capacity_enforcement(self):
        consolidator = MemoryConsolidator(max_memories=5,
                                          retention_threshold=0.0)
        entries = self._make_entries(10, confidence=0.8, access_count=5)
        retained, result = consolidator.consolidate(entries)
        self.assertLessEqual(len(retained), 5)

    def test_compression_ratio(self):
        consolidator = MemoryConsolidator(max_memories=3,
                                          retention_threshold=0.0)
        entries = self._make_entries(10, confidence=0.8, access_count=5)
        _, result = consolidator.consolidate(entries)
        self.assertGreater(result.compression_ratio, 0.0)

    def test_empty_input(self):
        consolidator = MemoryConsolidator()
        retained, result = consolidator.consolidate([])
        self.assertEqual(result.memories_before, 0)
        self.assertEqual(result.memories_after, 0)

    def test_summarize_with_summarizer(self):
        consolidator = MemoryConsolidator(summarize_after_days=0)

        def mock_summarizer(contents):
            return f"Summary of {len(contents)} items"

        entries = self._make_entries(6, confidence=0.8, access_count=5,
                                     created_at=time.time() - 86400 * 10)
        retained, result = consolidator.consolidate(entries,
                                                     summarizer=mock_summarizer)
        self.assertGreater(result.summarized, 0)

    def test_no_summarize_without_summarizer(self):
        consolidator = MemoryConsolidator()
        entries = self._make_entries(6, confidence=0.8, access_count=5,
                                     created_at=time.time() - 86400 * 10)
        _, result = consolidator.consolidate(entries)
        self.assertEqual(result.summarized, 0)


class TestNegativeExampleStore(unittest.TestCase):
    def test_add_and_retrieve(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            store = NegativeExampleStore(Path(f.name))
            store.add("fix auth", "timeout error", error_type="timeout")
            self.assertEqual(len(store.examples), 1)

    def test_find_similar_failures(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            store = NegativeExampleStore(Path(f.name))
            store.add("fix auth login", "token expired",
                       error_type="auth")
            store.add("refactor database", "migration failed",
                       error_type="db")
            similar = store.find_similar_failures("fix auth bug")
            self.assertEqual(len(similar), 1)
            self.assertIn("auth", similar[0]["goal"])

    def test_summary_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            store = NegativeExampleStore(Path(f.name))
            summary = store.get_summary()
            self.assertEqual(summary["total"], 0)

    def test_summary_with_data(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            store = NegativeExampleStore(Path(f.name))
            store.add("g1", "r1", error_type="timeout")
            store.add("g2", "r2", error_type="timeout")
            store.add("g3", "r3", error_type="auth")
            summary = store.get_summary()
            self.assertEqual(summary["total"], 3)
            self.assertEqual(summary["most_common"], "timeout")

    def test_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
            store1 = NegativeExampleStore(path)
            store1.add("goal1", "reason1")

            store2 = NegativeExampleStore(path)
            self.assertEqual(len(store2.examples), 1)


if __name__ == "__main__":
    unittest.main()
