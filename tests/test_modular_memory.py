"""Tests for the modular memory architecture (Capability 5, Issue #435).

Covers: MemoryModule interface, WorkingMemory, EpisodicMemory, SemanticMemory,
ProceduralMemory, and MemoryManager.
"""

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from memory.memory_module import MemoryEntry, MemoryModule


class TestMemoryEntry(unittest.TestCase):
    """Basic MemoryEntry dataclass sanity checks."""

    def test_defaults(self):
        entry = MemoryEntry(id="abc", content="hello")
        self.assertEqual(entry.id, "abc")
        self.assertEqual(entry.content, "hello")
        self.assertEqual(entry.metadata, {})
        self.assertEqual(entry.timestamp, 0.0)
        self.assertEqual(entry.score, 0.0)

    def test_custom_fields(self):
        entry = MemoryEntry(
            id="x", content="y", metadata={"k": 1}, timestamp=1.5, score=0.9
        )
        self.assertEqual(entry.metadata["k"], 1)
        self.assertAlmostEqual(entry.timestamp, 1.5)
        self.assertAlmostEqual(entry.score, 0.9)


# ======================================================================
# WorkingMemory
# ======================================================================


class TestWorkingMemory(unittest.TestCase):
    def _make(self, max_entries=10, summarize_fn=None):
        from memory.working_memory import WorkingMemory

        return WorkingMemory(max_entries=max_entries, summarize_fn=summarize_fn)

    def test_write_and_read(self):
        wm = self._make()
        eid = wm.write("first entry")
        self.assertIsInstance(eid, str)
        results = wm.read("first", top_k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "first entry")

    def test_read_empty_query_returns_recent(self):
        wm = self._make()
        for i in range(5):
            wm.write(f"entry {i}")
        results = wm.read("", top_k=3)
        self.assertEqual(len(results), 3)

    def test_eviction(self):
        wm = self._make(max_entries=3)
        for i in range(5):
            wm.write(f"item {i}")
        stats = wm.stats()
        self.assertEqual(stats["count"], 3)
        # Only the three most recent should remain.
        results = wm.read("", top_k=10)
        contents = [r.content for r in results]
        self.assertIn("item 4", contents)
        self.assertNotIn("item 0", contents)

    def test_eviction_with_summarize_fn(self):
        summaries = []

        def fake_summarize(entries):
            summary = f"summary of {len(entries)} entries"
            summaries.append(summary)
            return summary

        wm = self._make(max_entries=3, summarize_fn=fake_summarize)
        for i in range(5):
            wm.write(f"item {i}")
        # Summary should have been called at least once.
        self.assertTrue(len(summaries) > 0)
        # The condensed entry should exist in the buffer.
        results = wm.read("summary", top_k=10)
        self.assertTrue(any("summary" in r.content for r in results))

    def test_delete(self):
        wm = self._make()
        eid = wm.write("delete me")
        self.assertTrue(wm.delete(eid))
        self.assertFalse(wm.delete(eid))  # Already deleted
        self.assertEqual(wm.stats()["count"], 0)

    def test_clear(self):
        wm = self._make()
        wm.write("a")
        wm.write("b")
        wm.clear()
        self.assertEqual(wm.stats()["count"], 0)

    def test_search_is_alias_for_read(self):
        wm = self._make()
        wm.write("hello world")
        read_results = wm.read("hello")
        search_results = wm.search("hello")
        self.assertEqual(len(read_results), len(search_results))

    def test_stats(self):
        wm = self._make(max_entries=20)
        for i in range(10):
            wm.write(f"entry {i}")
        s = wm.stats()
        self.assertEqual(s["type"], "working")
        self.assertEqual(s["count"], 10)
        self.assertEqual(s["max_entries"], 20)
        self.assertAlmostEqual(s["utilization_pct"], 50.0)

    def test_thread_safety(self):
        wm = self._make(max_entries=200)
        errors = []

        def writer(prefix):
            try:
                for i in range(50):
                    wm.write(f"{prefix}-{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(f"t{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertFalse(errors, f"Concurrent writes raised: {errors}")
        self.assertLessEqual(wm.stats()["count"], 200)


# ======================================================================
# EpisodicMemory
# ======================================================================


class TestEpisodicMemory(unittest.TestCase):
    def _make(self, tmp_dir):
        from memory.episodic_memory import EpisodicMemory

        return EpisodicMemory(db_path=os.path.join(tmp_dir, "episodic_test.db"))

    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            em = self._make(tmp)
            eid = em.write(
                "Fixed the login bug by correcting the OAuth flow",
                metadata={"task_id": "T1", "task_type": "bugfix", "outcome": "success"},
            )
            self.assertIsInstance(eid, str)

            results = em.read("OAuth login", top_k=5)
            self.assertGreaterEqual(len(results), 1)
            self.assertIn("OAuth", results[0].content)

    def test_fts_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            em = self._make(tmp)
            em.write("Deployed microservice to production Kubernetes cluster")
            em.write("Refactored database migration scripts for PostgreSQL")
            em.write("Updated frontend React components for dark mode")

            results = em.search("database migration", top_k=5)
            self.assertGreaterEqual(len(results), 1)
            self.assertIn("database", results[0].content.lower())

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            em = self._make(tmp)
            eid = em.write("temporary entry")
            self.assertTrue(em.delete(eid))
            self.assertFalse(em.delete(eid))

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            em = self._make(tmp)
            em.write("entry 1")
            em.write("entry 2")
            em.clear()
            self.assertEqual(em.stats()["count"], 0)

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            em = self._make(tmp)
            em.write("alpha")
            em.write("beta")
            s = em.stats()
            self.assertEqual(s["type"], "episodic")
            self.assertEqual(s["count"], 2)
            self.assertGreater(s["storage_bytes"], 0)

    def test_query_by_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            em = self._make(tmp)
            em.write("step 1", metadata={"task_id": "T42"})
            em.write("step 2", metadata={"task_id": "T42"})
            em.write("other task", metadata={"task_id": "T99"})

            results = em.query_by_task("T42")
            self.assertEqual(len(results), 2)

    def test_query_by_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            em = self._make(tmp)
            em.write("succeeded task", metadata={"outcome": "success"})
            em.write("failed task", metadata={"outcome": "failure"})

            results = em.query_by_outcome("success")
            self.assertEqual(len(results), 1)
            self.assertIn("succeeded", results[0].content)

    def test_thread_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            em = self._make(tmp)
            errors = []

            def writer(prefix):
                try:
                    for i in range(20):
                        em.write(f"{prefix}-{i}")
                except Exception as exc:
                    errors.append(exc)

            threads = [
                threading.Thread(target=writer, args=(f"t{i}",)) for i in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)

            self.assertFalse(errors, f"Concurrent writes raised: {errors}")
            self.assertEqual(em.stats()["count"], 80)


# ======================================================================
# SemanticMemory
# ======================================================================


class TestSemanticMemory(unittest.TestCase):
    def _make(self, tmp_dir):
        from memory.semantic_memory import SemanticMemory

        return SemanticMemory(db_path=os.path.join(tmp_dir, "semantic_test.db"))

    def test_write_and_bm25_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            sm = self._make(tmp)
            sm.write("Python is a programming language used for data science")
            sm.write("JavaScript is used for web development and frontend")
            sm.write("Rust provides memory safety without garbage collection")

            results = sm.search("programming language data", top_k=3)
            self.assertGreaterEqual(len(results), 1)
            # The Python entry should score highest for this query.
            self.assertIn("Python", results[0].content)

    def test_bm25_relevance_ordering(self):
        with tempfile.TemporaryDirectory() as tmp:
            sm = self._make(tmp)
            sm.write("machine learning deep neural networks training")
            sm.write("cooking pasta recipe Italian food")
            sm.write("neural networks machine learning artificial intelligence")

            results = sm.search("machine learning neural", top_k=3)
            self.assertGreaterEqual(len(results), 2)
            # Both ML entries should rank above the cooking one.
            for r in results:
                if "cooking" in r.content:
                    self.fail("Cooking entry should not match ML query")

    def test_empty_query_returns_recent(self):
        with tempfile.TemporaryDirectory() as tmp:
            sm = self._make(tmp)
            sm.write("entry one")
            sm.write("entry two")
            results = sm.search("", top_k=5)
            self.assertEqual(len(results), 2)

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            sm = self._make(tmp)
            eid = sm.write("delete this")
            self.assertTrue(sm.delete(eid))
            self.assertEqual(sm.stats()["count"], 0)

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            sm = self._make(tmp)
            sm.write("a")
            sm.write("b")
            sm.clear()
            self.assertEqual(sm.stats()["count"], 0)

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            sm = self._make(tmp)
            sm.write("fact one")
            s = sm.stats()
            self.assertEqual(s["type"], "semantic")
            self.assertEqual(s["count"], 1)

    def test_thread_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            sm = self._make(tmp)
            errors = []

            def writer(prefix):
                try:
                    for i in range(20):
                        sm.write(f"{prefix} entry number {i} about topic {prefix}")
                except Exception as exc:
                    errors.append(exc)

            threads = [
                threading.Thread(target=writer, args=(f"t{i}",)) for i in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)

            self.assertFalse(errors, f"Concurrent writes raised: {errors}")
            self.assertEqual(sm.stats()["count"], 80)


class TestTokenizer(unittest.TestCase):
    def test_basic_tokenization(self):
        from memory.semantic_memory import tokenize

        tokens = tokenize("Hello World, this is a test!")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertIn("test", tokens)
        # Stopwords should be removed.
        self.assertNotIn("this", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("a", tokens)

    def test_empty_string(self):
        from memory.semantic_memory import tokenize

        self.assertEqual(tokenize(""), [])

    def test_only_stopwords(self):
        from memory.semantic_memory import tokenize

        self.assertEqual(tokenize("the a an is are"), [])


# ======================================================================
# ProceduralMemory
# ======================================================================


class TestProceduralMemory(unittest.TestCase):
    def _make(self, tmp_dir):
        from memory.procedural_memory import ProceduralMemory

        return ProceduralMemory(db_path=os.path.join(tmp_dir, "procedural_test.db"))

    def test_write_and_read_by_task_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            pm = self._make(tmp)
            pm.write(
                "Use pytest with fixtures for unit testing",
                metadata={"task_type": "testing", "strategy_name": "pytest-fixtures"},
            )
            results = pm.read("testing", top_k=5)
            self.assertEqual(len(results), 1)
            self.assertIn("pytest", results[0].content)

    def test_recommend_sorts_by_success_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            pm = self._make(tmp)
            id1 = pm.write(
                "Strategy A: brute force",
                metadata={"task_type": "optimization", "strategy_name": "brute-force"},
            )
            id2 = pm.write(
                "Strategy B: dynamic programming",
                metadata={"task_type": "optimization", "strategy_name": "dp"},
            )
            # Record outcomes: B is better.
            for _ in range(5):
                pm.record_outcome(id1, success=False)
            for _ in range(3):
                pm.record_outcome(id1, success=True)
            for _ in range(4):
                pm.record_outcome(id2, success=True)
            for _ in range(1):
                pm.record_outcome(id2, success=False)

            recommended = pm.recommend("optimization")
            self.assertGreaterEqual(len(recommended), 2)
            # B (dp) should be ranked first (4/5 = 0.8 > 3/8 = 0.375).
            self.assertIn("dynamic programming", recommended[0].content)

    def test_record_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            pm = self._make(tmp)
            sid = pm.write(
                "some strategy",
                metadata={"task_type": "test", "strategy_name": "s1"},
            )
            pm.record_outcome(sid, success=True)
            pm.record_outcome(sid, success=False)
            pm.record_outcome(sid, success=True)

            results = pm.read("test")
            self.assertEqual(len(results), 1)
            meta = results[0].metadata
            self.assertEqual(meta["attempts"], 3)
            self.assertEqual(meta["successes"], 2)
            self.assertAlmostEqual(meta["success_rate"], 0.667, places=2)

    def test_search_by_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            pm = self._make(tmp)
            pm.write("Use caching for repeated API calls", metadata={"task_type": "perf"})
            pm.write("Profile before optimizing", metadata={"task_type": "perf"})

            results = pm.search("caching API")
            self.assertGreaterEqual(len(results), 1)
            self.assertIn("caching", results[0].content)

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            pm = self._make(tmp)
            sid = pm.write("temp", metadata={"task_type": "x"})
            self.assertTrue(pm.delete(sid))
            self.assertEqual(pm.stats()["count"], 0)

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            pm = self._make(tmp)
            pm.write("a", metadata={"task_type": "t"})
            pm.write("b", metadata={"task_type": "t"})
            pm.clear()
            self.assertEqual(pm.stats()["count"], 0)

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            pm = self._make(tmp)
            sid = pm.write("s", metadata={"task_type": "t", "strategy_name": "n"})
            pm.record_outcome(sid, success=True)
            pm.record_outcome(sid, success=False)
            s = pm.stats()
            self.assertEqual(s["type"], "procedural")
            self.assertEqual(s["count"], 1)
            self.assertEqual(s["total_attempts"], 2)
            self.assertEqual(s["total_successes"], 1)
            self.assertAlmostEqual(s["overall_success_rate"], 0.5)


# ======================================================================
# MemoryManager
# ======================================================================


class TestMemoryManager(unittest.TestCase):
    def _make(self, tmp_dir):
        from memory.memory_manager import MemoryManager

        return MemoryManager(config={"db_dir": tmp_dir})

    def test_init_creates_all_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            mm = self._make(tmp)
            self.assertIsNotNone(mm.working)
            self.assertIsNotNone(mm.episodic)
            self.assertIsNotNone(mm.semantic)
            self.assertIsNotNone(mm.procedural)

    def test_search_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            mm = self._make(tmp)
            mm.working.write("working memory entry about Python")
            mm.semantic.write("Python is great for data science")
            mm.episodic.write("Fixed Python import bug")

            results = mm.search_all("Python", top_k=5)
            self.assertIn("working", results)
            self.assertIn("semantic", results)
            self.assertIn("episodic", results)
            # At least working and semantic should have results.
            total = sum(len(v) for v in results.values())
            self.assertGreaterEqual(total, 2)

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            mm = self._make(tmp)
            mm.working.write("x")
            mm.semantic.write("y")
            s = mm.stats()
            self.assertEqual(s["working"]["count"], 1)
            self.assertEqual(s["semantic"]["count"], 1)

    def test_clear_specific_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            mm = self._make(tmp)
            mm.working.write("a")
            mm.semantic.write("b")
            mm.clear("working")
            self.assertEqual(mm.working.stats()["count"], 0)
            self.assertEqual(mm.semantic.stats()["count"], 1)

    def test_clear_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            mm = self._make(tmp)
            mm.working.write("a")
            mm.semantic.write("b")
            mm.clear()
            self.assertEqual(mm.working.stats()["count"], 0)
            self.assertEqual(mm.semantic.stats()["count"], 0)

    def test_export_and_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            mm = self._make(tmp)
            mm.working.write("working fact")
            mm.semantic.write("semantic fact about Python")

            export_path = os.path.join(tmp, "export.json")
            mm.export_all(export_path)
            self.assertTrue(os.path.exists(export_path))

            with open(export_path) as fh:
                data = json.load(fh)
            self.assertIn("working", data)
            self.assertIn("semantic", data)

            # Import into a fresh manager.
            mm2 = self._make(os.path.join(tmp, "fresh"))
            mm2.import_all(export_path)
            self.assertGreaterEqual(mm2.semantic.stats()["count"], 1)

    def test_pre_think_hook(self):
        with tempfile.TemporaryDirectory() as tmp:
            mm = self._make(tmp)
            mm.working.write("The user prefers verbose output")
            mm.semantic.write("Verbose mode prints debug information")

            ctx = {"task": "enable verbose mode"}
            result = mm.pre_think_hook(ctx)
            self.assertIn("memory_context", result)
            # At least working memory should have injected context.
            mc = result["memory_context"]
            total_snippets = sum(len(v) for v in mc.values())
            self.assertGreaterEqual(total_snippets, 1)

    def test_pre_think_hook_empty_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            mm = self._make(tmp)
            ctx = {}
            result = mm.pre_think_hook(ctx)
            # Should return context unchanged (no crash).
            self.assertNotIn("memory_context", result)

    def test_post_observe_hook(self):
        with tempfile.TemporaryDirectory() as tmp:
            mm = self._make(tmp)
            ctx = {
                "observation": "The build succeeded after fixing the import",
                "task_complete": True,
                "task_id": "T123",
                "task_type": "bugfix",
                "outcome": "success",
            }
            mm.post_observe_hook(ctx)
            # Working memory should have the observation.
            self.assertEqual(mm.working.stats()["count"], 1)
            # Episodic memory should also have it (task_complete=True).
            self.assertEqual(mm.episodic.stats()["count"], 1)

    def test_post_observe_hook_no_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            mm = self._make(tmp)
            ctx = {"observation": "Intermediate result"}
            mm.post_observe_hook(ctx)
            # Only working memory should have the observation.
            self.assertEqual(mm.working.stats()["count"], 1)
            self.assertEqual(mm.episodic.stats()["count"], 0)


if __name__ == "__main__":
    unittest.main()
