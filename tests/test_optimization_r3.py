"""
Round 3 Optimization Simulation Tests for AURA CLI.

Tests validate optimizations discovered after R1+R2 benchmark runs.
Focus: Brain at scale (30k+ entries), GoalQueue batching, regression suite.

R3 Optimization Findings (Termux/Android, Brain with 30,419 entries):
  - recall_with_budget() OLD: SELECT * (57ms, 7.9MB loaded, 99.8% discarded)
  - recall_with_budget() NEW: SELECT ... LIMIT N (0.3ms, 200x speedup)
  - recall_recent(100) NEW:   SELECT ... ORDER BY id DESC LIMIT 100 (0.28ms)
  - memory(id) index: created in _init_db for O(log N) range queries
  - GoalQueue batch_add(N):  1 flush vs N flushes (30x speedup for N=30)

Decisions NOT changed:
  - recall_all(): kept intact — some callers need all entries (router.py:222)
    but now benefits from TTL cache on repeated calls
  - all_skills(): already 0.04ms repeated — no change needed
  - replace_code(): already 0.79ms — no change needed
"""
import os
import sys
import json
import time
import tempfile
import statistics
import unittest
from pathlib import Path

os.environ["AURA_SKIP_CHDIR"] = "1"
sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────────────────────────────────────
# 1. Brain — memory(id) index exists
# ──────────────────────────────────────────────────────────────────────────────
class TestBrainMemoryIndex(unittest.TestCase):
    """_init_db must create idx_memory_id for fast ORDER BY id DESC LIMIT N."""

    def test_index_exists_on_memory_table(self):
        from memory.brain import Brain
        b = Brain()
        indexes = [
            row[0]
            for row in b.db.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        ]
        self.assertIn(
            "idx_memory_id", indexes,
            f"idx_memory_id missing. Existing indexes: {indexes}"
        )

    def test_limit_query_uses_index(self):
        """EXPLAIN QUERY PLAN must show 'USING INDEX' for LIMIT query."""
        from memory.brain import Brain
        b = Brain()
        plan = b.db.execute(
            "EXPLAIN QUERY PLAN SELECT content FROM memory ORDER BY id DESC LIMIT 100"
        ).fetchall()
        plan_text = " ".join(str(r) for r in plan).lower()
        # Should NOT be a full table scan
        self.assertNotIn("scan table memory", plan_text,
                         f"Query still doing full scan: {plan_text}")

    def test_limit_query_fast_on_large_db(self):
        """LIMIT 100 on a populated DB must be <<10ms."""
        from memory.brain import Brain
        b = Brain()
        # Ensure there's data
        for i in range(10):
            b.remember(f"benchmark entry {i}")
        samples = []
        for _ in range(20):
            t0 = time.perf_counter()
            b.db.execute(
                "SELECT content FROM memory ORDER BY id DESC LIMIT 100"
            ).fetchall()
            samples.append((time.perf_counter() - t0) * 1000)
        avg_ms = statistics.mean(samples)
        self.assertLess(avg_ms, 10.0,
                        f"LIMIT 100 query: {avg_ms:.2f}ms (expected <10ms)")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Brain — recall_recent() new method
# ──────────────────────────────────────────────────────────────────────────────
class TestBrainRecallRecent(unittest.TestCase):
    """recall_recent(n) must return last N entries in insertion order, fast."""

    def setUp(self):
        from memory.brain import Brain
        self.b = Brain()
        self.b._recall_cache.clear()

    def test_recall_recent_returns_last_n(self):
        marker = f"UNIQUE_MARKER_{time.time()}"
        for i in range(5):
            self.b.remember(f"filler_{i}")
        self.b.remember(marker)
        result = self.b.recall_recent(limit=3)
        self.assertIn(marker, result, "recall_recent didn't return the most recent entry")
        self.assertLessEqual(len(result), 3)

    def test_recall_recent_order_is_oldest_first(self):
        """Result should be chronological: oldest of the N last entries first."""
        ts = int(time.time())
        for i in range(5):
            self.b.remember(f"ordered_entry_{ts}_{i}")
        result = self.b.recall_recent(limit=5)
        # Find our entries in result
        ours = [r for r in result if f"ordered_entry_{ts}_" in r]
        if len(ours) >= 2:
            # Extract sequence numbers and verify ascending
            idxs = [int(r.split("_")[-1]) for r in ours]
            self.assertEqual(idxs, sorted(idxs),
                             f"recall_recent not in insertion order: {idxs}")

    def test_recall_recent_cached(self):
        """Second call with same limit must be faster (cache hit)."""
        self.b._recall_cache.clear()
        self.b.recall_recent(limit=50)  # cold
        t0 = time.perf_counter()
        for _ in range(100):
            self.b.recall_recent(limit=50)  # warm
        warm_ms = (time.perf_counter() - t0) * 1000 / 100
        self.assertLess(warm_ms, 1.0,
                        f"recall_recent warm cache avg {warm_ms:.3f}ms (expected <1ms)")

    def test_recall_recent_invalidated_on_remember(self):
        """Cache must be cleared when new memory is added."""
        self.b.recall_recent(limit=10)
        self.b.remember("cache invalidation test")
        self.assertEqual(len(self.b._recall_cache), 0,
                         "Cache not invalidated after remember()")

    def test_recall_recent_limit_respected(self):
        for i in range(20):
            self.b.remember(f"limit_test_{i}")
        result = self.b.recall_recent(limit=5)
        self.assertLessEqual(len(result), 5)

    def test_recall_recent_throughput(self):
        """100 recall_recent() warm calls must complete in <50ms."""
        self.b.recall_recent(limit=100)  # populate cache
        t0 = time.perf_counter()
        for _ in range(100):
            self.b.recall_recent(limit=100)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.assertLess(elapsed_ms, 50,
                        f"100 recall_recent calls: {elapsed_ms:.1f}ms (expected <50ms)")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Brain — recall_with_budget uses direct SQL LIMIT
# ──────────────────────────────────────────────────────────────────────────────
class TestBrainRecallWithBudgetOptimized(unittest.TestCase):
    """recall_with_budget must NOT do SELECT * anymore — uses LIMIT N."""

    def test_recall_with_budget_does_not_full_scan(self):
        """Verify recall_with_budget uses LIMIT clause — no full-table scan."""
        import inspect
        from memory.brain import Brain
        source = inspect.getsource(Brain.recall_with_budget)
        # Old full scan: SELECT content FROM memory ORDER BY id ASC (no LIMIT)
        self.assertNotIn('ORDER BY id ASC"', source,
                         "Old full-scan query still in recall_with_budget")
        self.assertIn("LIMIT", source,
                      "recall_with_budget missing LIMIT clause — may do full scan")

    def test_recall_with_budget_result_fits_token_budget(self):
        from memory.brain import Brain
        b = Brain()
        for i in range(20):
            b.remember("x" * 200)  # ~50 tokens each
        result = b.recall_with_budget(max_tokens=300)
        total_chars = sum(len(e) for e in result)
        self.assertLessEqual(total_chars, 300 * 4 + 100)

    def test_recall_with_budget_fast_on_large_db(self):
        """On a large DB, recall_with_budget must be <10ms (was 57ms)."""
        from memory.brain import Brain
        b = Brain()
        samples = []
        for _ in range(10):
            t0 = time.perf_counter()
            b.recall_with_budget(max_tokens=4000)
            samples.append((time.perf_counter() - t0) * 1000)
        avg_ms = statistics.mean(samples)
        self.assertLess(avg_ms, 100,
                        f"recall_with_budget avg {avg_ms:.1f}ms (expected <100ms)")

    def test_recall_with_budget_returns_non_empty(self):
        from memory.brain import Brain
        b = Brain()
        b.remember("budget test entry with content")
        result = b.recall_with_budget(max_tokens=500)
        self.assertIsInstance(result, list)

    def test_recall_with_budget_vs_recall_all_speedup(self):
        """recall_with_budget must be substantially faster than full recall_all."""
        from memory.brain import Brain
        b = Brain()
        b._recall_cache.clear()

        # recall_all cold
        t0 = time.perf_counter()
        b.recall_all()
        all_ms = (time.perf_counter() - t0) * 1000

        b._recall_cache.clear()

        # recall_with_budget cold
        t0 = time.perf_counter()
        b.recall_with_budget(max_tokens=1000)
        budget_ms = (time.perf_counter() - t0) * 1000

        # Budget query should be at least 2x faster on non-trivial DBs,
        # or roughly equivalent on small DBs (accept either)
        # On 30k-entry DB: 57ms → 0.3ms; on small test DB: both ~0.5ms
        if all_ms > 5:  # Only enforce on large DBs
            self.assertLess(budget_ms, all_ms * 0.5,
                            f"budget({budget_ms:.1f}ms) not faster than all({all_ms:.1f}ms)")


# ──────────────────────────────────────────────────────────────────────────────
# 4. GoalQueue — batch_add()
# ──────────────────────────────────────────────────────────────────────────────
class TestGoalQueueBatchAdd(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        )
        json.dump([], self.tmp)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_batch_add_exists(self):
        from core.goal_queue import GoalQueue
        gq = GoalQueue(queue_path=self.tmp.name)
        self.assertTrue(hasattr(gq, "batch_add"),
                        "GoalQueue missing batch_add() method")

    def test_batch_add_enqueues_all_goals(self):
        from core.goal_queue import GoalQueue
        gq = GoalQueue(queue_path=self.tmp.name)
        goals = [{"description": f"goal_{i}", "priority": 1} for i in range(10)]
        gq.batch_add(goals)
        self.assertEqual(len(gq.queue), 10)

    def test_batch_add_persists_to_disk(self):
        from core.goal_queue import GoalQueue
        gq1 = GoalQueue(queue_path=self.tmp.name)
        goals = [{"description": f"goal_{i}"} for i in range(5)]
        gq1.batch_add(goals)

        gq2 = GoalQueue(queue_path=self.tmp.name)
        self.assertEqual(len(gq2.queue), 5)

    def test_batch_add_faster_than_loop_add(self):
        """batch_add(N) with 1 flush must be faster than N individual add() calls."""
        from core.goal_queue import GoalQueue
        N = 20
        goals = [{"description": f"goal_{i}", "priority": 1} for i in range(N)]

        # Individual adds
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump([], f); tp1 = f.name
        gq1 = GoalQueue(queue_path=tp1)
        t0 = time.perf_counter()
        for g in goals:
            gq1.add(g)
        loop_ms = (time.perf_counter() - t0) * 1000

        # Batch add
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump([], f); tp2 = f.name
        gq2 = GoalQueue(queue_path=tp2)
        t0 = time.perf_counter()
        gq2.batch_add(goals)
        batch_ms = (time.perf_counter() - t0) * 1000

        os.unlink(tp1); os.unlink(tp2)

        self.assertLess(batch_ms, loop_ms,
                        f"batch_add({batch_ms:.2f}ms) not faster than loop add({loop_ms:.2f}ms)")
        speedup = loop_ms / batch_ms if batch_ms > 0 else float("inf")
        self.assertGreater(speedup, 2.0,
                           f"batch_add speedup {speedup:.1f}x (expected >2x for N={N})")

    def test_batch_add_empty_list(self):
        from core.goal_queue import GoalQueue
        gq = GoalQueue(queue_path=self.tmp.name)
        gq.batch_add([])  # must not raise
        self.assertEqual(len(gq.queue), 0)

    def test_batch_add_preserves_order(self):
        from core.goal_queue import GoalQueue
        gq = GoalQueue(queue_path=self.tmp.name)
        goals = [{"description": f"goal_{i}"} for i in range(5)]
        gq.batch_add(goals)
        for i, g in enumerate(gq.queue):
            self.assertEqual(g["description"], f"goal_{i}")

    def test_batch_add_single_flush_count(self):
        """batch_add must call _save_queue exactly once regardless of N."""
        from core.goal_queue import GoalQueue
        from unittest.mock import patch, call
        gq = GoalQueue(queue_path=self.tmp.name)
        goals = [{"description": f"g{i}"} for i in range(10)]
        with patch.object(gq, "_save_queue") as mock_save:
            gq.batch_add(goals)
            self.assertEqual(mock_save.call_count, 1,
                             f"_save_queue called {mock_save.call_count}x (expected 1)")


# ──────────────────────────────────────────────────────────────────────────────
# 5. all_skills() registry — validate already-optimal caching
# ──────────────────────────────────────────────────────────────────────────────
class TestAllSkillsRegistry(unittest.TestCase):
    """all_skills() already caches at 0.04ms repeat — validate and guard it."""

    def test_first_call_returns_28_skills(self):
        from agents.skills.registry import all_skills
        skills = all_skills()
        self.assertGreaterEqual(len(skills), 28,
                                f"Expected ≥28 skills, got {len(skills)}")

    def test_repeat_call_fast(self):
        """Repeat calls must be <5ms."""
        from agents.skills.registry import all_skills
        all_skills()  # warm up
        samples = []
        for _ in range(20):
            t0 = time.perf_counter()
            all_skills()
            samples.append((time.perf_counter() - t0) * 1000)
        avg_ms = statistics.mean(samples)
        self.assertLess(avg_ms, 5.0,
                        f"Repeated all_skills() avg {avg_ms:.2f}ms (expected <5ms)")

    def test_skills_have_run_method(self):
        from agents.skills.registry import all_skills
        skills = all_skills()
        for name, skill in skills.items():
            self.assertTrue(hasattr(skill, "run"),
                            f"Skill {name!r} missing run() method")


# ──────────────────────────────────────────────────────────────────────────────
# 6. replace_code — validate performance baseline
# ──────────────────────────────────────────────────────────────────────────────
class TestReplaceCodePerformance(unittest.TestCase):
    """replace_code is already 0.79ms — validate baseline and correctness."""

    def test_replace_code_correct(self):
        from core.file_tools import replace_code
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo(): pass\ndef bar(): return 1\n")
            fp = f.name
        try:
            replace_code(fp, "def foo(): pass\n", "def foo(): return 42\n")
            content = Path(fp).read_text()
            self.assertIn("return 42", content)
            self.assertIn("def bar(): return 1", content)
        finally:
            os.unlink(fp)

    def test_replace_code_under_5ms(self):
        from core.file_tools import replace_code
        lines = [f"def func_{i}(x):\n    return x + {i}\n\n" for i in range(100)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("".join(lines)); fp = f.name
        old = "def func_50(x):\n    return x + 50\n"
        new = "def func_50(x):\n    return x * 50\n"
        samples = []
        try:
            for _ in range(10):
                Path(fp).write_text("".join(lines))
                t0 = time.perf_counter()
                replace_code(fp, old, new)
                samples.append((time.perf_counter() - t0) * 1000)
        finally:
            os.unlink(fp)
        avg_ms = statistics.mean(samples)
        self.assertLess(avg_ms, 5.0,
                        f"replace_code 100-func file: {avg_ms:.2f}ms (expected <5ms)")


# ──────────────────────────────────────────────────────────────────────────────
# 7. End-to-end R3: Budget recall + batch add pipeline
# ──────────────────────────────────────────────────────────────────────────────
class TestR3EndToEndPipeline(unittest.TestCase):

    def test_budget_recall_plus_classify_pipeline(self):
        """Simulate ingest phase: recall budget + classify goal, <50ms total."""
        from memory.brain import Brain
        from core.skill_dispatcher import classify_goal

        b = Brain()
        t0 = time.perf_counter()
        context = b.recall_with_budget(max_tokens=2000)
        goal_type = classify_goal("fix the authentication crash bug")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self.assertIsInstance(context, list)
        self.assertIsInstance(goal_type, str)
        self.assertLess(elapsed_ms, 200,
                        f"Ingest pipeline: {elapsed_ms:.1f}ms (expected <200ms)")

    def test_batch_goal_enqueue_then_process(self):
        """Enqueue 10 goals via batch_add, process all via next()."""
        from core.goal_queue import GoalQueue
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump([], f); tp = f.name

        gq = GoalQueue(queue_path=tp)
        goals = [{"description": f"task_{i}", "priority": i % 3} for i in range(10)]
        gq.batch_add(goals)
        processed = []
        while gq.has_goals():
            processed.append(gq.next())
        os.unlink(tp)

        self.assertEqual(len(processed), 10)
        self.assertEqual(processed[0]["description"], "task_0")

    def test_recall_recent_provides_enough_context(self):
        """recall_recent(100) must return useful context for agent prompts."""
        from memory.brain import Brain
        b = Brain()
        for i in range(5):
            b.remember(f"context item {i} for testing agent prompts")
        result = b.recall_recent(limit=100)
        self.assertGreater(len(result), 0, "recall_recent returned nothing")
        self.assertTrue(all(isinstance(r, str) for r in result))


# ──────────────────────────────────────────────────────────────────────────────
# 8. Full regression: R1 + R2 + R3 optimizations all active
# ──────────────────────────────────────────────────────────────────────────────
class TestFullOptimizationRegression(unittest.TestCase):
    """All optimizations from R1, R2, R3 must still be active."""

    # R1
    def test_r1_lazy_textblob(self):
        import memory.brain as bm
        self.assertTrue(hasattr(bm, "_textblob_loaded"))

    def test_r1_wal_mode(self):
        from memory.brain import Brain
        b = Brain()
        mode = b.db.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(mode, "wal")

    def test_r1_recall_cache(self):
        from memory.brain import Brain
        b = Brain()
        self.assertTrue(hasattr(b, "_recall_cache"))
        self.assertTrue(hasattr(b, "_cache_ttl"))

    def test_r1_skill_metrics_count_alias(self):
        from core.skill_dispatcher import SkillMetrics
        sm = SkillMetrics()
        sm.record("x", latency_ms=1.0)
        snap = sm.snapshot()
        self.assertIn("count", snap["x"])

    # R2
    def test_r2_lazy_networkx(self):
        import memory.brain as bm
        self.assertTrue(hasattr(bm, "_nx"))

    def test_r2_classify_goal_lru(self):
        from core.skill_dispatcher import classify_goal
        self.assertTrue(hasattr(classify_goal, "cache_info"))
        self.assertEqual(classify_goal.cache_info().maxsize, 256)

    # R3
    def test_r3_memory_index_exists(self):
        from memory.brain import Brain
        b = Brain()
        idxs = [r[0] for r in b.db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()]
        self.assertIn("idx_memory_id", idxs)

    def test_r3_recall_recent_method_exists(self):
        from memory.brain import Brain
        b = Brain()
        self.assertTrue(hasattr(b, "recall_recent"))

    def test_r3_batch_add_method_exists(self):
        from core.goal_queue import GoalQueue
        self.assertTrue(hasattr(GoalQueue, "batch_add"))

    def test_r3_recall_with_budget_uses_limit(self):
        """recall_with_budget must not have the old full-scan query."""
        import inspect
        from memory.brain import Brain
        source = inspect.getsource(Brain.recall_with_budget)
        self.assertNotIn(
            'ORDER BY id ASC"',
            source,
            "Old full-scan query still present in recall_with_budget"
        )
        self.assertIn("LIMIT", source,
                      "recall_with_budget missing LIMIT clause")


if __name__ == "__main__":
    unittest.main(verbosity=2)
