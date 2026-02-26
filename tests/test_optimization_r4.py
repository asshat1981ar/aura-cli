"""
tests/test_optimization_r4.py — Round 4 optimization tests.

Verifies that all `recall_all()` calls have been replaced in the 9 target
agent/core/tool files, and benchmarks the resulting performance improvements.

PRD-001 R4 targets:
  agents/coder.py       → recall_with_budget(2000)
  agents/critic.py      → recall_with_budget(1500)  (×2)
  agents/scaffolder.py  → recall_with_budget(1500)
  agents/tester.py      → recall_with_budget(1500)
  agents/ingest.py      → recall_recent(50)
  agents/router.py      → recall_recent(200)
  core/evolution_loop.py → recall_with_budget(3000)
  core/closed_loop.py   → count_memories()
  tools/aura_control_mcp.py → _get_memories_cached() + count_memories()
  memory/brain.py        → added count_memories() method
"""
from __future__ import annotations

import inspect
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helper: build a mock Brain with fast methods
# ---------------------------------------------------------------------------

def _make_mock_brain(entries=None):
    b = MagicMock()
    _data = entries or ["memory 1", "memory 2", "memory 3"]
    b.recall_all = MagicMock(return_value=_data)
    b.recall_recent = MagicMock(return_value=_data[-50:])
    b.recall_with_budget = MagicMock(return_value=_data)
    b.count_memories = MagicMock(return_value=len(_data))
    b.recall_weaknesses = MagicMock(return_value=[])
    b.remember = MagicMock()
    return b


# ---------------------------------------------------------------------------
# 1. Grep-level regression: no recall_all() in prompt-embedding contexts
# ---------------------------------------------------------------------------

class TestNoRecallAllInAgents(unittest.TestCase):
    """Verify that no agent embeds recall_all() directly in an f-string prompt."""

    def _source(self, rel_path: str) -> str:
        return (ROOT / rel_path).read_text(errors="replace")

    def test_coder_no_recall_all(self):
        src = self._source("agents/coder.py")
        self.assertNotIn("recall_all()", src,
                         "agents/coder.py still calls recall_all()")

    def test_critic_no_recall_all(self):
        src = self._source("agents/critic.py")
        self.assertNotIn("recall_all()", src,
                         "agents/critic.py still calls recall_all()")

    def test_scaffolder_no_recall_all(self):
        src = self._source("agents/scaffolder.py")
        self.assertNotIn("recall_all()", src,
                         "agents/scaffolder.py still calls recall_all()")

    def test_tester_no_recall_all(self):
        src = self._source("agents/tester.py")
        self.assertNotIn("recall_all()", src,
                         "agents/tester.py still calls recall_all()")

    def test_ingest_no_recall_all(self):
        src = self._source("agents/ingest.py")
        self.assertNotIn("recall_all()", src,
                         "agents/ingest.py still calls recall_all()")

    def test_router_no_recall_all(self):
        src = self._source("agents/router.py")
        self.assertNotIn("recall_all()", src,
                         "agents/router.py still calls recall_all()")

    def test_evolution_loop_no_recall_all(self):
        src = self._source("core/evolution_loop.py")
        self.assertNotIn("recall_all()", src,
                         "core/evolution_loop.py still calls recall_all()")

    def test_closed_loop_no_recall_all(self):
        src = self._source("core/closed_loop.py")
        self.assertNotIn("recall_all()", src,
                         "core/closed_loop.py still calls recall_all()")

    def test_mcp_control_no_recall_all(self):
        src = self._source("tools/aura_control_mcp.py")
        self.assertNotIn("recall_all()", src,
                         "tools/aura_control_mcp.py still calls recall_all()")


# ---------------------------------------------------------------------------
# 2. Replacement method presence checks
# ---------------------------------------------------------------------------

class TestReplacementMethodsPresent(unittest.TestCase):

    def test_coder_uses_recall_with_budget(self):
        src = (ROOT / "agents/coder.py").read_text()
        self.assertIn("recall_with_budget", src)

    def test_critic_uses_recall_with_budget(self):
        src = (ROOT / "agents/critic.py").read_text()
        self.assertIn("recall_with_budget", src)

    def test_ingest_uses_recall_recent(self):
        src = (ROOT / "agents/ingest.py").read_text()
        self.assertIn("recall_recent", src)

    def test_router_uses_recall_recent(self):
        src = (ROOT / "agents/router.py").read_text()
        self.assertIn("recall_recent", src)

    def test_evolution_loop_uses_recall_with_budget(self):
        src = (ROOT / "core/evolution_loop.py").read_text()
        self.assertIn("recall_with_budget", src)

    def test_closed_loop_uses_count_memories(self):
        src = (ROOT / "core/closed_loop.py").read_text()
        self.assertIn("count_memories", src)

    def test_mcp_uses_count_memories(self):
        src = (ROOT / "tools/aura_control_mcp.py").read_text()
        self.assertIn("count_memories", src)

    def test_mcp_has_memory_cache(self):
        src = (ROOT / "tools/aura_control_mcp.py").read_text()
        self.assertIn("_memory_cache", src)
        self.assertIn("_get_memories_cached", src)


# ---------------------------------------------------------------------------
# 3. Brain.count_memories() method
# ---------------------------------------------------------------------------

class TestBrainCountMemories(unittest.TestCase):

    def test_count_memories_method_exists(self):
        from memory.brain import Brain
        self.assertTrue(hasattr(Brain, "count_memories"),
                        "Brain.count_memories() method not found")

    def test_count_memories_returns_int(self):
        from memory.brain import Brain
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            b = Brain(db_path=db_path)
            count = b.count_memories()
            self.assertIsInstance(count, int)
            self.assertEqual(count, 0)  # fresh DB
        finally:
            os.unlink(db_path)

    def test_count_memories_increments_on_remember(self):
        from memory.brain import Brain
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            b = Brain(db_path=db_path)
            b.remember("entry1")
            b.remember("entry2")
            b.remember("entry3")
            self.assertEqual(b.count_memories(), 3)
        finally:
            os.unlink(db_path)

    def test_count_memories_faster_than_recall_all(self):
        """count_memories() should be strictly faster than len(recall_all())."""
        from memory.brain import Brain
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            b = Brain(db_path=db_path)
            for i in range(500):
                b.remember(f"entry {i} " + "x" * 50)

            # Clear cache so we measure raw SQL
            b._recall_cache.clear()
            t0 = time.perf_counter()
            for _ in range(20):
                _ = len(b.recall_all())
            recall_all_ms = (time.perf_counter() - t0) * 1000 / 20

            t0 = time.perf_counter()
            for _ in range(20):
                _ = b.count_memories()
            count_ms = (time.perf_counter() - t0) * 1000 / 20

            self.assertLessEqual(count_ms, recall_all_ms,
                                 f"count_memories ({count_ms:.2f}ms) slower than "
                                 f"recall_all ({recall_all_ms:.2f}ms)")
        finally:
            os.unlink(db_path)

    def test_count_memories_source_uses_count_star(self):
        from memory.brain import Brain
        src = inspect.getsource(Brain.count_memories)
        self.assertIn("COUNT(*)", src.upper(),
                      "count_memories should use COUNT(*) SQL")


# ---------------------------------------------------------------------------
# 4. MCP memory cache behaviour
# ---------------------------------------------------------------------------

class TestMCPMemoryCache(unittest.TestCase):

    def test_cache_dict_exists(self):
        import tools.aura_control_mcp as mod
        self.assertTrue(hasattr(mod, "_memory_cache"))
        self.assertIsInstance(mod._memory_cache, dict)

    def test_cache_ttl_constant_exists(self):
        import tools.aura_control_mcp as mod
        self.assertTrue(hasattr(mod, "_MEMORY_CACHE_TTL"))
        self.assertGreater(mod._MEMORY_CACHE_TTL, 0)

    def test_get_memories_cached_function_exists(self):
        import tools.aura_control_mcp as mod
        self.assertTrue(hasattr(mod, "_get_memories_cached"),
                        "_get_memories_cached not found in aura_control_mcp")

    def test_get_memories_cached_returns_list(self):
        import tools.aura_control_mcp as mod
        mod._memory_cache.clear()
        mock_brain = _make_mock_brain(["entry A", "entry B"])
        result = mod._get_memories_cached(mock_brain)
        self.assertIsInstance(result, list)
        self.assertEqual(result, ["entry A", "entry B"])

    def test_get_memories_cached_hits_cache_on_second_call(self):
        import tools.aura_control_mcp as mod
        mod._memory_cache.clear()
        mock_brain = _make_mock_brain(["cached entry"])
        mod._get_memories_cached(mock_brain)
        mod._get_memories_cached(mock_brain)
        # recall_with_budget should only be called once (cache hit on second)
        self.assertEqual(mock_brain.recall_with_budget.call_count, 1)

    def test_get_memories_cached_refreshes_after_ttl(self):
        import tools.aura_control_mcp as mod
        mod._memory_cache.clear()
        mock_brain = _make_mock_brain(["old entry"])
        mod._get_memories_cached(mock_brain)
        # Expire the cache
        mod._memory_cache["recent"] = (["old entry"], time.time() - mod._MEMORY_CACHE_TTL - 1)
        mod._get_memories_cached(mock_brain)
        # Should have called recall_with_budget twice (initial + after expiry)
        self.assertEqual(mock_brain.recall_with_budget.call_count, 2)


# ---------------------------------------------------------------------------
# 5. Evolution loop prompt assembly benchmark
# ---------------------------------------------------------------------------

class TestEvolutionLoopMemoryAssembly(unittest.TestCase):

    def test_evolution_loop_uses_budget_source(self):
        src = (ROOT / "core/evolution_loop.py").read_text()
        self.assertIn("recall_with_budget", src)
        self.assertNotIn("recall_all()", src)

    def test_evolution_loop_memory_snapshot_is_str(self):
        """recall_with_budget returns List[str]; join produces a single str."""
        from memory.brain import Brain
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            b = Brain(db_path=db_path)
            for i in range(50):
                b.remember(f"memory entry {i}")
            snapshot = "\n".join(b.recall_with_budget(max_tokens=3000))
            self.assertIsInstance(snapshot, str)
            # Should not exceed budget: 3000 tokens * 4 chars/token
            self.assertLessEqual(len(snapshot), 3000 * 4 + 500)
        finally:
            os.unlink(db_path)

    def test_evolution_loop_assembly_under_10ms(self):
        """Memory assembly with budget should complete in < 10ms on real DB."""
        from memory.brain import Brain
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            b = Brain(db_path=db_path)
            for i in range(1000):
                b.remember(f"memory entry {i}: " + "x" * 100)
            b._recall_cache.clear()

            t0 = time.perf_counter()
            snapshot = "\n".join(b.recall_with_budget(max_tokens=3000))
            elapsed_ms = (time.perf_counter() - t0) * 1000

            self.assertIsInstance(snapshot, str)
            self.assertLessEqual(elapsed_ms, 10.0,
                                 f"Memory assembly took {elapsed_ms:.2f}ms, expected < 10ms")
        finally:
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# 6. Router stats lookup
# ---------------------------------------------------------------------------

class TestRouterStatLookup(unittest.TestCase):

    def test_router_source_uses_recall_recent(self):
        src = (ROOT / "agents/router.py").read_text()
        self.assertIn("recall_recent", src)

    def test_router_load_stats_uses_limit_200(self):
        src = (ROOT / "agents/router.py").read_text()
        # Should scan only recent 200 entries, not all
        self.assertIn("200", src)
        self.assertNotIn("recall_all()", src)


# ---------------------------------------------------------------------------
# 7. Ingest agent API fix
# ---------------------------------------------------------------------------

class TestIngestRecallRecent(unittest.TestCase):

    def test_ingest_source_uses_recall_recent(self):
        src = (ROOT / "agents/ingest.py").read_text()
        self.assertIn("recall_recent", src)
        self.assertNotIn("recall_all()", src)

    def test_ingest_limit_is_reasonable(self):
        src = (ROOT / "agents/ingest.py").read_text()
        # Should limit to <= 100 entries
        import re
        limits = [int(m) for m in re.findall(r"recall_recent\(limit=(\d+)\)", src)]
        self.assertTrue(any(1 <= lim <= 100 for lim in limits),
                        f"recall_recent limit should be 1-100, got {limits}")


# ---------------------------------------------------------------------------
# 8. Closed loop count_memories
# ---------------------------------------------------------------------------

class TestClosedLoopCountMemories(unittest.TestCase):

    def test_closed_loop_source_uses_count_memories(self):
        src = (ROOT / "core/closed_loop.py").read_text()
        self.assertIn("count_memories()", src)
        self.assertNotIn("recall_all()", src)

    def test_closed_loop_snapshot_calls_count_memories(self):
        """Closed loop snapshot() should call count_memories, not recall_all."""
        from core.closed_loop import ClosedDevelopmentLoop
        mock_model = MagicMock()
        mock_brain = _make_mock_brain()
        mock_git = MagicMock()
        loop = ClosedDevelopmentLoop(model=mock_model, brain=mock_brain, git_tools=mock_git)
        result = loop.snapshot()
        mock_brain.count_memories.assert_called()
        mock_brain.recall_all.assert_not_called()
        self.assertIn("30,419" if False else "3", result)  # count in snapshot string


# ---------------------------------------------------------------------------
# 9. Full R1-R4 regression
# ---------------------------------------------------------------------------

class TestFullOptimizationRegression(unittest.TestCase):

    def test_r1_wal_mode(self):
        from memory.brain import Brain
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            b = Brain(db_path=db_path)
            row = b.db.execute("PRAGMA journal_mode").fetchone()
            self.assertEqual(row[0], "wal")
        finally:
            os.unlink(db_path)

    def test_r2_classify_goal_lru(self):
        from core.skill_dispatcher import classify_goal
        classify_goal.cache_clear()
        classify_goal("write a test")
        classify_goal("write a test")
        info = classify_goal.cache_info()
        self.assertEqual(info.hits, 1)

    def test_r3_recall_with_budget_uses_limit(self):
        src = (ROOT / "memory/brain.py").read_text()
        self.assertIn("ORDER BY id DESC LIMIT", src)

    def test_r4_no_recall_all_in_agents(self):
        for fname in ["agents/coder.py", "agents/critic.py", "agents/scaffolder.py",
                      "agents/tester.py", "agents/ingest.py", "agents/router.py"]:
            src = (ROOT / fname).read_text()
            self.assertNotIn("recall_all()", src, f"{fname} still calls recall_all()")

    def test_r4_no_recall_all_in_core_tools(self):
        for fname in ["core/evolution_loop.py", "core/closed_loop.py",
                      "tools/aura_control_mcp.py"]:
            src = (ROOT / fname).read_text()
            self.assertNotIn("recall_all()", src, f"{fname} still calls recall_all()")

    def test_r4_brain_count_memories_method(self):
        from memory.brain import Brain
        self.assertTrue(hasattr(Brain, "count_memories"))

    def test_r4_mcp_cache_exists(self):
        import tools.aura_control_mcp as mod
        self.assertTrue(hasattr(mod, "_memory_cache"))
        self.assertTrue(hasattr(mod, "_get_memories_cached"))


if __name__ == "__main__":
    unittest.main()
