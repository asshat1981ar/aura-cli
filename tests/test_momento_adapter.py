"""
Tests for MomentoAdapter and the L1/L2 fallback pattern.

All tests run without a real Momento API key (no MOMENTO_API_KEY env var),
so the adapter is always in fallback/no-op mode.  This verifies:
  - is_available() returns False when no key
  - cache_get/set/delete/list operations return safe defaults
  - publish returns False
  - Circuit breaker functionality
  - Lock operations in fallback mode
  - MomentoBrain behaves identically to Brain in fallback mode
  - MomentoMemoryStore behaves identically to MemoryStore in fallback mode
  - SkillWeightAdapter works with momento=None
  - ModelAdapter.enable_cache works with momento=None
"""

import os
import sys
import tempfile
import unittest
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch


def setup_momento_mocks():
    """Inject mock Momento SDK into sys.modules to avoid import errors."""
    if "momento" not in sys.modules:
        momento_mock = MagicMock()
        sys.modules["momento"] = momento_mock
        sys.modules["momento.responses"] = MagicMock()
        sys.modules["momento.requests"] = MagicMock()


setup_momento_mocks()


class TestCircuitBreaker(unittest.TestCase):
    """Test the _CircuitBreaker class."""

    def test_circuit_breaker_init(self):
        from memory.momento_adapter import _CircuitBreaker
        cb = _CircuitBreaker()
        self.assertEqual(cb._state, "closed")
        self.assertEqual(cb._failures, 0)

    def test_circuit_breaker_allow_closed(self):
        from memory.momento_adapter import _CircuitBreaker
        cb = _CircuitBreaker()
        self.assertTrue(cb.allow_request())
        self.assertTrue(cb.allow_request())

    def test_circuit_breaker_record_success(self):
        from memory.momento_adapter import _CircuitBreaker
        cb = _CircuitBreaker()
        cb._failures = 3
        cb.record_success()
        self.assertEqual(cb._failures, 0)
        self.assertEqual(cb._state, "closed")

    def test_circuit_breaker_opens_after_threshold(self):
        from memory.momento_adapter import _CircuitBreaker
        cb = _CircuitBreaker()
        for _ in range(cb.THRESHOLD):
            cb.record_failure()
        self.assertEqual(cb._state, "open")
        self.assertFalse(cb.allow_request())

    def test_circuit_breaker_half_open_after_reset(self):
        from memory.momento_adapter import _CircuitBreaker
        cb = _CircuitBreaker()
        for _ in range(cb.THRESHOLD):
            cb.record_failure()
        cb._opened_at = time.monotonic() - cb.RESET_SECONDS - 1
        result = cb.allow_request()
        self.assertTrue(result)
        self.assertEqual(cb._state, "half_open")

    def test_circuit_breaker_half_open_success_closes(self):
        from memory.momento_adapter import _CircuitBreaker
        cb = _CircuitBreaker()
        for _ in range(cb.THRESHOLD):
            cb.record_failure()
        cb._opened_at = time.monotonic() - cb.RESET_SECONDS - 1
        cb.allow_request()
        cb.record_success()
        self.assertEqual(cb._state, "closed")

    def test_circuit_breaker_half_open_failure_reopens(self):
        from memory.momento_adapter import _CircuitBreaker
        cb = _CircuitBreaker()
        for _ in range(cb.THRESHOLD):
            cb.record_failure()
        cb._opened_at = time.monotonic() - cb.RESET_SECONDS - 1
        cb.allow_request()
        cb.record_failure()
        self.assertEqual(cb._state, "open")

    def test_circuit_breaker_thread_safe(self):
        from memory.momento_adapter import _CircuitBreaker
        cb = _CircuitBreaker()
        def record_failures():
            for _ in range(3):
                cb.record_failure()
        threads = [threading.Thread(target=record_failures) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertGreaterEqual(cb._failures, 5)


class TestMomentoAdapterFallback(unittest.TestCase):
    """Adapter is a no-op when MOMENTO_API_KEY is absent."""

    def setUp(self):
        # Remove API key env var if set
        self._orig = os.environ.pop("MOMENTO_API_KEY", None)

    def tearDown(self):
        if self._orig is not None:
            os.environ["MOMENTO_API_KEY"] = self._orig

    def _make_adapter(self):
        # Re-import to pick up cleared env var
        from memory.momento_adapter import MomentoAdapter

        return MomentoAdapter()

    def test_is_available_false_without_key(self):
        adapter = self._make_adapter()
        self.assertFalse(adapter.is_available())

    def test_cache_get_returns_none(self):
        adapter = self._make_adapter()
        self.assertIsNone(adapter.cache_get("cache", "key"))

    def test_cache_set_returns_false(self):
        adapter = self._make_adapter()
        self.assertFalse(adapter.cache_set("cache", "key", "value"))

    def test_cache_delete_returns_false(self):
        adapter = self._make_adapter()
        self.assertFalse(adapter.cache_delete("cache", "key"))

    def test_list_push_returns_false(self):
        adapter = self._make_adapter()
        self.assertFalse(adapter.list_push("cache", "key", "value"))

    def test_list_range_returns_empty(self):
        adapter = self._make_adapter()
        self.assertEqual(adapter.list_range("cache", "key"), [])

    def test_publish_returns_false(self):
        adapter = self._make_adapter()
        self.assertFalse(adapter.publish("topic", "message"))

    def test_acquire_lock_returns_true_in_fallback(self):
        """In fallback mode (no Momento), acquire_lock returns True (local mode)."""
        adapter = self._make_adapter()
        self.assertTrue(adapter.acquire_lock("test-lock"))

    def test_release_lock_returns_true_in_fallback(self):
        """In fallback mode, release_lock returns True."""
        adapter = self._make_adapter()
        self.assertTrue(adapter.release_lock("test-lock"))

    def test_list_range_with_params(self):
        """list_range with start/end params still returns empty list."""
        adapter = self._make_adapter()
        result = adapter.list_range("cache", "key", start=0, end=10)
        self.assertEqual(result, [])

    def test_cache_set_with_ttl(self):
        """cache_set with ttl_seconds still returns False."""
        adapter = self._make_adapter()
        result = adapter.cache_set("cache", "key", "value", ttl_seconds=120)
        self.assertFalse(result)

    def test_list_push_with_max_size(self):
        """list_push with max_size still returns False."""
        adapter = self._make_adapter()
        result = adapter.list_push("cache", "key", "value", ttl_seconds=60, max_size=10)
        self.assertFalse(result)

    def test_cache_constants_exist(self):
        """Verify cache name constants are defined."""
        from memory.momento_adapter import (
            WORKING_MEMORY_CACHE,
            EPISODIC_MEMORY_CACHE,
            TOPIC_CYCLE_COMPLETE,
            TOPIC_WEAKNESS_FOUND,
            TOPIC_GOAL_QUEUED,
        )
        self.assertEqual(WORKING_MEMORY_CACHE, "aura-working-memory")
        self.assertEqual(EPISODIC_MEMORY_CACHE, "aura-episodic-memory")
        self.assertTrue(TOPIC_CYCLE_COMPLETE.startswith("aura."))
        self.assertTrue(TOPIC_WEAKNESS_FOUND.startswith("aura."))
        self.assertTrue(TOPIC_GOAL_QUEUED.startswith("aura."))


class TestMomentoBrainFallback(unittest.TestCase):
    """MomentoBrain behaves identically to Brain when adapter is no-op."""

    def setUp(self):
        os.environ.pop("MOMENTO_API_KEY", None)
        os.environ["AURA_SKIP_CHDIR"] = "1"

    def test_remember_and_recall(self):
        from memory.momento_adapter import MomentoAdapter
        from memory.momento_brain import MomentoBrain

        adapter = MomentoAdapter()
        brain = MomentoBrain(adapter)
        brain.remember("hello momento")
        recalled = brain.recall_all()
        self.assertIn("hello momento", recalled)

    def test_add_and_recall_weakness(self):
        from memory.momento_adapter import MomentoAdapter
        from memory.momento_brain import MomentoBrain

        adapter = MomentoAdapter()
        brain = MomentoBrain(adapter)
        brain.add_weakness("test weakness from unit test")
        weaknesses = brain.recall_weaknesses()
        self.assertTrue(any("test weakness" in w for w in weaknesses))

    def test_remember_dict(self):
        from memory.momento_adapter import MomentoAdapter
        from memory.momento_brain import MomentoBrain

        adapter = MomentoAdapter()
        brain = MomentoBrain(adapter)
        brain.remember({"type": "test", "value": 42})
        recalled = brain.recall_all()
        self.assertTrue(any("test" in str(r) for r in recalled))


class TestMomentoMemoryStoreFallback(unittest.TestCase):
    """MomentoMemoryStore behaves identically to MemoryStore in fallback mode."""

    def setUp(self):
        os.environ.pop("MOMENTO_API_KEY", None)
        os.environ["AURA_SKIP_CHDIR"] = "1"
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_put_and_query(self):
        from memory.momento_adapter import MomentoAdapter
        from memory.momento_memory_store import MomentoMemoryStore

        adapter = MomentoAdapter()
        store = MomentoMemoryStore(Path(self._tmpdir.name), adapter)
        store.put("test_tier", {"key": "value", "n": 1})
        results = store.query("test_tier")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["key"], "value")

    def test_append_log_and_read_log(self):
        from memory.momento_adapter import MomentoAdapter
        from memory.momento_memory_store import MomentoMemoryStore

        adapter = MomentoAdapter()
        store = MomentoMemoryStore(Path(self._tmpdir.name), adapter)
        entry = {"cycle_id": "test123", "goal_type": "test", "phase_outputs": {}}
        store.append_log(entry)
        log = store.read_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["cycle_id"], "test123")

    def test_query_returns_last_n(self):
        from memory.momento_adapter import MomentoAdapter
        from memory.momento_memory_store import MomentoMemoryStore

        adapter = MomentoAdapter()
        store = MomentoMemoryStore(Path(self._tmpdir.name), adapter)
        for i in range(10):
            store.put("summaries", {"i": i})
        results = store.query("summaries", limit=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[-1]["i"], 9)


class TestSkillWeightAdapterWithMomento(unittest.TestCase):
    """SkillWeightAdapter works correctly with momento=None (fallback)."""

    def setUp(self):
        os.environ.pop("MOMENTO_API_KEY", None)
        os.environ["AURA_SKIP_CHDIR"] = "1"
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_load_and_save_without_momento(self):
        from core.skill_weight_adapter import SkillWeightAdapter

        adapter = SkillWeightAdapter(memory_root=self._tmpdir.name, momento=None)
        # Simulate a cycle update
        entry = {
            "goal_type": "bug_fix",
            "phase_outputs": {
                "skill_context": {
                    "security_scanner": {"findings": ["issue1"], "critical_count": 1},
                }
            },
        }
        adapter.on_cycle_complete(entry)
        weights = adapter.get_weights_summary()
        self.assertIn("bug_fix", weights)
        self.assertIn("security_scanner", weights["bug_fix"])

    def test_ranked_skills_fallback(self):
        from core.skill_weight_adapter import SkillWeightAdapter

        adapter = SkillWeightAdapter(memory_root=self._tmpdir.name, momento=None)
        ranked = adapter.ranked_skills("default")
        self.assertIsInstance(ranked, list)
        self.assertGreater(len(ranked), 0)


class TestModelAdapterCacheWithMomento(unittest.TestCase):
    """ModelAdapter.enable_cache accepts momento=None without error."""

    def test_enable_cache_no_momento(self):
        import sqlite3
        from core.model_adapter import ModelAdapter

        adapter = ModelAdapter()
        db = sqlite3.connect(":memory:", check_same_thread=False)
        # Should not raise even with momento=None
        adapter.enable_cache(db, ttl_seconds=60, momento=None)
        self.assertIs(adapter.cache_db, db)

    def test_get_cached_response_miss(self):
        import sqlite3
        from core.model_adapter import ModelAdapter

        adapter = ModelAdapter()
        db = sqlite3.connect(":memory:", check_same_thread=False)
        adapter.enable_cache(db, ttl_seconds=60, momento=None)
        result = adapter._get_cached_response("test prompt")
        self.assertIsNone(result)

    def test_save_and_get_cached_response(self):
        import sqlite3
        from core.model_adapter import ModelAdapter

        adapter = ModelAdapter()
        db = sqlite3.connect(":memory:", check_same_thread=False)
        adapter.enable_cache(db, ttl_seconds=3600, momento=None)
        adapter._save_to_cache("test prompt", "test response")
        result = adapter._get_cached_response("test prompt")
        self.assertEqual(result, "test response")


if __name__ == "__main__":
    unittest.main()
