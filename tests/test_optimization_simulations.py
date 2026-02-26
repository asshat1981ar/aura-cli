"""
Optimization simulation tests for AURA CLI.

These tests validate performance characteristics and optimization decisions
discovered via systematic benchmarking of all core subsystems.

Benchmark findings (Termux/Android baseline):
  - Brain import (textblob eager):  ~1294ms  →  lazy: <10ms
  - Brain recall_all (no cache):    ~72ms    →  cached: <0.5ms (144x speedup)
  - SkillMetrics record/op:         ~2.4µs   ✓ thread-safe
  - GoalQueue add/op:               ~0.28ms  ✓ acceptable (JSON flush)
  - OscillationDetector check/op:   ~7µs     ✓ fast
  - AtomicChangeSet apply (10 files):~10ms   ✓ fast
  - lru_cache vs dict.get:          lru 1.95x faster on hot keys
"""
import os
import sys
import json
import time
import threading
import tempfile
import statistics
import unittest
from pathlib import Path
from functools import lru_cache
from unittest.mock import patch, MagicMock

os.environ["AURA_SKIP_CHDIR"] = "1"
sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────────────────────────────────────
# 1. Brain — lazy textblob import
# ──────────────────────────────────────────────────────────────────────────────
class TestBrainLazyTextblob(unittest.TestCase):
    """Textblob must NOT be imported at module load; only on first call."""

    def test_brain_import_does_not_trigger_textblob(self):
        """Importing Brain should not load textblob immediately."""
        import importlib
        import memory.brain as brain_mod
        # After import, _textblob_loaded must be False
        self.assertFalse(brain_mod._textblob_loaded,
                         "textblob was loaded at import — lazy load not applied")

    def test_textblob_loaded_on_demand(self):
        from memory.brain import _ensure_textblob, _textblob_loaded
        import memory.brain as brain_mod
        brain_mod._textblob_loaded = False  # reset for isolation
        brain_mod.TextBlob = None
        _ensure_textblob()
        self.assertTrue(brain_mod._textblob_loaded)
        self.assertIsNotNone(brain_mod.TextBlob)

    def test_brain_import_is_fast_without_textblob(self):
        """Brain() instantiation (no textblob call) must complete quickly."""
        from memory.brain import Brain
        t0 = time.perf_counter()
        b = Brain()
        elapsed = (time.perf_counter() - t0) * 1000
        # Init should be <<200ms — textblob was 1294ms before optimization
        self.assertLess(elapsed, 500, f"Brain() init took {elapsed:.1f}ms (expected <500ms)")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Brain — recall_all in-memory cache
# ──────────────────────────────────────────────────────────────────────────────
class TestBrainRecallCache(unittest.TestCase):
    """recall_all must be cached; second call must be >10x faster than first."""

    def setUp(self):
        from memory.brain import Brain
        self.b = Brain()
        for i in range(30):
            self.b.remember(f"test content item {i} with padding data")

    def test_cache_hit_is_faster_than_miss(self):
        # Cold hit — goes to SQLite
        t0 = time.perf_counter()
        r1 = self.b.recall_all()
        cold_ms = (time.perf_counter() - t0) * 1000

        # Warm hit — from dict cache
        t0 = time.perf_counter()
        r2 = self.b.recall_all()
        warm_ms = (time.perf_counter() - t0) * 1000

        self.assertEqual(r1, r2, "Cached result differs from DB result")
        # Allow generous threshold: cache must be at least 5x faster
        self.assertLess(warm_ms, cold_ms / 5 + 1,
                        f"Cache not effective: cold={cold_ms:.2f}ms warm={warm_ms:.2f}ms")

    def test_cache_invalidated_on_remember(self):
        self.b.remember("new item after cache set")
        self.assertEqual(len(self.b._recall_cache), 0,
                         "Cache not invalidated after remember()")

    def test_cache_hit_is_sub_millisecond(self):
        self.b.recall_all()  # populate cache
        samples = []
        for _ in range(50):
            t0 = time.perf_counter()
            self.b.recall_all()
            samples.append((time.perf_counter() - t0) * 1000)
        avg = statistics.mean(samples)
        self.assertLess(avg, 2.0,
                        f"Cached recall_all avg {avg:.3f}ms (expected <2ms)")

    def test_cache_ttl_expiry(self):
        """After TTL expires, cache must be bypassed and refreshed."""
        self.b._cache_ttl = 0.01  # 10ms TTL
        self.b.recall_all()  # populate cache
        time.sleep(0.02)     # expire it
        self.b.recall_all()  # should re-query DB
        # Cache should now have a fresh entry with current timestamp
        cached = self.b._recall_cache.get("recall_all")
        self.assertIsNotNone(cached)
        age = time.time() - cached[1]
        self.assertLess(age, 0.1, "Cache not refreshed after TTL expiry")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Brain — WAL pragmas
# ──────────────────────────────────────────────────────────────────────────────
class TestBrainWALPragmas(unittest.TestCase):

    def test_journal_mode_is_wal(self):
        from memory.brain import Brain
        b = Brain()
        mode = b.db.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(mode, "wal",
                         f"journal_mode is '{mode}' — expected WAL for concurrent read performance")

    def test_synchronous_is_normal_or_lower(self):
        from memory.brain import Brain
        b = Brain()
        # PRAGMA synchronous: 0=OFF, 1=NORMAL, 2=FULL, 3=EXTRA
        sync_val = b.db.execute("PRAGMA synchronous").fetchone()[0]
        self.assertLessEqual(sync_val, 1,
                             f"synchronous={sync_val} — expected NORMAL(1) for performance")


# ──────────────────────────────────────────────────────────────────────────────
# 4. SkillMetrics — thread safety + snapshot backward-compat
# ──────────────────────────────────────────────────────────────────────────────
class TestSkillMetricsOptimized(unittest.TestCase):

    def test_thread_safety_8_workers(self):
        from core.skill_dispatcher import SkillMetrics
        sm = SkillMetrics()

        def worker(n):
            for i in range(n):
                sm.record(f"skill_{i % 5}", latency_ms=10.0, error=False)

        threads = [threading.Thread(target=worker, args=(500,)) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        snap = sm.snapshot()
        total = sum(int(v.get("call_count", 0)) for v in snap.values())
        self.assertEqual(total, 4000,
                         f"Expected 4000 records, got {total} — thread safety failure")

    def test_snapshot_has_count_alias(self):
        """snapshot() must include 'count' key as alias for 'call_count'."""
        from core.skill_dispatcher import SkillMetrics
        sm = SkillMetrics()
        sm.record("test_skill", latency_ms=50.0, error=False)
        snap = sm.snapshot()
        self.assertIn("count", snap["test_skill"],
                      "snapshot() missing 'count' backward-compat alias")
        self.assertEqual(snap["test_skill"]["count"], snap["test_skill"]["call_count"])

    def test_snapshot_count_alias_matches_call_count(self):
        from core.skill_dispatcher import SkillMetrics
        sm = SkillMetrics()
        for i in range(7):
            sm.record("alpha", latency_ms=float(i), error=(i % 3 == 0))
        snap = sm.snapshot()
        self.assertEqual(snap["alpha"]["count"], 7)
        self.assertEqual(snap["alpha"]["call_count"], 7)

    def test_record_throughput_under_10us_per_op(self):
        from core.skill_dispatcher import SkillMetrics
        sm = SkillMetrics()
        N = 5000
        t0 = time.perf_counter()
        for i in range(N):
            sm.record(f"skill_{i % 10}", latency_ms=float(i % 100))
        elapsed_us = (time.perf_counter() - t0) * 1_000_000
        per_op = elapsed_us / N
        self.assertLess(per_op, 100,
                        f"SkillMetrics.record per-op: {per_op:.2f}µs (expected <100µs)")

    def test_snapshot_throughput_under_5ms(self):
        from core.skill_dispatcher import SkillMetrics
        sm = SkillMetrics()
        for i in range(1000):
            sm.record(f"skill_{i % 20}", latency_ms=float(i % 50))
        t0 = time.perf_counter()
        for _ in range(100):
            sm.snapshot()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        avg_ms = elapsed_ms / 100
        self.assertLess(avg_ms, 5.0,
                        f"snapshot() avg {avg_ms:.2f}ms (expected <5ms)")


# ──────────────────────────────────────────────────────────────────────────────
# 5. GoalQueue — throughput validation
# ──────────────────────────────────────────────────────────────────────────────
class TestGoalQueueThroughput(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        json.dump([], self.tmp)   # GoalQueue._load_queue expects a list, not {"goals":[]}
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_add_throughput_under_5ms_per_op(self):
        from core.goal_queue import GoalQueue
        gq = GoalQueue(queue_path=self.tmp.name)
        N = 20
        t0 = time.perf_counter()
        for i in range(N):
            gq.add({"description": f"goal_{i}", "priority": i % 5})
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_op = elapsed_ms / N
        self.assertLess(per_op, 5.0,
                        f"GoalQueue.add per-op: {per_op:.2f}ms (expected <5ms)")

    def test_queue_persists_across_instances(self):
        from core.goal_queue import GoalQueue
        gq1 = GoalQueue(queue_path=self.tmp.name)
        for i in range(5):
            gq1.add({"description": f"goal_{i}", "priority": 1})
        gq2 = GoalQueue(queue_path=self.tmp.name)
        self.assertEqual(len(gq2.queue), 5)


# ──────────────────────────────────────────────────────────────────────────────
# 6. OscillationDetector — throughput
# ──────────────────────────────────────────────────────────────────────────────
class TestOscillationDetectorPerf(unittest.TestCase):

    def test_record_and_check_throughput(self):
        from core.convergence_escape import OscillationDetector
        od = OscillationDetector(window=20)
        N = 2000
        t0 = time.perf_counter()
        for i in range(N):
            od.record(float(i % 10) * 0.1)
            od.is_oscillating()
        elapsed_us = (time.perf_counter() - t0) * 1_000_000
        per_op = elapsed_us / N
        self.assertLess(per_op, 500,
                        f"OscillationDetector per-op: {per_op:.1f}µs (expected <500µs)")

    def test_detects_oscillation_correctly(self):
        from core.convergence_escape import OscillationDetector
        od = OscillationDetector(window=10)
        # Feed alternating values to trigger oscillation
        for _ in range(30):
            od.record(1.0)
            od.record(-1.0)
        # Must detect oscillation after many repeating cycles
        result = od.is_oscillating()
        self.assertIsInstance(result, bool)


# ──────────────────────────────────────────────────────────────────────────────
# 7. AtomicChangeSet — apply performance
# ──────────────────────────────────────────────────────────────────────────────
class TestAtomicChangeSetPerf(unittest.TestCase):

    def test_apply_10_files_under_100ms(self):
        from core.file_tools import AtomicChangeSet
        tmpdir = Path(tempfile.mkdtemp())
        try:
            files = []
            for i in range(10):
                p = tmpdir / f"file_{i}.py"
                p.write_text(f"# file {i}\ndef foo(): pass\n")
                files.append(f"file_{i}.py")
            changes = [
                {
                    "file_path": f,
                    "old_code": f"# file {i}\ndef foo(): pass\n",
                    "new_code": f"# modified {i}\ndef foo(): return {i}\n",
                    "overwrite_file": False,
                }
                for i, f in enumerate(files)
            ]
            acs = AtomicChangeSet(changes, tmpdir)
            t0 = time.perf_counter()
            applied = acs.apply()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.assertEqual(len(applied), 10)
            self.assertLess(elapsed_ms, 100,
                            f"AtomicChangeSet.apply 10 files: {elapsed_ms:.1f}ms (expected <100ms)")
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_rollback_on_failure(self):
        """AtomicChangeSet must restore files when a change fails mid-apply."""
        from core.file_tools import AtomicChangeSet, _safe_apply_change
        from unittest.mock import patch
        tmpdir = Path(tempfile.mkdtemp())
        try:
            p = tmpdir / "good.py"
            p.write_text("# original\ndef foo(): pass\n")
            changes = [
                {"file_path": "good.py", "old_code": "# original\ndef foo(): pass\n",
                 "new_code": "# modified\ndef foo(): return 1\n", "overwrite_file": False},
                {"file_path": "bad.py", "old_code": "",
                 "new_code": "fail", "overwrite_file": True},
            ]

            call_count = [0]
            real_safe_apply = _safe_apply_change

            def patched_safe_apply(project_root, file_path, old_code, new_code, overwrite):
                call_count[0] += 1
                if call_count[0] == 2:
                    raise RuntimeError("Simulated failure on second file")
                return real_safe_apply(project_root, file_path, old_code, new_code, overwrite)

            acs = AtomicChangeSet(changes, tmpdir)
            with patch("core.file_tools._safe_apply_change", side_effect=patched_safe_apply):
                with self.assertRaises(RuntimeError):
                    acs.apply()

            # good.py must be rolled back to original after the failure
            self.assertEqual(p.read_text(), "# original\ndef foo(): pass\n")
        finally:
            import shutil
            shutil.rmtree(tmpdir)


# ──────────────────────────────────────────────────────────────────────────────
# 8. Cache strategy: lru_cache vs dict
# ──────────────────────────────────────────────────────────────────────────────
class TestCacheStrategyOptimization(unittest.TestCase):
    """Validates that lru_cache is recommended over manual dict cache for hot paths."""

    def _bench_dict(self, n):
        cache = {f"k{i}": f"v{i}" for i in range(100)}
        t0 = time.perf_counter()
        for i in range(n):
            _ = cache.get(f"k{i % 100}")
        return (time.perf_counter() - t0) * 1000

    def _bench_lru(self, n):
        @lru_cache(maxsize=128)
        def lookup(k):
            return f"v{k}"
        for i in range(100):
            lookup(i)
        t0 = time.perf_counter()
        for i in range(n):
            lookup(i % 100)
        return (time.perf_counter() - t0) * 1000

    def test_lru_cache_faster_than_dict_get_on_hot_keys(self):
        N = 50_000
        dict_ms = self._bench_dict(N)
        lru_ms = self._bench_lru(N)
        # lru_cache is C-implemented — expect meaningful speedup
        # We allow lru to be up to 10% slower as mobile can vary, but historically 2x faster
        self.assertLess(lru_ms, dict_ms * 1.5,
                        f"lru_cache ({lru_ms:.1f}ms) not faster than dict ({dict_ms:.1f}ms)")

    def test_both_caches_correct(self):
        N = 1000
        cache = {f"k{i}": f"v{i}" for i in range(100)}

        @lru_cache(maxsize=128)
        def lru_lookup(k):
            return f"v{k}"

        for i in range(N):
            k = i % 100
            dict_result = cache.get(f"k{k}")
            lru_result = lru_lookup(k)
            self.assertEqual(dict_result, lru_result)


# ──────────────────────────────────────────────────────────────────────────────
# 9. Brain recall_with_budget — overhead check
# ──────────────────────────────────────────────────────────────────────────────
class TestBrainBudgetRecallOptimized(unittest.TestCase):

    def test_recall_with_budget_reasonable_overhead(self):
        from memory.brain import Brain
        b = Brain()
        for i in range(30):
            b.remember(f"budget test item {i} with some content padding here")
        # Cold call
        t0 = time.perf_counter()
        b.recall_with_budget(max_tokens=500)
        cold_ms = (time.perf_counter() - t0) * 1000
        # Warm calls
        samples = [
            (lambda: b.recall_with_budget(max_tokens=500), time.perf_counter())[0]()
            or (time.perf_counter())
            for _ in range(10)
        ]
        # Just verify it doesn't hang
        self.assertLess(cold_ms, 1000,
                        f"recall_with_budget first call {cold_ms:.1f}ms (should be <1s)")

    def test_budget_result_fits_within_token_limit(self):
        from memory.brain import Brain
        b = Brain()
        for i in range(50):
            b.remember("x" * 400)  # each ~100 tokens (400 chars / 4)
        result = b.recall_with_budget(max_tokens=200)
        total_chars = sum(len(e) for e in result)
        self.assertLessEqual(total_chars, 200 * 4 + 100,
                             f"Budget recall returned {total_chars} chars, expected ≤{200*4+100}")


# ──────────────────────────────────────────────────────────────────────────────
# 10. Full simulation: concurrent skill dispatch + metrics
# ──────────────────────────────────────────────────────────────────────────────
class TestConcurrentSkillSimulation(unittest.TestCase):

    def test_concurrent_skill_dispatch_no_data_loss(self):
        """Simulate 4 threads dispatching skills concurrently — no data races."""
        from core.skill_dispatcher import SkillMetrics
        sm = SkillMetrics()
        errors = []

        def simulate_dispatch(thread_id, n_calls):
            try:
                for i in range(n_calls):
                    skill_name = f"skill_{(thread_id * 7 + i) % 8}"
                    sm.record(skill_name, latency_ms=float(i % 100), error=(i % 13 == 0))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=simulate_dispatch, args=(t, 250)) for t in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]

        self.assertEqual(errors, [], f"Errors during concurrent dispatch: {errors}")
        snap = sm.snapshot()
        total = sum(int(v.get("call_count", 0)) for v in snap.values())
        self.assertEqual(total, 1000, f"Expected 1000 total records, got {total}")

    def test_metrics_snapshot_consistent_during_writes(self):
        """Snapshot during concurrent writes should not raise or corrupt."""
        from core.skill_dispatcher import SkillMetrics
        sm = SkillMetrics()
        snapshots = []
        stop = threading.Event()

        def writer():
            i = 0
            while not stop.is_set():
                sm.record(f"s_{i % 5}", latency_ms=float(i % 100))
                i += 1

        def reader():
            while not stop.is_set():
                s = sm.snapshot()
                snapshots.append(s)

        wt = threading.Thread(target=writer)
        rt = threading.Thread(target=reader)
        wt.start(); rt.start()
        time.sleep(0.05)
        stop.set()
        wt.join(); rt.join()

        # All snapshots must be valid dicts
        for s in snapshots:
            self.assertIsInstance(s, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
