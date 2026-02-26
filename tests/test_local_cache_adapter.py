"""
Tests for LocalCacheAdapter and cache_adapter_factory.

Run with::

    AURA_SKIP_CHDIR=1 python3 -m pytest tests/test_local_cache_adapter.py -v
"""
import json
import os
import tempfile
import threading
import time
import unittest

os.environ.setdefault("AURA_SKIP_CHDIR", "1")


def _make_adapter(tmp_dir=None):
    from memory.local_cache_adapter import LocalCacheAdapter
    db = os.path.join(tmp_dir or tempfile.mkdtemp(), "test_cache.db")
    return LocalCacheAdapter(db_path=db)


class TestLocalCacheAdapterBasics(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.adapter = _make_adapter(self.tmp)

    def test_is_always_available(self):
        self.assertTrue(self.adapter.is_available())

    def test_cache_set_and_get(self):
        ok = self.adapter.cache_set("c", "k", "hello")
        self.assertTrue(ok)
        self.assertEqual(self.adapter.cache_get("c", "k"), "hello")

    def test_cache_miss_returns_none(self):
        self.assertIsNone(self.adapter.cache_get("c", "nonexistent"))

    def test_cache_delete(self):
        self.adapter.cache_set("c", "k", "v")
        self.adapter.cache_delete("c", "k")
        self.assertIsNone(self.adapter.cache_get("c", "k"))

    def test_cache_overwrite(self):
        self.adapter.cache_set("c", "k", "v1")
        self.adapter.cache_set("c", "k", "v2")
        self.assertEqual(self.adapter.cache_get("c", "k"), "v2")

    def test_different_caches_are_isolated(self):
        self.adapter.cache_set("cache-a", "key", "alpha")
        self.adapter.cache_set("cache-b", "key", "beta")
        self.assertEqual(self.adapter.cache_get("cache-a", "key"), "alpha")
        self.assertEqual(self.adapter.cache_get("cache-b", "key"), "beta")


class TestLocalCacheAdapterTTL(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.adapter = _make_adapter(self.tmp)

    def test_ttl_expiry(self):
        self.adapter.cache_set("c", "k", "value", ttl_seconds=1)
        self.assertEqual(self.adapter.cache_get("c", "k"), "value")
        time.sleep(1.05)
        self.assertIsNone(self.adapter.cache_get("c", "k"))

    def test_no_ttl_persists(self):
        self.adapter.cache_set("c", "k", "permanent", ttl_seconds=0)
        time.sleep(0.05)
        self.assertEqual(self.adapter.cache_get("c", "k"), "permanent")

    def test_evict_expired(self):
        self.adapter.cache_set("c", "k1", "v1", ttl_seconds=1)
        self.adapter.cache_set("c", "k2", "v2", ttl_seconds=60)
        time.sleep(1.05)
        removed = self.adapter.evict_expired()
        self.assertEqual(removed, 1)
        stats = self.adapter.stats()
        self.assertEqual(stats["kv_expired_pending_eviction"], 0)


class TestLocalCacheAdapterLists(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.adapter = _make_adapter(self.tmp)

    def test_list_push_and_range(self):
        for i in range(5):
            self.adapter.list_push("c", "lst", str(i))
        result = self.adapter.list_range("c", "lst")
        self.assertEqual(result, ["0", "1", "2", "3", "4"])

    def test_list_range_slice(self):
        for i in range(10):
            self.adapter.list_push("c", "lst", str(i))
        self.assertEqual(self.adapter.list_range("c", "lst", start=2, end=4),
                         ["2", "3", "4"])

    def test_list_range_negative_end(self):
        for i in range(5):
            self.adapter.list_push("c", "lst", str(i))
        self.assertEqual(self.adapter.list_range("c", "lst", start=3, end=-1),
                         ["3", "4"])

    def test_list_max_size_trim(self):
        for i in range(10):
            self.adapter.list_push("c", "lst", str(i), max_size=5)
        result = self.adapter.list_range("c", "lst")
        self.assertEqual(len(result), 5)
        # Should keep the most recent 5
        self.assertEqual(result, ["5", "6", "7", "8", "9"])

    def test_empty_list_returns_empty(self):
        self.assertEqual(self.adapter.list_range("c", "missing"), [])

    def test_list_is_a_copy(self):
        self.adapter.list_push("c", "lst", "a")
        result = self.adapter.list_range("c", "lst")
        result.append("mutation")
        self.assertEqual(self.adapter.list_range("c", "lst"), ["a"])


class TestLocalCacheAdapterPubSub(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.adapter = _make_adapter(self.tmp)

    def test_publish_and_read_events(self):
        msg = json.dumps({"cycle_id": "test-1", "status": "pass"})
        ok = self.adapter.publish("aura.cycle_complete", msg)
        self.assertTrue(ok)
        events = self.adapter.read_events("aura.cycle_complete", since_ts=0.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["topic"], "aura.cycle_complete")
        self.assertEqual(json.loads(events[0]["message"])["cycle_id"], "test-1")

    def test_read_events_since_ts_filter(self):
        self.adapter.publish("t", "old")
        ts_after_first = time.time()
        time.sleep(0.01)
        self.adapter.publish("t", "new")
        events = self.adapter.read_events("t", since_ts=ts_after_first)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["message"], "new")

    def test_topic_isolation(self):
        self.adapter.publish("topic-a", "msg-a")
        self.adapter.publish("topic-b", "msg-b")
        a_events = self.adapter.read_events("topic-a")
        b_events = self.adapter.read_events("topic-b")
        self.assertEqual(len(a_events), 1)
        self.assertEqual(len(b_events), 1)
        self.assertEqual(a_events[0]["message"], "msg-a")


class TestLocalCacheAdapterThreadSafety(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.adapter = _make_adapter(self.tmp)

    def test_concurrent_set_get(self):
        errors = []

        def worker(idx):
            try:
                self.adapter.cache_set("c", f"k{idx}", f"v{idx}")
                val = self.adapter.cache_get("c", f"k{idx}")
                if val != f"v{idx}":
                    errors.append(f"expected v{idx} got {val}")
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [], f"Thread-safety errors: {errors}")

    def test_concurrent_list_push(self):
        def pusher():
            for _ in range(20):
                self.adapter.list_push("c", "shared", "x", max_size=200)

        threads = [threading.Thread(target=pusher) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        result = self.adapter.list_range("c", "shared")
        self.assertEqual(len(result), 100)


class TestLocalCacheAdapterStats(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.adapter = _make_adapter(self.tmp)

    def test_stats(self):
        self.adapter.cache_set("c", "k1", "v1")
        self.adapter.cache_set("c", "k2", "v2")
        self.adapter.list_push("c", "lst", "a")
        self.adapter.list_push("c", "lst", "b")
        s = self.adapter.stats()
        self.assertEqual(s["kv_live"], 2)
        self.assertEqual(s["list_keys"], 1)
        self.assertEqual(s["list_items"], 2)

    def test_flush_clears_state(self):
        self.adapter.cache_set("c", "k", "v")
        self.adapter.list_push("c", "lst", "x")
        self.adapter.flush()
        self.assertIsNone(self.adapter.cache_get("c", "k"))
        self.assertEqual(self.adapter.list_range("c", "lst"), [])


class TestCacheAdapterFactory(unittest.TestCase):

    def test_returns_local_without_api_key(self):
        env_backup = os.environ.pop("MOMENTO_API_KEY", None)
        try:
            from memory.cache_adapter_factory import create_cache_adapter
            from memory.local_cache_adapter import LocalCacheAdapter
            adapter = create_cache_adapter()
            self.assertIsInstance(adapter, LocalCacheAdapter)
        finally:
            if env_backup is not None:
                os.environ["MOMENTO_API_KEY"] = env_backup

    def test_returns_momento_with_api_key(self):
        os.environ["MOMENTO_API_KEY"] = "dummy-key-for-test"
        try:
            from memory.cache_adapter_factory import create_cache_adapter
            from memory.momento_adapter import MomentoAdapter
            adapter = create_cache_adapter()
            self.assertIsInstance(adapter, MomentoAdapter)
        finally:
            del os.environ["MOMENTO_API_KEY"]


if __name__ == "__main__":
    unittest.main()
