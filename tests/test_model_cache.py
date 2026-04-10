"""Tests for core/model_cache.py — CacheMixin."""

import hashlib
import sqlite3
import unittest
from unittest.mock import MagicMock, patch

from core.model_cache import CacheMixin


def _make_cache_mixin():
    """Create a CacheMixin instance with required attributes."""
    obj = CacheMixin()
    obj.cache_db = None
    obj.cache_ttl = 3600
    obj._mem_cache = {}
    obj._momento = None
    return obj


class TestEnableCache(unittest.TestCase):
    """Tests for enable_cache."""

    def test_creates_table_and_preloads(self):
        mixin = _make_cache_mixin()
        conn = sqlite3.connect(":memory:")
        mixin.enable_cache(conn, ttl_seconds=1800)
        self.assertEqual(mixin.cache_db, conn)
        self.assertEqual(mixin.cache_ttl, 1800)
        # Verify the table exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prompt_cache'")
        self.assertIsNotNone(cursor.fetchone())
        conn.close()

    def test_enable_cache_with_momento(self):
        mixin = _make_cache_mixin()
        conn = sqlite3.connect(":memory:")
        mock_momento = MagicMock()
        mock_momento.is_available.return_value = True
        mixin.enable_cache(conn, momento=mock_momento)
        self.assertEqual(mixin._momento, mock_momento)
        conn.close()

    def test_enable_cache_handles_db_error(self):
        mixin = _make_cache_mixin()
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError("disk full")
        # Should not raise
        mixin.enable_cache(mock_conn)


class TestPreloadCache(unittest.TestCase):
    """Tests for preload_cache."""

    def test_preload_populates_mem_cache(self):
        mixin = _make_cache_mixin()
        conn = sqlite3.connect(":memory:")
        mixin.cache_db = conn
        mixin.cache_ttl = 3600
        conn.execute("""
            CREATE TABLE prompt_cache (
                prompt_hash TEXT PRIMARY KEY,
                response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("INSERT INTO prompt_cache (prompt_hash, response) VALUES (?, ?)", ("hash1", "resp1"))
        conn.execute("INSERT INTO prompt_cache (prompt_hash, response) VALUES (?, ?)", ("hash2", "resp2"))
        conn.commit()

        mixin.preload_cache()
        self.assertEqual(mixin._mem_cache["hash1"], "resp1")
        self.assertEqual(mixin._mem_cache["hash2"], "resp2")
        conn.close()

    def test_preload_no_db(self):
        mixin = _make_cache_mixin()
        mixin.cache_db = None
        # Should not raise
        mixin.preload_cache()
        self.assertEqual(mixin._mem_cache, {})

    def test_preload_handles_error(self):
        mixin = _make_cache_mixin()
        mock_db = MagicMock()
        mock_db.execute.side_effect = sqlite3.OperationalError("error")
        mixin.cache_db = mock_db
        # Should not raise
        mixin.preload_cache()


class TestGetCachedResponse(unittest.TestCase):
    """Tests for _get_cached_response."""

    def test_l0_hit(self):
        mixin = _make_cache_mixin()
        prompt = "hello world"
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        mixin._mem_cache[prompt_hash] = "cached_response"

        result = mixin._get_cached_response(prompt)
        self.assertEqual(result, "cached_response")

    def test_l2_hit(self):
        mixin = _make_cache_mixin()
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE prompt_cache (
                prompt_hash TEXT PRIMARY KEY,
                response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        prompt = "test prompt"
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        conn.execute("INSERT INTO prompt_cache (prompt_hash, response) VALUES (?, ?)", (prompt_hash, "db_response"))
        conn.commit()
        mixin.cache_db = conn
        mixin.cache_ttl = 3600

        result = mixin._get_cached_response(prompt)
        self.assertEqual(result, "db_response")
        conn.close()

    def test_cache_miss(self):
        mixin = _make_cache_mixin()
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE prompt_cache (
                prompt_hash TEXT PRIMARY KEY,
                response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        mixin.cache_db = conn
        mixin.cache_ttl = 3600

        result = mixin._get_cached_response("nonexistent")
        self.assertIsNone(result)
        conn.close()

    def test_no_db_returns_none(self):
        mixin = _make_cache_mixin()
        result = mixin._get_cached_response("anything")
        self.assertIsNone(result)

    @patch("core.model_cache.log_json")
    def test_l1_momento_hit(self, mock_log):
        mixin = _make_cache_mixin()
        mock_momento = MagicMock()
        mock_momento.is_available.return_value = True
        mock_momento.cache_get.return_value = "momento_response"
        mixin._momento = mock_momento

        with patch("core.model_cache.CacheMixin._get_cached_response.__module__", "core.model_cache"):
            result = mixin._get_cached_response("test")
        self.assertEqual(result, "momento_response")

    def test_l1_momento_unavailable_falls_through(self):
        mixin = _make_cache_mixin()
        mock_momento = MagicMock()
        mock_momento.is_available.return_value = False
        mixin._momento = mock_momento
        # No DB either, so should return None
        result = mixin._get_cached_response("test")
        self.assertIsNone(result)

    def test_l2_query_error_returns_none(self):
        mixin = _make_cache_mixin()
        mock_db = MagicMock()
        mock_db.execute.side_effect = sqlite3.OperationalError("broken")
        mixin.cache_db = mock_db
        result = mixin._get_cached_response("test")
        self.assertIsNone(result)


class TestSaveToCache(unittest.TestCase):
    """Tests for _save_to_cache."""

    def test_saves_to_l0_and_l2(self):
        mixin = _make_cache_mixin()
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE prompt_cache (
                prompt_hash TEXT PRIMARY KEY,
                response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        mixin.cache_db = conn

        prompt = "save this"
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        mixin._save_to_cache(prompt, "saved_response")

        # L0
        self.assertEqual(mixin._mem_cache[prompt_hash], "saved_response")
        # L2
        cursor = conn.execute("SELECT response FROM prompt_cache WHERE prompt_hash = ?", (prompt_hash,))
        self.assertEqual(cursor.fetchone()[0], "saved_response")
        conn.close()

    def test_saves_to_l0_only_when_no_db(self):
        mixin = _make_cache_mixin()
        prompt = "no db"
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        mixin._save_to_cache(prompt, "response")
        self.assertEqual(mixin._mem_cache[prompt_hash], "response")

    def test_saves_to_l1_momento(self):
        mixin = _make_cache_mixin()
        mock_momento = MagicMock()
        mock_momento.is_available.return_value = True
        mixin._momento = mock_momento

        with patch("core.model_cache.CacheMixin._save_to_cache.__module__", "core.model_cache"):
            mixin._save_to_cache("prompt", "response")
        mock_momento.cache_set.assert_called_once()

    def test_l2_save_error_no_raise(self):
        mixin = _make_cache_mixin()
        mock_db = MagicMock()
        mock_db.execute.side_effect = sqlite3.OperationalError("full")
        mixin.cache_db = mock_db
        # Should not raise
        mixin._save_to_cache("prompt", "response")

    def test_overwrite_existing(self):
        mixin = _make_cache_mixin()
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE prompt_cache (
                prompt_hash TEXT PRIMARY KEY,
                response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        mixin.cache_db = conn

        mixin._save_to_cache("prompt", "v1")
        mixin._save_to_cache("prompt", "v2")
        prompt_hash = hashlib.sha256("prompt".encode()).hexdigest()
        self.assertEqual(mixin._mem_cache[prompt_hash], "v2")
        cursor = conn.execute("SELECT response FROM prompt_cache WHERE prompt_hash = ?", (prompt_hash,))
        self.assertEqual(cursor.fetchone()[0], "v2")
        conn.close()


class TestCacheRoundTrip(unittest.TestCase):
    """Integration-style round-trip tests."""

    def test_save_then_get(self):
        mixin = _make_cache_mixin()
        conn = sqlite3.connect(":memory:")
        mixin.enable_cache(conn, ttl_seconds=3600)

        mixin._save_to_cache("question", "answer")
        result = mixin._get_cached_response("question")
        self.assertEqual(result, "answer")
        conn.close()

    def test_different_prompts_different_hashes(self):
        mixin = _make_cache_mixin()
        conn = sqlite3.connect(":memory:")
        mixin.enable_cache(conn)

        mixin._save_to_cache("prompt_a", "answer_a")
        mixin._save_to_cache("prompt_b", "answer_b")
        self.assertEqual(mixin._get_cached_response("prompt_a"), "answer_a")
        self.assertEqual(mixin._get_cached_response("prompt_b"), "answer_b")
        conn.close()


if __name__ == "__main__":
    unittest.main()
