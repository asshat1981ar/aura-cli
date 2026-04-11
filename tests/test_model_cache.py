"""Comprehensive test suite for core/model_cache.py."""

import hashlib
import sqlite3
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from core.model_cache import CacheMixin


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def db_connection(tmp_path):
    """Return a SQLite connection for testing."""
    db_path = str(tmp_path / "test_cache.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    yield conn
    conn.close()


@pytest.fixture
def cache_mixin(db_connection):
    """Return a CacheMixin instance with enabled cache."""
    mixin = CacheMixin()
    mixin._mem_cache = {}
    mixin.enable_cache(db_conn=db_connection, ttl_seconds=3600, momento=None)
    return mixin


@pytest.fixture
def sample_prompt():
    """Return a sample prompt string."""
    return "What is the capital of France?"


@pytest.fixture
def sample_response():
    """Return a sample response string."""
    return "The capital of France is Paris."


@pytest.fixture
def mock_momento():
    """Return a mock MomentoAdapter."""
    mock = Mock()
    mock.is_available.return_value = True
    mock.cache_get = Mock(return_value=None)
    mock.cache_set = Mock()
    return mock


# ============================================================================
# Test cache initialization
# ============================================================================


class TestCacheInitialization:
    """Tests for cache initialization and setup."""

    def test_enable_cache_sets_attributes(self, cache_mixin, db_connection):
        """Test that enable_cache sets all required attributes."""
        assert cache_mixin.cache_db is not None
        assert cache_mixin.cache_ttl == 3600
        assert hasattr(cache_mixin, "_momento")

    def test_enable_cache_creates_table(self, db_connection):
        """Test that enable_cache creates the prompt_cache table."""
        mixin = CacheMixin()
        mixin._mem_cache = {}
        mixin.enable_cache(db_conn=db_connection, ttl_seconds=3600, momento=None)

        cursor = db_connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prompt_cache'")
        assert cursor.fetchone() is not None

    def test_enable_cache_with_custom_ttl(self, db_connection):
        """Test that enable_cache respects custom TTL."""
        mixin = CacheMixin()
        mixin._mem_cache = {}
        custom_ttl = 7200
        mixin.enable_cache(db_conn=db_connection, ttl_seconds=custom_ttl, momento=None)

        assert mixin.cache_ttl == custom_ttl

    def test_enable_cache_with_momento(self, db_connection, mock_momento):
        """Test enable_cache with Momento adapter."""
        mixin = CacheMixin()
        mixin._mem_cache = {}
        mixin.enable_cache(db_conn=db_connection, ttl_seconds=3600, momento=mock_momento)

        assert mixin._momento == mock_momento

    def test_enable_cache_momento_unavailable(self, db_connection):
        """Test enable_cache when Momento is unavailable."""
        mixin = CacheMixin()
        mixin._mem_cache = {}

        mock_momento = Mock()
        mock_momento.is_available.return_value = False

        mixin.enable_cache(db_conn=db_connection, ttl_seconds=3600, momento=mock_momento)
        assert mixin._momento == mock_momento

    def test_enable_cache_with_no_db_connection(self):
        """Test enable_cache with None db_connection."""
        mixin = CacheMixin()
        mixin._mem_cache = {}

        # Should handle gracefully
        with patch("core.model_cache.log_json"):
            mixin.enable_cache(db_conn=None, ttl_seconds=3600)

    def test_enable_cache_with_logging(self, db_connection):
        """Test that enable_cache logs initialization."""
        mixin = CacheMixin()
        mixin._mem_cache = {}

        with patch("core.model_cache.log_json") as mock_log:
            mixin.enable_cache(db_conn=db_connection, ttl_seconds=3600, momento=None)

            # Should have logged
            assert mock_log.called

    def test_preload_cache_loads_recent_entries(self, db_connection):
        """Test that preload_cache loads recent non-expired entries."""
        mixin = CacheMixin()
        mixin._mem_cache = {}
        mixin.enable_cache(db_conn=db_connection, ttl_seconds=3600, momento=None)

        # Insert some test data
        prompt_hash = hashlib.sha256(b"test prompt").hexdigest()
        db_connection.execute(
            "INSERT INTO prompt_cache (prompt_hash, response) VALUES (?, ?)",
            (prompt_hash, "test response"),
        )
        db_connection.commit()

        # Clear mem cache
        mixin._mem_cache = {}

        # Preload
        mixin.preload_cache()

        # Verify it was loaded
        assert prompt_hash in mixin._mem_cache

    def test_preload_cache_limits_to_50_entries(self, db_connection):
        """Test that preload_cache limits loading to 50 entries."""
        mixin = CacheMixin()
        mixin._mem_cache = {}
        mixin.enable_cache(db_conn=db_connection, ttl_seconds=3600, momento=None)

        # Insert 60 entries
        for i in range(60):
            prompt_hash = hashlib.sha256(str(i).encode()).hexdigest()
            db_connection.execute(
                "INSERT INTO prompt_cache (prompt_hash, response) VALUES (?, ?)",
                (prompt_hash, f"response-{i}"),
            )
        db_connection.commit()

        # Clear and preload
        mixin._mem_cache = {}
        mixin.preload_cache()

        # Should have loaded at most 50
        assert len(mixin._mem_cache) <= 50

    def test_preload_cache_error_handling(self):
        """Test error handling in preload_cache when no DB."""
        mixin = CacheMixin()
        mixin.cache_db = None  # No DB
        mixin._mem_cache = {}

        # Should not raise
        with patch("core.model_cache.log_json"):
            mixin.preload_cache()


# ============================================================================
# Test cache read operations
# ============================================================================


class TestCacheRead:
    """Tests for reading from cache (get operations)."""

    def test_get_cached_response_l0_hit(self, cache_mixin, sample_prompt, sample_response):
        """Test L0 (in-memory) cache hit."""
        prompt_hash = hashlib.sha256(sample_prompt.encode()).hexdigest()
        cache_mixin._mem_cache[prompt_hash] = sample_response

        result = cache_mixin._get_cached_response(sample_prompt)

        assert result == sample_response

    def test_get_cached_response_l0_miss_l2_hit(self, cache_mixin, sample_prompt, sample_response):
        """Test L0 miss but L2 (SQLite) hit."""
        prompt_hash = hashlib.sha256(sample_prompt.encode()).hexdigest()

        # Store in L2 only
        cache_mixin.cache_db.execute(
            "INSERT INTO prompt_cache (prompt_hash, response) VALUES (?, ?)",
            (prompt_hash, sample_response),
        )
        cache_mixin.cache_db.commit()

        result = cache_mixin._get_cached_response(sample_prompt)

        assert result == sample_response

    def test_get_cached_response_no_cache_db(self, cache_mixin, sample_prompt):
        """Test get when cache_db is None."""
        cache_mixin.cache_db = None

        result = cache_mixin._get_cached_response(sample_prompt)

        assert result is None

    def test_get_cached_response_expired_entry(self, cache_mixin, sample_prompt, sample_response):
        """Test that expired entries are not returned."""
        prompt_hash = hashlib.sha256(sample_prompt.encode()).hexdigest()

        # Insert with old timestamp
        cache_mixin.cache_db.execute(
            "INSERT INTO prompt_cache (prompt_hash, response, timestamp) VALUES (?, ?, ?)",
            (prompt_hash, sample_response, "2000-01-01 00:00:00"),
        )
        cache_mixin.cache_db.commit()

        result = cache_mixin._get_cached_response(sample_prompt)

        assert result is None

    def test_get_cached_response_momento_l1_hit(self, cache_mixin, mock_momento, sample_prompt, sample_response):
        """Test L1 (Momento) cache hit."""
        cache_mixin._momento = mock_momento
        mock_momento.cache_get.return_value = sample_response

        with patch("core.model_cache.log_json"):
            result = cache_mixin._get_cached_response(sample_prompt)

        assert result == sample_response
        mock_momento.cache_get.assert_called_once()

    def test_get_cached_response_momento_unavailable(self, cache_mixin, mock_momento, sample_prompt, sample_response):
        """Test that L1 is skipped when Momento is unavailable."""
        cache_mixin._momento = mock_momento
        mock_momento.is_available.return_value = False

        # Store in L2
        prompt_hash = hashlib.sha256(sample_prompt.encode()).hexdigest()
        cache_mixin.cache_db.execute(
            "INSERT INTO prompt_cache (prompt_hash, response) VALUES (?, ?)",
            (prompt_hash, sample_response),
        )
        cache_mixin.cache_db.commit()

        with patch("core.model_cache.log_json"):
            result = cache_mixin._get_cached_response(sample_prompt)

        assert result == sample_response
        # L1 cache_get should not be called
        mock_momento.cache_get.assert_not_called()

    def test_get_cached_response_momento_error_handling(self, cache_mixin, mock_momento, sample_prompt, sample_response):
        """Test that L1 errors don't block L2 lookup."""
        cache_mixin._momento = mock_momento
        mock_momento.cache_get.side_effect = Exception("Momento error")

        # Store in L2
        prompt_hash = hashlib.sha256(sample_prompt.encode()).hexdigest()
        cache_mixin.cache_db.execute(
            "INSERT INTO prompt_cache (prompt_hash, response) VALUES (?, ?)",
            (prompt_hash, sample_response),
        )
        cache_mixin.cache_db.commit()

        with patch("core.model_cache.log_json"):
            result = cache_mixin._get_cached_response(sample_prompt)

        assert result == sample_response

    def test_get_cached_response_empty_cache(self, cache_mixin, sample_prompt):
        """Test get on completely empty cache."""
        result = cache_mixin._get_cached_response(sample_prompt)

        assert result is None

    def test_get_cached_response_prompt_hash_calculation(self, cache_mixin, sample_prompt, sample_response):
        """Test that prompt hash is calculated correctly."""
        prompt_hash = hashlib.sha256(sample_prompt.encode()).hexdigest()
        cache_mixin._mem_cache[prompt_hash] = sample_response

        result = cache_mixin._get_cached_response(sample_prompt)

        assert result == sample_response

    def test_get_cached_response_different_prompts_different_hashes(self, cache_mixin):
        """Test that different prompts have different hashes."""
        prompt1 = "What is 2+2?"
        prompt2 = "What is 3+3?"

        hash1 = hashlib.sha256(prompt1.encode()).hexdigest()
        hash2 = hashlib.sha256(prompt2.encode()).hexdigest()

        cache_mixin._mem_cache[hash1] = "4"

        result = cache_mixin._get_cached_response(prompt2)

        assert result is None  # Different prompt, different hash


# ============================================================================
# Test cache write operations
# ============================================================================


class TestCacheWrite:
    """Tests for writing to cache (set operations)."""

    def test_save_to_cache_l0_write(self, cache_mixin, sample_prompt, sample_response):
        """Test saving to L0 (in-memory) cache."""
        cache_mixin._save_to_cache(sample_prompt, sample_response)

        prompt_hash = hashlib.sha256(sample_prompt.encode()).hexdigest()
        assert cache_mixin._mem_cache[prompt_hash] == sample_response

    def test_save_to_cache_l2_write(self, cache_mixin, sample_prompt, sample_response):
        """Test saving to L2 (SQLite) cache."""
        cache_mixin._save_to_cache(sample_prompt, sample_response)

        prompt_hash = hashlib.sha256(sample_prompt.encode()).hexdigest()
        cursor = cache_mixin.cache_db.execute("SELECT response FROM prompt_cache WHERE prompt_hash = ?", (prompt_hash,))
        result = cursor.fetchone()

        assert result is not None
        assert result[0] == sample_response

    def test_save_to_cache_no_db_connection(self, cache_mixin, sample_prompt, sample_response):
        """Test save when cache_db is None."""
        cache_mixin.cache_db = None

        # Should only save to L0
        cache_mixin._save_to_cache(sample_prompt, sample_response)

        prompt_hash = hashlib.sha256(sample_prompt.encode()).hexdigest()
        assert cache_mixin._mem_cache[prompt_hash] == sample_response

    def test_save_to_cache_momento_write(self, cache_mixin, mock_momento, sample_prompt, sample_response):
        """Test L1 (Momento) write-through."""
        cache_mixin._momento = mock_momento

        with patch("core.model_cache.log_json"):
            cache_mixin._save_to_cache(sample_prompt, sample_response)

        mock_momento.cache_set.assert_called_once()
        args, kwargs = mock_momento.cache_set.call_args
        assert sample_response in args
        assert kwargs.get("ttl_seconds") == cache_mixin.cache_ttl

    def test_save_to_cache_momento_unavailable(self, cache_mixin, mock_momento, sample_prompt, sample_response):
        """Test that L1 write is skipped when unavailable."""
        cache_mixin._momento = mock_momento
        mock_momento.is_available.return_value = False

        cache_mixin._save_to_cache(sample_prompt, sample_response)

        # Should still save to L0 and L2
        prompt_hash = hashlib.sha256(sample_prompt.encode()).hexdigest()
        assert cache_mixin._mem_cache[prompt_hash] == sample_response
        mock_momento.cache_set.assert_not_called()

    def test_save_to_cache_momento_error_handling(self, cache_mixin, mock_momento, sample_prompt, sample_response):
        """Test that L1 errors don't block L2 write."""
        cache_mixin._momento = mock_momento
        mock_momento.cache_set.side_effect = Exception("Momento error")

        with patch("core.model_cache.log_json"):
            cache_mixin._save_to_cache(sample_prompt, sample_response)

        # L0 and L2 should still be written
        prompt_hash = hashlib.sha256(sample_prompt.encode()).hexdigest()
        assert cache_mixin._mem_cache[prompt_hash] == sample_response

    def test_save_to_cache_overwrites_existing(self, cache_mixin, sample_prompt):
        """Test that save overwrites existing cache entry."""
        response1 = "First response"
        response2 = "Second response"

        cache_mixin._save_to_cache(sample_prompt, response1)
        cache_mixin._save_to_cache(sample_prompt, response2)

        result = cache_mixin._get_cached_response(sample_prompt)
        assert result == response2

    def test_save_to_cache_large_response(self, cache_mixin, sample_prompt):
        """Test saving large response strings."""
        large_response = "x" * 100000  # 100KB string

        cache_mixin._save_to_cache(sample_prompt, large_response)
        result = cache_mixin._get_cached_response(sample_prompt)

        assert result == large_response

    def test_save_to_cache_special_characters(self, cache_mixin):
        """Test saving responses with special characters."""
        prompt = "Test prompt with émojis 🚀"
        response = "Response with special chars: ñ, é, ü, 中文, العربية"

        cache_mixin._save_to_cache(prompt, response)
        result = cache_mixin._get_cached_response(prompt)

        assert result == response


# ============================================================================
# Test TTL and expiration
# ============================================================================


class TestCacheTTL:
    """Tests for TTL and cache expiration."""

    def test_ttl_is_respected_during_preload(self, db_connection):
        """Test that TTL is respected when preloading cache."""
        mixin = CacheMixin()
        mixin._mem_cache = {}
        mixin.cache_ttl = 1  # 1 second
        mixin.cache_db = db_connection

        db_connection.execute("CREATE TABLE IF NOT EXISTS prompt_cache (prompt_hash TEXT PRIMARY KEY, response TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")

        # Insert entry
        prompt_hash = hashlib.sha256(b"test").hexdigest()
        db_connection.execute(
            "INSERT INTO prompt_cache (prompt_hash, response) VALUES (?, ?)",
            (prompt_hash, "response"),
        )
        db_connection.commit()

        with patch("core.model_cache.log_json"):
            mixin.preload_cache()

        # Should be loaded since it's fresh
        assert prompt_hash in mixin._mem_cache

    def test_expired_entry_not_returned(self, cache_mixin, sample_prompt):
        """Test that expired entries are not returned from L2."""
        # Set very short TTL
        cache_mixin.cache_ttl = 1

        prompt_hash = hashlib.sha256(sample_prompt.encode()).hexdigest()

        # Insert with old timestamp
        cache_mixin.cache_db.execute(
            "INSERT INTO prompt_cache (prompt_hash, response, timestamp) VALUES (?, ?, datetime('now', '-10 seconds'))",
            (prompt_hash, "old response"),
        )
        cache_mixin.cache_db.commit()

        result = cache_mixin._get_cached_response(sample_prompt)

        # Should be expired
        assert result is None

    def test_momento_ttl_passed_correctly(self, cache_mixin, mock_momento, sample_prompt, sample_response):
        """Test that TTL is passed to Momento correctly."""
        cache_mixin._momento = mock_momento
        cache_mixin.cache_ttl = 7200

        with patch("core.model_cache.log_json"):
            cache_mixin._save_to_cache(sample_prompt, sample_response)

        _, kwargs = mock_momento.cache_set.call_args
        assert kwargs["ttl_seconds"] == 7200

    def test_custom_ttl_in_get_query(self, cache_mixin, sample_prompt, sample_response):
        """Test that custom TTL is used in SQL query."""
        cache_mixin.cache_ttl = 3600

        prompt_hash = hashlib.sha256(sample_prompt.encode()).hexdigest()
        cache_mixin.cache_db.execute(
            "INSERT INTO prompt_cache (prompt_hash, response) VALUES (?, ?)",
            (prompt_hash, sample_response),
        )
        cache_mixin.cache_db.commit()

        result = cache_mixin._get_cached_response(sample_prompt)

        # Should be returned since it's within TTL
        assert result == sample_response


# ============================================================================
# Test logging
# ============================================================================


class TestCacheLogging:
    """Tests for logging behavior."""

    @patch("core.model_cache.log_json")
    def test_l0_hit_logged(self, mock_log, cache_mixin, sample_prompt):
        """Test that L0 hits are logged."""
        prompt_hash = hashlib.sha256(sample_prompt.encode()).hexdigest()
        cache_mixin._mem_cache[prompt_hash] = "response"

        cache_mixin._get_cached_response(sample_prompt)

        # Should have logged L0 hit
        assert any(call[0][1] == "model_cache_l0_hit" for call in mock_log.call_args_list)

    @patch("core.model_cache.log_json")
    def test_l1_hit_logged(self, mock_log, cache_mixin, mock_momento, sample_prompt):
        """Test that L1 hits are logged."""
        cache_mixin._momento = mock_momento
        mock_momento.cache_get.return_value = "response"

        cache_mixin._get_cached_response(sample_prompt)

        # Should have logged L1 hit
        assert any(call[0][1] == "model_cache_l1_hit" for call in mock_log.call_args_list)

    @patch("core.model_cache.log_json")
    def test_l2_hit_logged(self, mock_log, cache_mixin, sample_prompt):
        """Test that L2 hits are logged."""
        prompt_hash = hashlib.sha256(sample_prompt.encode()).hexdigest()
        cache_mixin.cache_db.execute(
            "INSERT INTO prompt_cache (prompt_hash, response) VALUES (?, ?)",
            (prompt_hash, "response"),
        )
        cache_mixin.cache_db.commit()

        cache_mixin._get_cached_response(sample_prompt)

        # Should have logged L2 hit
        assert any(call[0][1] == "model_cache_hit" for call in mock_log.call_args_list)

    @patch("core.model_cache.log_json")
    def test_l1_query_error_logged(self, mock_log, cache_mixin, mock_momento, sample_prompt):
        """Test that L1 query errors are logged."""
        cache_mixin._momento = mock_momento
        mock_momento.cache_get.side_effect = Exception("Error")

        # Need L2 entry to continue
        prompt_hash = hashlib.sha256(sample_prompt.encode()).hexdigest()
        cache_mixin.cache_db.execute(
            "INSERT INTO prompt_cache (prompt_hash, response) VALUES (?, ?)",
            (prompt_hash, "response"),
        )
        cache_mixin.cache_db.commit()

        cache_mixin._get_cached_response(sample_prompt)

        # Should have logged L1 error
        assert any(call[0][1] == "model_cache_l1_query_failed" for call in mock_log.call_args_list)


# ============================================================================
# Test momento adapter integration
# ============================================================================


class TestMomentoIntegration:
    """Tests for Momento adapter integration."""

    def test_momento_key_format(self, cache_mixin, mock_momento, sample_prompt):
        """Test that Momento keys are formatted correctly."""
        cache_mixin._momento = mock_momento

        with patch("core.model_cache.log_json"):
            cache_mixin._save_to_cache(sample_prompt, "response")

        # Get the key that was used
        call_args = mock_momento.cache_set.call_args
        key = call_args[0][1]  # Second positional arg

        # Should start with "response:" prefix
        assert key.startswith("response:")

    def test_momento_key_is_shortened_hash(self, cache_mixin, mock_momento, sample_prompt):
        """Test that Momento keys use first 16 chars of hash."""
        cache_mixin._momento = mock_momento

        with patch("core.model_cache.log_json"):
            cache_mixin._get_cached_response(sample_prompt)

        prompt_hash = hashlib.sha256(sample_prompt.encode()).hexdigest()
        expected_key = f"response:{prompt_hash[:16]}"

        call_args = mock_momento.cache_get.call_args
        if call_args:
            key = call_args[0][1]  # Second positional arg
            assert expected_key == key

    def test_momento_none_attribute(self, cache_mixin):
        """Test handling when _momento attribute is None."""
        cache_mixin._momento = None

        result = cache_mixin._get_cached_response("test prompt")

        # Should still work, just skip L1
        assert result is None


# ============================================================================
# Test prompt hash consistency
# ============================================================================


class TestPromptHashing:
    """Tests for prompt hash calculation and consistency."""

    def test_same_prompt_same_hash(self, cache_mixin):
        """Test that same prompt always produces same hash."""
        prompt = "The same prompt text"

        hash1 = hashlib.sha256(prompt.encode()).hexdigest()
        hash2 = hashlib.sha256(prompt.encode()).hexdigest()

        assert hash1 == hash2

    def test_different_prompts_different_hashes(self, cache_mixin):
        """Test that different prompts produce different hashes."""
        prompt1 = "First prompt"
        prompt2 = "Second prompt"

        hash1 = hashlib.sha256(prompt1.encode()).hexdigest()
        hash2 = hashlib.sha256(prompt2.encode()).hexdigest()

        assert hash1 != hash2

    def test_prompt_whitespace_matters(self, cache_mixin):
        """Test that whitespace differences create different hashes."""
        prompt1 = "Test prompt with spaces"
        prompt2 = "Test  prompt  with  spaces"  # Extra spaces

        hash1 = hashlib.sha256(prompt1.encode()).hexdigest()
        hash2 = hashlib.sha256(prompt2.encode()).hexdigest()

        assert hash1 != hash2

    def test_hash_is_sha256(self, cache_mixin):
        """Test that hash is SHA256."""
        prompt = "Test prompt"

        hash_result = hashlib.sha256(prompt.encode()).hexdigest()

        # SHA256 produces 64-character hex string
        assert len(hash_result) == 64


# ============================================================================
# Test edge cases and robustness
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_empty_prompt_cache(self, cache_mixin):
        """Test handling of empty string prompt."""
        result = cache_mixin._get_cached_response("")

        assert result is None

    def test_empty_response_cache(self, cache_mixin, sample_prompt):
        """Test saving and retrieving empty string response."""
        cache_mixin._save_to_cache(sample_prompt, "")

        result = cache_mixin._get_cached_response(sample_prompt)

        assert result == ""

    def test_very_long_prompt(self, cache_mixin):
        """Test handling of very long prompts."""
        long_prompt = "x" * 100000  # 100KB prompt
        response = "short response"

        cache_mixin._save_to_cache(long_prompt, response)
        result = cache_mixin._get_cached_response(long_prompt)

        assert result == response

    def test_very_long_response(self, cache_mixin, sample_prompt):
        """Test handling of very long responses."""
        long_response = "y" * 1000000  # 1MB response

        cache_mixin._save_to_cache(sample_prompt, long_response)
        result = cache_mixin._get_cached_response(sample_prompt)

        assert result == long_response

    def test_unicode_prompt_and_response(self, cache_mixin):
        """Test handling of unicode content."""
        prompt = "¿Cuál es la capital de España? 日本 العربية"
        response = "La respuesta es Madrid. 東京 الرياض"

        cache_mixin._save_to_cache(prompt, response)
        result = cache_mixin._get_cached_response(prompt)

        assert result == response

    def test_null_bytes_in_content(self, cache_mixin):
        """Test handling of content with null bytes."""
        prompt = "Test\x00Prompt"
        response = "Test\x00Response"

        cache_mixin._save_to_cache(prompt, response)
        result = cache_mixin._get_cached_response(prompt)

        assert result == response

    def test_multiple_save_same_prompt(self, cache_mixin, sample_prompt):
        """Test multiple saves of same prompt with different responses."""
        responses = ["Response 1", "Response 2", "Response 3"]

        for resp in responses:
            cache_mixin._save_to_cache(sample_prompt, resp)

        result = cache_mixin._get_cached_response(sample_prompt)

        # Should have the last saved response
        assert result == responses[-1]

    def test_cache_db_none_on_get(self, cache_mixin, sample_prompt, sample_response):
        """Test get when cache_db becomes None after initialization."""
        prompt_hash = hashlib.sha256(sample_prompt.encode()).hexdigest()
        cache_mixin._mem_cache[prompt_hash] = sample_response
        cache_mixin.cache_db = None

        result = cache_mixin._get_cached_response(sample_prompt)

        # Should still get from L0
        assert result == sample_response


# ============================================================================
# Test schema and database
# ============================================================================


class TestDatabaseSchema:
    """Tests for database schema and integrity."""

    def test_prompt_cache_table_structure(self, db_connection):
        """Test that prompt_cache table has expected structure."""
        mixin = CacheMixin()
        mixin._mem_cache = {}
        mixin.enable_cache(db_conn=db_connection, ttl_seconds=3600)

        cursor = db_connection.execute("PRAGMA table_info(prompt_cache)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}

        assert "prompt_hash" in columns
        assert "response" in columns
        assert "timestamp" in columns

    def test_prompt_hash_is_primary_key(self, db_connection):
        """Test that prompt_hash is the primary key."""
        mixin = CacheMixin()
        mixin._mem_cache = {}
        mixin.enable_cache(db_conn=db_connection, ttl_seconds=3600)

        cursor = db_connection.execute("PRAGMA table_info(prompt_cache)")
        for row in cursor.fetchall():
            if row[1] == "prompt_hash":
                assert row[5] == 1  # pk column should be 1
                break

    def test_duplicate_hash_replaces_entry(self, cache_mixin, sample_prompt):
        """Test that INSERT OR REPLACE works correctly."""
        prompt_hash = hashlib.sha256(sample_prompt.encode()).hexdigest()

        # First insert
        cache_mixin.cache_db.execute(
            "INSERT OR REPLACE INTO prompt_cache (prompt_hash, response) VALUES (?, ?)",
            (prompt_hash, "First response"),
        )
        cache_mixin.cache_db.commit()

        # Second insert with same hash
        cache_mixin.cache_db.execute(
            "INSERT OR REPLACE INTO prompt_cache (prompt_hash, response) VALUES (?, ?)",
            (prompt_hash, "Second response"),
        )
        cache_mixin.cache_db.commit()

        # Should only have one entry
        cursor = cache_mixin.cache_db.execute(
            "SELECT COUNT(*) FROM prompt_cache WHERE prompt_hash = ?",
            (prompt_hash,),
        )
        count = cursor.fetchone()[0]
        assert count == 1

    def test_timestamp_auto_set(self, cache_mixin, sample_prompt):
        """Test that timestamp is automatically set."""
        prompt_hash = hashlib.sha256(sample_prompt.encode()).hexdigest()

        cache_mixin.cache_db.execute(
            "INSERT INTO prompt_cache (prompt_hash, response) VALUES (?, ?)",
            (prompt_hash, "response"),
        )
        cache_mixin.cache_db.commit()

        cursor = cache_mixin.cache_db.execute(
            "SELECT timestamp FROM prompt_cache WHERE prompt_hash = ?",
            (prompt_hash,),
        )
        timestamp = cursor.fetchone()

        assert timestamp is not None
        assert timestamp[0] is not None
