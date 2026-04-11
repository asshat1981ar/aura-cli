"""Tests for memory.redis_cache_adapter and the factory 'redis' type."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_adapter(**kwargs):
    """Return a RedisCacheAdapter with a clean slate (no cached client)."""
    # Re-import so module-level state doesn't bleed between tests.
    if "memory.redis_cache_adapter" in sys.modules:
        del sys.modules["memory.redis_cache_adapter"]
    from memory.redis_cache_adapter import RedisCacheAdapter  # noqa: PLC0415

    return RedisCacheAdapter(**kwargs)


# ---------------------------------------------------------------------------
# Test: Redis server unreachable  → get() returns None
# ---------------------------------------------------------------------------


def test_redis_unavailable_returns_none():
    """When redis.from_url raises ConnectionError, get() returns None."""
    mock_redis_module = MagicMock()
    mock_redis_module.from_url.side_effect = ConnectionError("Connection refused")

    with patch.dict(sys.modules, {"redis": mock_redis_module}):
        adapter = _fresh_adapter(url="redis://localhost:6379/0")
        result = adapter.get("some_key")

    assert result is None


# ---------------------------------------------------------------------------
# Test: redis package not installed  → get() returns None
# ---------------------------------------------------------------------------


def test_redis_not_installed_returns_none():
    """When 'import redis' raises ImportError, get() returns None."""
    # Remove redis from sys.modules so the import inside the adapter fails.
    with patch.dict(sys.modules, {"redis": None}):
        adapter = _fresh_adapter()
        result = adapter.get("some_key")

    assert result is None


# ---------------------------------------------------------------------------
# Test: set then get round-trip  → value recovered
# ---------------------------------------------------------------------------


def test_cache_set_get_roundtrip():
    """set() stores a value; get() returns the same value."""
    import json

    mock_client = MagicMock()
    # Simulate Redis SETEX / GET behaviour in-memory.
    _store: dict[str, str] = {}

    def fake_setex(key, ttl, val):
        _store[key] = val

    def fake_set(key, val):
        _store[key] = val

    def fake_get(key):
        raw = _store.get(key)
        return raw.encode() if isinstance(raw, str) else raw

    def fake_ping():
        return True

    mock_client.setex.side_effect = fake_setex
    mock_client.set.side_effect = fake_set
    mock_client.get.side_effect = fake_get
    mock_client.ping.side_effect = fake_ping

    mock_redis_module = MagicMock()
    mock_redis_module.from_url.return_value = mock_client

    with patch.dict(sys.modules, {"redis": mock_redis_module}):
        adapter = _fresh_adapter()
        adapter.set("my_key", {"hello": "world"}, ttl_seconds=60)
        result = adapter.get("my_key")

    assert result == {"hello": "world"}


# ---------------------------------------------------------------------------
# Test: delete removes the key
# ---------------------------------------------------------------------------


def test_cache_delete_removes_key():
    """delete() calls client.delete and subsequent get() returns None."""
    import json

    mock_client = MagicMock()
    _store: dict[str, str] = {}

    mock_client.ping.return_value = True
    mock_client.setex.side_effect = lambda k, ttl, v: _store.update({k: v})
    mock_client.get.side_effect = lambda k: _store.get(k)
    mock_client.delete.side_effect = lambda k: _store.pop(k, None)

    mock_redis_module = MagicMock()
    mock_redis_module.from_url.return_value = mock_client

    with patch.dict(sys.modules, {"redis": mock_redis_module}):
        adapter = _fresh_adapter()
        adapter.set("del_key", "to_delete", ttl_seconds=30)
        adapter.delete("del_key")
        result = adapter.get("del_key")

    assert result is None


# ---------------------------------------------------------------------------
# Test: clear flushes the database
# ---------------------------------------------------------------------------


def test_cache_clear_calls_flushdb():
    """clear() calls client.flushdb()."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True

    mock_redis_module = MagicMock()
    mock_redis_module.from_url.return_value = mock_client

    with patch.dict(sys.modules, {"redis": mock_redis_module}):
        adapter = _fresh_adapter()
        adapter.clear()

    mock_client.flushdb.assert_called_once()


# ---------------------------------------------------------------------------
# Test: factory creates RedisCacheAdapter for type "redis"
# ---------------------------------------------------------------------------


def test_factory_creates_redis_adapter():
    """get_adapter('redis') returns a RedisCacheAdapter instance."""
    if "memory.cache_adapter_factory" in sys.modules:
        del sys.modules["memory.cache_adapter_factory"]
    if "memory.redis_cache_adapter" in sys.modules:
        del sys.modules["memory.redis_cache_adapter"]

    from memory.cache_adapter_factory import get_adapter  # noqa: PLC0415
    from memory.redis_cache_adapter import RedisCacheAdapter  # noqa: PLC0415

    adapter = get_adapter("redis")
    assert isinstance(adapter, RedisCacheAdapter)


# ---------------------------------------------------------------------------
# Test: factory raises ValueError for unknown adapter type
# ---------------------------------------------------------------------------


def test_factory_raises_for_unknown_type():
    """get_adapter('unknown') raises ValueError."""
    if "memory.cache_adapter_factory" in sys.modules:
        del sys.modules["memory.cache_adapter_factory"]

    from memory.cache_adapter_factory import get_adapter  # noqa: PLC0415

    with pytest.raises(ValueError, match="Unknown adapter type"):
        get_adapter("unknown")


# ---------------------------------------------------------------------------
# Test: is_available() reflects Redis connection status
# ---------------------------------------------------------------------------


def test_is_available_when_redis_unavailable():
    """is_available() returns False when Redis is unavailable."""
    mock_redis_module = MagicMock()
    mock_redis_module.from_url.side_effect = ConnectionError("Connection refused")

    with patch.dict(sys.modules, {"redis": mock_redis_module}):
        adapter = _fresh_adapter(url="redis://localhost:6379/0")
        assert adapter.is_available() is False


def test_is_available_when_redis_available():
    """is_available() returns True when Redis is connected."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True

    mock_redis_module = MagicMock()
    mock_redis_module.from_url.return_value = mock_client

    with patch.dict(sys.modules, {"redis": mock_redis_module}):
        adapter = _fresh_adapter()
        assert adapter.is_available() is True


# ---------------------------------------------------------------------------
# Test: get() returns None on JSON decode error
# ---------------------------------------------------------------------------


def test_get_json_decode_error_returns_none():
    """get() returns None when Redis returns invalid JSON."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.get.return_value = b"not valid json {{"

    mock_redis_module = MagicMock()
    mock_redis_module.from_url.return_value = mock_client

    with patch.dict(sys.modules, {"redis": mock_redis_module}):
        adapter = _fresh_adapter()
        result = adapter.get("bad_json_key")

    assert result is None


# ---------------------------------------------------------------------------
# Test: set() with ttl_seconds > 0 uses SETEX
# ---------------------------------------------------------------------------


def test_set_with_ttl_uses_setex():
    """set() with ttl_seconds > 0 uses client.setex()."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True

    mock_redis_module = MagicMock()
    mock_redis_module.from_url.return_value = mock_client

    with patch.dict(sys.modules, {"redis": mock_redis_module}):
        adapter = _fresh_adapter()
        adapter.set("key_with_ttl", {"data": "value"}, ttl_seconds=300)

    mock_client.setex.assert_called_once()
    mock_client.set.assert_not_called()


# ---------------------------------------------------------------------------
# Test: set() with ttl_seconds == 0 uses SET (no expiry)
# ---------------------------------------------------------------------------


def test_set_with_no_ttl_uses_set():
    """set() with ttl_seconds == 0 uses client.set() without expiry."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True

    mock_redis_module = MagicMock()
    mock_redis_module.from_url.return_value = mock_client

    with patch.dict(sys.modules, {"redis": mock_redis_module}):
        adapter = _fresh_adapter()
        adapter.set("key_no_ttl", {"data": "value"}, ttl_seconds=0)

    mock_client.set.assert_called_once()
    mock_client.setex.assert_not_called()


# ---------------------------------------------------------------------------
# Test: set() error handling
# ---------------------------------------------------------------------------


def test_set_error_handling():
    """set() returns silently on Redis error."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.setex.side_effect = RuntimeError("Redis error")

    mock_redis_module = MagicMock()
    mock_redis_module.from_url.return_value = mock_client

    with patch.dict(sys.modules, {"redis": mock_redis_module}):
        adapter = _fresh_adapter()
        # Should not raise
        adapter.set("error_key", {"data": "value"}, ttl_seconds=60)


# ---------------------------------------------------------------------------
# Test: delete error handling
# ---------------------------------------------------------------------------


def test_delete_error_handling():
    """delete() returns silently on Redis error."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.delete.side_effect = RuntimeError("Redis error")

    mock_redis_module = MagicMock()
    mock_redis_module.from_url.return_value = mock_client

    with patch.dict(sys.modules, {"redis": mock_redis_module}):
        adapter = _fresh_adapter()
        # Should not raise
        adapter.delete("error_key")


# ---------------------------------------------------------------------------
# Test: clear error handling
# ---------------------------------------------------------------------------


def test_clear_error_handling():
    """clear() returns silently on Redis error."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.flushdb.side_effect = RuntimeError("Redis error")

    mock_redis_module = MagicMock()
    mock_redis_module.from_url.return_value = mock_client

    with patch.dict(sys.modules, {"redis": mock_redis_module}):
        adapter = _fresh_adapter()
        # Should not raise
        adapter.clear()


# ---------------------------------------------------------------------------
# Test: Multiple get/set with different data types
# ---------------------------------------------------------------------------


def test_set_get_various_data_types():
    """set/get round-trip various JSON-compatible types."""
    mock_client = MagicMock()
    _store: dict[str, str] = {}

    def fake_setex(key, ttl, val):
        _store[key] = val

    def fake_set(key, val):
        _store[key] = val

    def fake_get(key):
        raw = _store.get(key)
        return raw.encode() if isinstance(raw, str) else raw

    def fake_ping():
        return True

    mock_client.setex.side_effect = fake_setex
    mock_client.set.side_effect = fake_set
    mock_client.get.side_effect = fake_get
    mock_client.ping.side_effect = fake_ping

    mock_redis_module = MagicMock()
    mock_redis_module.from_url.return_value = mock_client

    with patch.dict(sys.modules, {"redis": mock_redis_module}):
        adapter = _fresh_adapter()

        # Test list
        adapter.set("list_key", [1, 2, 3])
        assert adapter.get("list_key") == [1, 2, 3]

        # Test dict with nested structures
        adapter.set("nested_key", {"a": {"b": {"c": 1}}})
        assert adapter.get("nested_key") == {"a": {"b": {"c": 1}}}

        # Test string
        adapter.set("string_key", "hello world")
        assert adapter.get("string_key") == "hello world"

        # Test number
        adapter.set("number_key", 42)
        assert adapter.get("number_key") == 42

        # Test null
        adapter.set("null_key", None)
        assert adapter.get("null_key") is None


# ---------------------------------------------------------------------------
# Test: _get_client lazy initialization
# ---------------------------------------------------------------------------


def test_get_client_lazy_initialization():
    """_get_client() only initializes once, reusing cached result."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True

    mock_redis_module = MagicMock()
    mock_redis_module.from_url.return_value = mock_client

    with patch.dict(sys.modules, {"redis": mock_redis_module}):
        adapter = _fresh_adapter()

        # First call initializes
        client1 = adapter._get_client()
        # Second call returns cached client
        client2 = adapter._get_client()

        assert client1 is client2
        # from_url should only be called once
        mock_redis_module.from_url.assert_called_once()


# ---------------------------------------------------------------------------
# Test: _get_client caches failure state
# ---------------------------------------------------------------------------


def test_get_client_caches_failure():
    """_get_client() caches failure and doesn't retry on subsequent calls."""
    mock_redis_module = MagicMock()
    mock_redis_module.from_url.side_effect = ConnectionError("Connection refused")

    with patch.dict(sys.modules, {"redis": mock_redis_module}):
        adapter = _fresh_adapter()

        # First call fails
        result1 = adapter._get_client()
        assert result1 is None

        # Second call returns None without retrying
        result2 = adapter._get_client()
        assert result2 is None

        # from_url should only be called once (not retried)
        assert mock_redis_module.from_url.call_count == 1


# ---------------------------------------------------------------------------
# Test: RedisCacheAdapter with custom URL
# ---------------------------------------------------------------------------


def test_adapter_with_custom_url():
    """RedisCacheAdapter accepts custom Redis URL."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True

    mock_redis_module = MagicMock()
    mock_redis_module.from_url.return_value = mock_client

    with patch.dict(sys.modules, {"redis": mock_redis_module}):
        adapter = _fresh_adapter(url="redis://custom:6380/1")
        adapter.is_available()

    # Verify custom URL was passed to from_url
    call_args = mock_redis_module.from_url.call_args
    assert call_args[0][0] == "redis://custom:6380/1"


# ---------------------------------------------------------------------------
# Test: RedisCacheAdapter defaults to REDIS_URL env var
# ---------------------------------------------------------------------------


def test_adapter_uses_redis_url_env_var():
    """RedisCacheAdapter uses REDIS_URL environment variable by default."""
    import os

    mock_client = MagicMock()
    mock_client.ping.return_value = True

    mock_redis_module = MagicMock()
    mock_redis_module.from_url.return_value = mock_client

    custom_url = "redis://env-host:6380/2"
    with patch.dict(os.environ, {"REDIS_URL": custom_url}):
        with patch.dict(sys.modules, {"redis": mock_redis_module}):
            adapter = _fresh_adapter()
            adapter.is_available()

        call_args = mock_redis_module.from_url.call_args
        assert call_args[0][0] == custom_url


# ---------------------------------------------------------------------------
# Test: delete key that doesn't exist
# ---------------------------------------------------------------------------


def test_delete_nonexistent_key():
    """delete() on nonexistent key doesn't raise error."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.delete.return_value = 0  # Key doesn't exist

    mock_redis_module = MagicMock()
    mock_redis_module.from_url.return_value = mock_client

    with patch.dict(sys.modules, {"redis": mock_redis_module}):
        adapter = _fresh_adapter()
        # Should not raise
        adapter.delete("nonexistent_key")

        mock_client.delete.assert_called_once_with("nonexistent_key")


# ---------------------------------------------------------------------------
# Test: get with key miss (None from Redis)
# ---------------------------------------------------------------------------


def test_get_key_miss():
    """get() returns None when key doesn't exist in Redis."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.get.return_value = None

    mock_redis_module = MagicMock()
    mock_redis_module.from_url.return_value = mock_client

    with patch.dict(sys.modules, {"redis": mock_redis_module}):
        adapter = _fresh_adapter()
        result = adapter.get("missing_key")

    assert result is None

