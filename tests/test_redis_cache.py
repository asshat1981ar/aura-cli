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
