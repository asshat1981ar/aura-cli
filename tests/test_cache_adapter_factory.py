"""
Tests for memory.cache_adapter_factory module.

Covers:
  - create_cache_adapter() with and without MOMENTO_API_KEY env var
  - get_adapter() for all adapter types (local, momento, redis)
  - Error handling for invalid adapter types
"""

import os
import unittest
from unittest.mock import MagicMock, patch
import sys


def setup_mocks():
    """Inject mock adapters into sys.modules to avoid import errors."""
    if "momento" not in sys.modules:
        sys.modules["momento"] = MagicMock()
        sys.modules["momento.responses"] = MagicMock()
        sys.modules["momento.requests"] = MagicMock()


setup_mocks()


class TestCreateCacheAdapter(unittest.TestCase):
    """Test create_cache_adapter() factory function."""

    def setUp(self):
        self._orig_key = os.environ.pop("MOMENTO_API_KEY", None)

    def tearDown(self):
        if self._orig_key is not None:
            os.environ["MOMENTO_API_KEY"] = self._orig_key

    def test_creates_local_adapter_without_env_key(self):
        """When MOMENTO_API_KEY is not set, returns LocalCacheAdapter."""
        # Ensure no key
        os.environ.pop("MOMENTO_API_KEY", None)
        from memory.cache_adapter_factory import create_cache_adapter

        adapter = create_cache_adapter()
        self.assertEqual(type(adapter).__name__, "LocalCacheAdapter")

    def test_creates_momento_adapter_with_env_key(self):
        """When MOMENTO_API_KEY is set, returns MomentoAdapter."""
        os.environ["MOMENTO_API_KEY"] = "test-key-12345"
        from memory.cache_adapter_factory import create_cache_adapter

        adapter = create_cache_adapter()
        self.assertEqual(type(adapter).__name__, "MomentoAdapter")

    def test_creates_local_adapter_with_empty_env_key(self):
        """Empty MOMENTO_API_KEY is treated as not set."""
        os.environ["MOMENTO_API_KEY"] = ""
        from memory.cache_adapter_factory import create_cache_adapter

        adapter = create_cache_adapter()
        self.assertEqual(type(adapter).__name__, "LocalCacheAdapter")


class TestGetAdapter(unittest.TestCase):
    """Test get_adapter() function for explicit adapter selection."""

    def test_get_local_adapter(self):
        """get_adapter('local') returns LocalCacheAdapter."""
        from memory.cache_adapter_factory import get_adapter

        adapter = get_adapter("local")
        self.assertEqual(type(adapter).__name__, "LocalCacheAdapter")

    def test_get_momento_adapter(self):
        """get_adapter('momento') returns MomentoAdapter."""
        from memory.cache_adapter_factory import get_adapter

        adapter = get_adapter("momento")
        self.assertEqual(type(adapter).__name__, "MomentoAdapter")

    def test_get_redis_adapter(self):
        """get_adapter('redis') returns RedisCacheAdapter."""
        from memory.cache_adapter_factory import get_adapter

        adapter = get_adapter("redis")
        self.assertEqual(type(adapter).__name__, "RedisCacheAdapter")

    def test_get_adapter_case_insensitive(self):
        """get_adapter() is case-insensitive."""
        from memory.cache_adapter_factory import get_adapter

        adapter1 = get_adapter("LOCAL")
        self.assertEqual(type(adapter1).__name__, "LocalCacheAdapter")

        adapter2 = get_adapter("Momento")
        self.assertEqual(type(adapter2).__name__, "MomentoAdapter")

        adapter3 = get_adapter("REDIS")
        self.assertEqual(type(adapter3).__name__, "RedisCacheAdapter")

    def test_get_adapter_invalid_type_raises(self):
        """get_adapter() raises ValueError for unknown adapter type."""
        from memory.cache_adapter_factory import get_adapter

        with self.assertRaises(ValueError) as ctx:
            get_adapter("invalid_type")
        self.assertIn("Unknown adapter type", str(ctx.exception))
        self.assertIn("invalid_type", str(ctx.exception))

    def test_get_adapter_invalid_type_shows_options(self):
        """ValueError message includes valid options."""
        from memory.cache_adapter_factory import get_adapter

        with self.assertRaises(ValueError) as ctx:
            get_adapter("unknown")
        msg = str(ctx.exception)
        self.assertIn("local", msg)
        self.assertIn("momento", msg)
        self.assertIn("redis", msg)

    def test_get_adapter_empty_string_raises(self):
        """get_adapter() with empty string raises ValueError."""
        from memory.cache_adapter_factory import get_adapter

        with self.assertRaises(ValueError):
            get_adapter("")

    def test_get_adapter_none_raises(self):
        """get_adapter(None) raises AttributeError or TypeError."""
        from memory.cache_adapter_factory import get_adapter

        with self.assertRaises((AttributeError, TypeError)):
            get_adapter(None)  # type: ignore


class TestAdapterTypesConstant(unittest.TestCase):
    """Test the _ADAPTER_TYPES constant."""

    def test_adapter_types_defined(self):
        """_ADAPTER_TYPES constant contains all adapter types."""
        from memory.cache_adapter_factory import _ADAPTER_TYPES

        self.assertIsInstance(_ADAPTER_TYPES, tuple)
        self.assertIn("local", _ADAPTER_TYPES)
        self.assertIn("momento", _ADAPTER_TYPES)
        self.assertIn("redis", _ADAPTER_TYPES)
        self.assertEqual(len(_ADAPTER_TYPES), 3)


class TestAdapterInterface(unittest.TestCase):
    """Test that all adapters returned have the expected interface."""

    def test_local_adapter_has_cache_methods(self):
        """LocalCacheAdapter has cache_get, cache_set, cache_delete."""
        from memory.cache_adapter_factory import get_adapter

        adapter = get_adapter("local")
        self.assertTrue(callable(getattr(adapter, "cache_get", None)))
        self.assertTrue(callable(getattr(adapter, "cache_set", None)))
        self.assertTrue(callable(getattr(adapter, "cache_delete", None)))

    def test_momento_adapter_has_cache_methods(self):
        """MomentoAdapter has cache_get, cache_set, cache_delete."""
        from memory.cache_adapter_factory import get_adapter

        adapter = get_adapter("momento")
        self.assertTrue(callable(getattr(adapter, "cache_get", None)))
        self.assertTrue(callable(getattr(adapter, "cache_set", None)))
        self.assertTrue(callable(getattr(adapter, "cache_delete", None)))

    def test_redis_adapter_has_cache_methods(self):
        """RedisCacheAdapter has get and set methods (different naming convention)."""
        from memory.cache_adapter_factory import get_adapter

        adapter = get_adapter("redis")
        self.assertTrue(callable(getattr(adapter, "get", None)))
        self.assertTrue(callable(getattr(adapter, "set", None)))


if __name__ == "__main__":
    unittest.main()
