"""Unit tests for the Dependency Injection Container.

Tests cover:
- Singleton registration and resolution
- Factory registration and lazy initialization
- Resolution errors
- Container clearing and management
"""

import pytest
from typing import Protocol

from core.container import Container


# Test fixtures and protocols
class IDatabase(Protocol):
    """Test database interface."""

    def query(self, sql: str) -> list: ...


class ICache(Protocol):
    """Test cache interface."""

    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> None: ...


class MockDatabase:
    """Mock database implementation."""

    def __init__(self, connection_string: str = "default"):
        self.connection_string = connection_string
        self.queries: list[str] = []

    def query(self, sql: str) -> list:
        self.queries.append(sql)
        return []


class MockCache:
    """Mock cache implementation."""

    def __init__(self):
        self._store: dict[str, str] = {}
        self.initialized = True

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: str) -> None:
        self._store[key] = value


@pytest.fixture(autouse=True)
def clear_container():
    """Clear container before and after each test."""
    Container.clear()
    yield
    Container.clear()


class TestContainerSingletonRegistration:
    """Tests for singleton registration."""

    def test_register_singleton_and_resolve(self):
        """Test basic singleton registration and resolution."""
        db = MockDatabase()
        Container.register_singleton(IDatabase, db)

        resolved = Container.resolve(IDatabase)

        assert resolved is db
        assert resolved.connection_string == "default"

    def test_singleton_returns_same_instance(self):
        """Test that singleton returns the same instance on multiple resolves."""
        db = MockDatabase()
        Container.register_singleton(IDatabase, db)

        resolved1 = Container.resolve(IDatabase)
        resolved2 = Container.resolve(IDatabase)

        assert resolved1 is resolved2
        assert resolved1 is db

    def test_multiple_singleton_registrations(self):
        """Test registering multiple different singletons."""
        db = MockDatabase()
        cache = MockCache()

        Container.register_singleton(IDatabase, db)
        Container.register_singleton(ICache, cache)

        assert Container.resolve(IDatabase) is db
        assert Container.resolve(ICache) is cache

    def test_register_replaces_existing_singleton(self):
        """Test that registering a new singleton replaces the old one."""
        db1 = MockDatabase("first")
        db2 = MockDatabase("second")

        Container.register_singleton(IDatabase, db1)
        Container.register_singleton(IDatabase, db2)

        resolved = Container.resolve(IDatabase)
        assert resolved is db2
        assert resolved.connection_string == "second"


class TestContainerFactoryRegistration:
    """Tests for factory registration."""

    def test_register_factory_and_resolve(self):
        """Test basic factory registration and resolution."""
        call_count = 0

        def create_db():
            nonlocal call_count
            call_count += 1
            return MockDatabase()

        Container.register_factory(IDatabase, create_db)

        resolved = Container.resolve(IDatabase)

        assert isinstance(resolved, MockDatabase)
        assert call_count == 1

    def test_factory_caches_result_as_singleton(self):
        """Test that factory result is cached after first resolution."""
        call_count = 0

        def create_db():
            nonlocal call_count
            call_count += 1
            return MockDatabase()

        Container.register_factory(IDatabase, create_db)

        resolved1 = Container.resolve(IDatabase)
        resolved2 = Container.resolve(IDatabase)

        assert resolved1 is resolved2
        assert call_count == 1  # Factory called only once

    def test_factory_with_complex_initialization(self):
        """Test factory with complex initialization logic."""
        config = {"host": "localhost", "port": 5432}

        def create_db_with_config():
            return MockDatabase(f"{config['host']}:{config['port']}")

        Container.register_factory(IDatabase, create_db_with_config)

        db = Container.resolve(IDatabase)
        assert db.connection_string == "localhost:5432"


class TestContainerResolutionErrors:
    """Tests for resolution error handling."""

    def test_resolve_unregistered_raises_keyerror(self):
        """Test that resolving unregistered interface raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            Container.resolve(IDatabase)

        assert "No registration for" in str(exc_info.value)
        assert "IDatabase" in str(exc_info.value)

    def test_try_resolve_unregistered_returns_none(self):
        """Test that try_resolve returns None for unregistered interface."""
        result = Container.try_resolve(IDatabase)

        assert result is None

    def test_try_resolve_registered_returns_instance(self):
        """Test that try_resolve returns instance for registered interface."""
        db = MockDatabase()
        Container.register_singleton(IDatabase, db)

        result = Container.try_resolve(IDatabase)

        assert result is db


class TestContainerStateManagement:
    """Tests for container state management."""

    def test_is_registered_returns_true_for_singleton(self):
        """Test is_registered returns True for registered singleton."""
        Container.register_singleton(IDatabase, MockDatabase())

        assert Container.is_registered(IDatabase) is True

    def test_is_registered_returns_true_for_factory(self):
        """Test is_registered returns True for registered factory."""
        Container.register_factory(IDatabase, lambda: MockDatabase())

        assert Container.is_registered(IDatabase) is True

    def test_is_registered_returns_false_for_unregistered(self):
        """Test is_registered returns False for unregistered interface."""
        assert Container.is_registered(IDatabase) is False

    def test_clear_removes_all_registrations(self):
        """Test that clear removes all registrations."""
        Container.register_singleton(IDatabase, MockDatabase())
        Container.register_singleton(ICache, MockCache())

        Container.clear()

        assert Container.is_registered(IDatabase) is False
        assert Container.is_registered(ICache) is False

    def test_unregister_specific_interface(self):
        """Test unregistering a specific interface."""
        Container.register_singleton(IDatabase, MockDatabase())
        Container.register_singleton(ICache, MockCache())

        result = Container.unregister(IDatabase)

        assert result is True
        assert Container.is_registered(IDatabase) is False
        assert Container.is_registered(ICache) is True

    def test_unregister_nonexistent_returns_false(self):
        """Test unregistering non-existent interface returns False."""
        result = Container.unregister(IDatabase)

        assert result is False


class TestContainerEdgeCases:
    """Edge case tests for container."""

    def test_register_none_as_singleton(self):
        """Test that None can be registered as a singleton."""
        Container.register_singleton(IDatabase, None)

        result = Container.resolve(IDatabase)

        assert result is None

    def test_concrete_class_registration(self):
        """Test registration using concrete classes instead of protocols."""
        Container.register_singleton(MockDatabase, MockDatabase("test"))

        result = Container.resolve(MockDatabase)

        assert isinstance(result, MockDatabase)

    def test_factory_raises_exception(self):
        """Test handling of factory that raises exception."""

        def failing_factory():
            raise RuntimeError("Factory failed")

        Container.register_factory(IDatabase, failing_factory)

        with pytest.raises(RuntimeError, match="Factory failed"):
            Container.resolve(IDatabase)
