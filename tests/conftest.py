"""Root-level pytest configuration and shared fixtures for the AURA test suite.

Fixtures defined here are automatically available to all tests under
``tests/`` without explicit import.
"""

from __future__ import annotations

import signal
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from tests.fixtures.mock_llm import MockModelAdapter


# ---------------------------------------------------------------------------
# Marker registration (mirrors pyproject.toml [tool.pytest.ini_options])
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: pure unit tests — no I/O, no external services")
    config.addinivalue_line("markers", "integration: tests that require external services")
    config.addinivalue_line("markers", "e2e: end-to-end tests (full pipeline, may be slow)")
    config.addinivalue_line("markers", "slow: tests that are expected to run >10 s")
    config.addinivalue_line("markers", "security: security-focused tests")


# ---------------------------------------------------------------------------
# Timeout helper fixture
# ---------------------------------------------------------------------------

@contextmanager
def _hard_timeout(seconds: int):
    """SIGALRM-based timeout guard (Unix only)."""
    def _handler(signum, frame):
        raise TimeoutError(f"Test exceeded hard timeout of {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@pytest.fixture
def hard_timeout():
    """Return a context-manager factory ``hard_timeout(seconds)`` that raises
    ``TimeoutError`` if the block exceeds the given duration.

    Example::

        def test_something(hard_timeout):
            with hard_timeout(5):
                result = potentially_slow_function()
    """
    return _hard_timeout


@pytest.fixture
def mock_model_adapter() -> MockModelAdapter:
    """Return a :class:`~tests.fixtures.mock_llm.MockModelAdapter` pre-loaded
    with default pipeline responses (planner, coder, critic, reflector).

    Use this fixture when a test needs a model adapter object but should not
    hit any external LLM API.
    """
    return MockModelAdapter()


@pytest.fixture
def mock_adapter_patch(mock_model_adapter: MockModelAdapter):
    """Patch ``core.model_adapter.ModelAdapter`` with the mock adapter.

    Yields the :class:`~tests.fixtures.mock_llm.MockModelAdapter` instance so
    tests can inspect ``call_log`` or register additional responses via
    ``set_response``.

    Example::

        def test_something(mock_adapter_patch):
            mock_adapter_patch.set_response("my prompt", "my response")
            # code under test that imports ModelAdapter will receive the mock
    """
    with patch("core.model_adapter.ModelAdapter", return_value=mock_model_adapter):
        yield mock_model_adapter
