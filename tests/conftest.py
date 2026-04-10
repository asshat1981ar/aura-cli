"""Root-level pytest configuration and shared fixtures for the AURA test suite.

Fixtures defined here are automatically available to all tests under
``tests/`` without explicit import.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.fixtures.mock_llm import MockModelAdapter


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
