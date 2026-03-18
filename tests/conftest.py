"""Shared test fixtures for the AURA CLI test suite."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture()
def mock_brain() -> MagicMock:
    """A mock Brain instance with common methods stubbed."""
    brain = MagicMock(name="brain")
    brain.search.return_value = []
    brain.get_context.return_value = ""
    brain.store.return_value = None
    return brain


@pytest.fixture()
def mock_model() -> MagicMock:
    """A mock Model/LLM adapter."""
    model = MagicMock(name="model")
    model.generate.return_value = "mocked response"
    model.classify.return_value = "default"
    return model


@pytest.fixture()
def mock_agents(mock_brain: MagicMock, mock_model: MagicMock) -> dict:
    """Standard agents dict used by the orchestrator and skills."""
    return {
        "brain": mock_brain,
        "model": mock_model,
        "critic": MagicMock(name="critic"),
    }
