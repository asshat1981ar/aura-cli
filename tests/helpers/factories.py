"""Object factories for test data generation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConfigFactory:
    """Factory for creating test configurations."""

    model_name: str = "google/gemini-2.0-flash-exp:free"
    log_level: str = "DEBUG"
    dry_run: bool = True
    timeout: int = 5
    max_iterations: int = 3
    max_cycles: int = 2

    def build(self) -> Dict[str, Any]:
        """Build configuration dictionary."""
        return {
            "model_name": self.model_name,
            "log_level": self.log_level,
            "dry_run": self.dry_run,
            "timeout": self.timeout,
            "max_iterations": self.max_iterations,
            "max_cycles": self.max_cycles,
            "version": "1.0.0",
        }

    def with_model(self, model: str) -> "ConfigFactory":
        """Set model name."""
        self.model_name = model
        return self

    def with_timeout(self, timeout: int) -> "ConfigFactory":
        """Set timeout."""
        self.timeout = timeout
        return self

    def with_max_iterations(self, iterations: int) -> "ConfigFactory":
        """Set max iterations."""
        self.max_iterations = iterations
        return self


@dataclass
class AgentFactory:
    """Factory for creating test agent configurations."""

    name: str = "test_agent"
    capabilities: List[str] = field(default_factory=list)
    timeout: float = 30.0
    enabled: bool = True

    def __post_init__(self):
        if not self.capabilities:
            self.capabilities = ["test_capability"]

    def build(self) -> Dict[str, Any]:
        """Build agent configuration dictionary."""
        return {
            "name": self.name,
            "capabilities": self.capabilities,
            "timeout": self.timeout,
            "enabled": self.enabled,
        }

    def with_capability(self, capability: str) -> "AgentFactory":
        """Add a capability."""
        self.capabilities.append(capability)
        return self

    def disabled(self) -> "AgentFactory":
        """Disable the agent."""
        self.enabled = False
        return self


@dataclass
class GoalFactory:
    """Factory for creating test goals."""

    description: str = "Test goal"
    priority: int = 1
    auto_run: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def build(self) -> Dict[str, Any]:
        """Build goal dictionary."""
        return {
            "id": str(uuid.uuid4()),
            "description": self.description,
            "priority": self.priority,
            "auto_run": self.auto_run,
            "status": "pending",
            "metadata": self.metadata,
        }

    def with_high_priority(self) -> "GoalFactory":
        """Set high priority."""
        self.priority = 10
        return self

    def with_auto_run(self) -> "GoalFactory":
        """Enable auto-run."""
        self.auto_run = True
        return self

    def with_metadata(self, **kwargs: Any) -> "GoalFactory":
        """Add metadata."""
        self.metadata.update(kwargs)
        return self


# Convenience functions for quick test data creation


def create_test_config(**overrides: Any) -> Dict[str, Any]:
    """Create a test configuration with optional overrides.

    Args:
        **overrides: Configuration values to override.

    Returns:
        Configuration dictionary.
    """
    factory = ConfigFactory()
    config = factory.build()
    config.update(overrides)
    return config


def create_test_agent(name: str = "test_agent", **overrides: Any) -> Dict[str, Any]:
    """Create a test agent configuration.

    Args:
        name: Agent name.
        **overrides: Values to override.

    Returns:
        Agent configuration dictionary.
    """
    factory = AgentFactory(name=name)
    config = factory.build()
    config.update(overrides)
    return config


def create_test_goal(description: str = "Test goal", **overrides: Any) -> Dict[str, Any]:
    """Create a test goal.

    Args:
        description: Goal description.
        **overrides: Values to override.

    Returns:
        Goal dictionary.
    """
    factory = GoalFactory(description=description)
    config = factory.build()
    config.update(overrides)
    return config


class TestDataBuilder:
    """Builder for complex test data scenarios."""

    def __init__(self) -> None:
        self._configs: List[Dict[str, Any]] = []
        self._agents: List[Dict[str, Any]] = []
        self._goals: List[Dict[str, Any]] = []

    def add_config(self, **kwargs: Any) -> "TestDataBuilder":
        """Add a configuration."""
        self._configs.append(create_test_config(**kwargs))
        return self

    def add_agent(self, name: str, **kwargs: Any) -> "TestDataBuilder":
        """Add an agent."""
        self._agents.append(create_test_agent(name, **kwargs))
        return self

    def add_goal(self, description: str, **kwargs: Any) -> "TestDataBuilder":
        """Add a goal."""
        self._goals.append(create_test_goal(description, **kwargs))
        return self

    def build(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build all test data."""
        return {
            "configs": self._configs,
            "agents": self._agents,
            "goals": self._goals,
        }


# Fixture builders for specific scenarios


def create_minimal_config() -> Dict[str, Any]:
    """Create minimal valid configuration."""
    return {
        "model_name": "test-model",
        "log_level": "INFO",
    }


def create_full_config() -> Dict[str, Any]:
    """Create full configuration with all options."""
    return {
        "model_name": "google/gemini-2.0-flash-exp:free",
        "api_key": "test-api-key",
        "openai_api_key": "test-openai-key",
        "anthropic_api_key": "test-anthropic-key",
        "dry_run": True,
        "decompose": False,
        "max_iterations": 10,
        "max_cycles": 5,
        "strict_schema": False,
        "policy_name": "sliding_window",
        "policy_max_cycles": 5,
        "policy_max_seconds": 120,
        "log_level": "DEBUG",
        "beads": {
            "enabled": True,
            "required": True,
            "timeout_seconds": 20,
        },
        "model_routing": {
            "code_generation": "google/gemini-2.0-flash-exp:free",
            "planning": "google/gemini-2.0-flash-exp:free",
        },
    }


def create_error_scenario_config() -> Dict[str, Any]:
    """Create configuration for error scenario testing."""
    return {
        "model_name": "test-model",
        "log_level": "ERROR",
        "dry_run": True,
        "max_iterations": 1,
        "max_cycles": 1,
    }
