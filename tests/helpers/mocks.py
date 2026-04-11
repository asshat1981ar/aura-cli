"""Mock utilities for dependency injection and testing."""

from __future__ import annotations

from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar
from unittest.mock import MagicMock, Mock, PropertyMock, patch

T = TypeVar("T")


class MockContainer:
    """Mock dependency injection container for testing.
    
    Provides a clean way to mock dependencies in unit tests without
    modifying the global Container state.
    
    Example:
        >>> container = MockContainer()
        >>> mock_db = container.mock(DatabaseInterface)
        >>> mock_db.query.return_value = ["result"]
        >>> service = MyService(container.resolve(DatabaseInterface))
    """

    def __init__(self) -> None:
        """Initialize empty mock container."""
        self._mocks: Dict[Type[Any], Mock] = {}
        self._patches: list[patch] = []

    def mock(self, interface: Type[T], **kwargs: Any) -> Mock:
        """Get or create a mock for an interface.
        
        Args:
            interface: The type/interface to mock.
            **kwargs: Additional attributes to set on the mock.
            
        Returns:
            Mock object configured for the interface.
        """
        if interface not in self._mocks:
            # Create mock with spec from interface if possible
            try:
                self._mocks[interface] = Mock(spec=interface)
            except (TypeError, AttributeError):
                self._mocks[interface] = Mock()
        mock = self._mocks[interface]
        for key, value in kwargs.items():
            setattr(mock, key, value)
        return mock

    def resolve(self, interface: Type[T]) -> Mock:
        """Resolve interface to mock (DI container compatibility).
        
        Args:
            interface: The type/interface to resolve.
            
        Returns:
            Mock object for the interface.
        """
        return self.mock(interface)

    def register_mock(self, interface: Type[T], mock: Mock) -> None:
        """Register a pre-configured mock.
        
        Args:
            interface: The type/interface being mocked.
            mock: Pre-configured mock object.
        """
        self._mocks[interface] = mock

    def create_magic_mock(self, interface: Type[T], **kwargs: Any) -> MagicMock:
        """Create a MagicMock for an interface.
        
        Args:
            interface: The type/interface to mock.
            **kwargs: Additional attributes to set.
            
        Returns:
            MagicMock object.
        """
        magic = MagicMock(spec=interface, **kwargs)
        self._mocks[interface] = magic
        return magic

    def patch(self, target: str, **kwargs: Any) -> Mock:
        """Create a patch and track it for cleanup.
        
        Args:
            target: Module path to patch.
            **kwargs: Attributes to set on the mock.
            
        Returns:
            Mock object from the patch.
        """
        p = patch(target, **kwargs)
        mock = p.start()
        self._patches.append(p)
        return mock

    def cleanup(self) -> None:
        """Stop all patches and clear mocks."""
        for p in self._patches:
            p.stop()
        self._patches.clear()
        self._mocks.clear()

    def __enter__(self) -> "MockContainer":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        self.cleanup()


class MockLLMResponse:
    """Builder for mock LLM responses."""

    def __init__(self, content: str = "", model: str = "test-model") -> None:
        self.content = content
        self.model = model
        self.usage = {"prompt_tokens": 100, "completion_tokens": 50}
        self.metadata: Dict[str, Any] = {}

    def with_usage(self, prompt: int, completion: int) -> "MockLLMResponse":
        """Set token usage."""
        self.usage = {"prompt_tokens": prompt, "completion_tokens": completion}
        return self

    def with_metadata(self, **kwargs: Any) -> "MockLLMResponse":
        """Add metadata fields."""
        self.metadata.update(kwargs)
        return self

    def build(self) -> Dict[str, Any]:
        """Build the response dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            **self.metadata,
        }


def create_mock_llm_response(
    content: str = "",
    model: str = "test-model",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> Dict[str, Any]:
    """Create a mock LLM response dictionary.
    
    Args:
        content: Response content.
        model: Model name.
        prompt_tokens: Number of prompt tokens.
        completion_tokens: Number of completion tokens.
        
    Returns:
        Mock LLM response dictionary.
    """
    return {
        "content": content,
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def create_mock_agent_response(
    agent_name: str,
    output: Dict[str, Any],
    execution_time: float = 0.1,
) -> Dict[str, Any]:
    """Create a mock agent execution response.
    
    Args:
        agent_name: Name of the agent.
        output: Agent output data.
        execution_time: Simulated execution time.
        
    Returns:
        Mock agent response dictionary.
    """
    return {
        "agent": agent_name,
        "output": output,
        "execution_time": execution_time,
        "success": True,
    }


class MockConfigManager:
    """Mock configuration manager for testing."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._defaults: Dict[str, Any] = {
            "model_name": "test-model",
            "log_level": "DEBUG",
            "dry_run": True,
        }

    def load(self) -> Dict[str, Any]:
        """Return merged config with defaults."""
        return {**self._defaults, **self.config}

    def save(self, config: Dict[str, Any]) -> None:
        """Store config (no-op for mock)."""
        self.config = config

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value."""
        return self.load().get(key, default)


class MockModelAdapter:
    """Mock model adapter for testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or ["Mock response"]
        self._call_count = 0
        self._calls: list[Dict[str, Any]] = []

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Generate mock response."""
        self._calls.append({
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
        })
        response = self.responses[self._call_count % len(self.responses)]
        self._call_count += 1
        return create_mock_llm_response(response)

    def get_calls(self) -> list[Dict[str, Any]]:
        """Get recorded calls."""
        return self._calls

    def assert_called_with_model(self, model: str) -> None:
        """Assert that a specific model was used."""
        assert any(c["model"] == model for c in self._calls), \
            f"Model {model} was not used in any call"


# Pre-built common mocks

def mock_orchestrator() -> Mock:
    """Create a pre-configured mock orchestrator."""
    mock = Mock(spec="core.orchestrator.LoopOrchestrator")
    mock.run_goal.return_value = {"success": True, "cycles": 1}
    return mock


def mock_brain() -> Mock:
    """Create a pre-configured mock brain."""
    mock = Mock(spec="memory.brain.Brain")
    mock.recall.return_value = []
    mock.store.return_value = True
    return mock


def mock_goal_queue() -> Mock:
    """Create a pre-configured mock goal queue."""
    mock = Mock(spec="core.goal_queue.GoalQueue")
    mock.get_pending.return_value = []
    mock.add.return_value = True
    return mock
