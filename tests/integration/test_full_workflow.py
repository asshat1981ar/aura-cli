"""End-to-end workflow integration tests.

Tests cover:
- Complete goal lifecycle from creation to completion
- Multi-agent coordination workflows
- Error recovery and retry mechanisms
- State persistence across phases
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

# Test helpers
from tests.helpers.fixtures import TestFixture, temp_fixture
from tests.helpers.factories import create_test_goal, create_test_config
from tests.helpers.mocks import MockContainer, create_mock_llm_response

pytestmark = [pytest.mark.integration]


class TestGoalLifecycle:
    """End-to-end tests for complete goal lifecycle."""

    def test_goal_creation_and_queueing(self, tmp_path):
        """Test creating a goal and adding it to the queue."""
        from core.goal_queue import GoalQueue
        
        queue_path = tmp_path / "goal_queue.json"
        queue = GoalQueue(queue_path=str(queue_path))
        
        goal = create_test_goal("Test goal for lifecycle")
        queue.add(goal)
        
        # Check that queue has goals using has_goals() method
        assert queue.has_goals() is True
        
        # Retrieve the goal using next()
        retrieved = queue.next()
        assert retrieved["description"] == "Test goal for lifecycle"

    def test_goal_persistence_across_instances(self, tmp_path):
        """Test that goals persist across queue instances."""
        from core.goal_queue import GoalQueue
        
        queue_path = tmp_path / "goal_queue.json"
        
        # Create first instance and add goal
        queue1 = GoalQueue(queue_path=str(queue_path))
        goal = create_test_goal("Persistent goal")
        queue1.add(goal)
        
        # Create second instance and verify goal exists
        queue2 = GoalQueue(queue_path=str(queue_path))
        
        assert queue2.has_goals() is True
        retrieved = queue2.next()
        assert retrieved["description"] == "Persistent goal"

    def test_goal_status_transitions(self, tmp_path):
        """Test goal status transitions through lifecycle."""
        from core.goal_queue import GoalQueue
        
        queue = GoalQueue(queue_path=str(tmp_path / "goal_queue.json"))
        
        goal = create_test_goal("Status transition test")
        goal_id = goal["id"]
        queue.add(goal)
        
        # Initial queue should have the goal
        assert queue.has_goals() is True
        
        # Get the goal (moves to in-flight)
        retrieved = queue.next()
        assert retrieved["id"] == goal_id
        
        # Queue should be empty now
        assert queue.has_goals() is False


class TestConfigurationWorkflow:
    """Tests for configuration loading and validation workflow."""

    def test_config_loading_from_file(self, tmp_path):
        """Test loading configuration from a file."""
        from core.config_schema import AuraConfig, validate_config
        
        config_path = tmp_path / "config.json"
        config_data = create_test_config(model_name="test-model")
        config_path.write_text(json.dumps(config_data))
        
        # Load and validate
        loaded = json.loads(config_path.read_text())
        is_valid, errors = validate_config(loaded)
        
        assert is_valid is True
        assert len(errors) == 0

    def test_config_validation_with_invalid_values(self):
        """Test config validation catches invalid values."""
        from core.config_schema import validate_config
        
        invalid_config = create_test_config(max_iterations=-1)
        is_valid, errors = validate_config(invalid_config)
        
        assert is_valid is False
        assert len(errors) > 0

    def test_config_default_values(self):
        """Test that config provides appropriate defaults."""
        from core.config_schema import AuraConfig
        
        config = AuraConfig()
        
        assert config.max_iterations == 10
        assert config.max_cycles == 5
        assert config.dry_run is False


class TestAgentCoordination:
    """Tests for multi-agent coordination workflows."""

    def test_agent_registry_contains_expected_agents(self):
        """Test that agent registry contains expected agents."""
        try:
            from agents.registry import default_agents
            
            # Mock dependencies
            mock_brain = Mock()
            mock_model = Mock()
            
            agents = default_agents(mock_brain, mock_model)
            
            # Should have expected agent types
            assert isinstance(agents, dict)
            assert len(agents) > 0
        except ImportError:
            pytest.skip("Agent registry not available")

    def test_agent_base_class_exists(self):
        """Test that agent base class exists with expected interface."""
        try:
            from agents.base import Agent
            
            # Verify Agent base class exists
            assert hasattr(Agent, 'run')
        except ImportError:
            pytest.skip("Agent base class not available")


class TestMemoryWorkflow:
    """Tests for memory storage and retrieval workflows."""

    def test_memory_brain_operations(self, tmp_path):
        """Test basic memory brain operations."""
        try:
            from memory.brain import Brain
            
            db_path = tmp_path / "brain.db"
            brain = Brain(str(db_path))
            
            # Store a memory (Brain.remember takes single data argument)
            test_data = {"key": "value", "test": True}
            brain.remember(test_data)
            
            # Verify the brain was initialized properly
            assert brain._db_path.exists()
        except ImportError:
            pytest.skip("Memory brain not available")


class TestErrorHandlingWorkflow:
    """Tests for error handling and recovery workflows."""

    def test_exception_hierarchy_catchability(self):
        """Test that exceptions can be caught by parent classes."""
        from core.exceptions import (
            AURAError, FileToolsError, PathTraversalError,
            AgentError, AgentExecutionError
        )
        
        # Test that specific exceptions can be caught by parents
        exceptions_to_test = [
            PathTraversalError("test"),
            AgentExecutionError("test"),
        ]
        
        for exc in exceptions_to_test:
            caught = False
            try:
                raise exc
            except AURAError:
                caught = True
            assert caught, f"{type(exc).__name__} should be catchable as AURAError"

    def test_error_context_preservation(self):
        """Test that error context is preserved in exceptions."""
        from core.exceptions import ConfigValidationError
        
        error_msg = "Invalid config at path /test"
        error = ConfigValidationError(error_msg)
        
        assert error_msg in str(error)


class TestFileToolsWorkflow:
    """Tests for file operations workflow."""

    def test_safe_file_operations(self, tmp_path):
        """Test safe file read/write operations."""
        try:
            from core.file_tools import read_file, write_file
            
            test_file = tmp_path / "test.txt"
            test_content = "Hello, World!"
            
            # Write file
            write_file(str(test_file), test_content)
            
            # Read file
            content = read_file(str(test_file))
            
            assert content == test_content
        except ImportError:
            pytest.skip("File tools not available")

    def test_file_operations_with_security_checks(self, tmp_path):
        """Test that file operations respect security constraints."""
        try:
            from core.file_tools import write_file
            from core.exceptions import PathTraversalError
            
            # Attempt path traversal should be blocked
            with pytest.raises((PathTraversalError, ValueError)):
                write_file(str(tmp_path / ".." / "outside.txt"), "content")
        except ImportError:
            pytest.skip("File tools not available")


class TestOrchestratorIntegration:
    """Integration tests for the orchestrator with real components."""

    def test_orchestrator_initialization(self, tmp_path):
        """Test orchestrator can be initialized with dependencies."""
        try:
            from core.orchestrator import LoopOrchestrator
            
            mock_agents = {"test": Mock()}
            mock_brain = Mock()
            mock_model = Mock()
            
            orchestrator = LoopOrchestrator(
                agents=mock_agents,
                brain=mock_brain,
                model=mock_model
            )
            
            assert orchestrator is not None
        except ImportError:
            pytest.skip("Orchestrator not available")


class TestPolicyWorkflow:
    """Tests for policy enforcement workflow."""

    def test_policy_creation_and_validation(self):
        """Test policy creation and validation."""
        try:
            from core.policy import Policy
            from core.policies.sliding_window import SlidingWindowPolicy
            
            policy = Policy(max_cycles=3)
            
            # Policy delegates to implementation
            assert isinstance(policy.impl, SlidingWindowPolicy)
            assert policy.impl.max_cycles == 3
        except ImportError:
            pytest.skip("Policy not available")

    def test_policy_cycle_tracking(self):
        """Test policy cycle tracking via evaluate method."""
        try:
            from core.policy import Policy
            
            policy = Policy(max_cycles=2)
            
            # Test through the evaluate interface
            # Empty history and no verification - should allow continuation
            result = policy.evaluate([], {"status": "fail"})
            assert result == ""  # Can continue
            
            # After 2 cycles with failures - should hit MAX_CYCLES
            history = [{"cycle": 1}, {"cycle": 2}]
            result = policy.evaluate(history, {"status": "fail"})
            assert result == "MAX_CYCLES"
        except ImportError:
            pytest.skip("Policy not available")


class TestGitToolsWorkflow:
    """Tests for git operations workflow."""

    def test_git_tools_import(self):
        """Test git tools module can be imported."""
        try:
            from core.git_tools import GitTools
            
            # Just verify the class exists and can be instantiated
            git = GitTools()
            assert git is not None
        except ImportError:
            pytest.skip("Git tools not available")


class TestEndToEndScenarios:
    """Complete end-to-end scenarios."""

    def test_full_config_to_execution_flow(self, tmp_path):
        """Test complete flow from config to execution setup."""
        from core.config_schema import AuraConfig, validate_config
        
        # Create config
        config_data = create_test_config(
            model_name="test-model",
            max_iterations=5,
            max_cycles=3
        )
        
        # Validate config
        is_valid, errors = validate_config(config_data)
        assert is_valid is True
        
        # Create config object
        config = AuraConfig(**config_data)
        assert config.model_name == "test-model"
        assert config.max_iterations == 5
        assert config.max_cycles == 3

    def test_error_recovery_flow(self):
        """Test error detection and recovery flow."""
        from core.exceptions import AURAError, ValidationError
        
        error_recorded = False
        
        try:
            # Simulate an error
            raise ValidationError("Test validation error")
        except AURAError as e:
            # Should be catchable as AURAError
            error_recorded = True
            assert "validation" in str(e).lower()
        
        assert error_recorded is True

    def test_dependency_injection_workflow(self):
        """Test dependency injection container workflow."""
        from core.container import Container
        
        # Define a test interface
        class ITestService:
            def do_work(self) -> str: ...
        
        class TestService:
            def do_work(self) -> str:
                return "work done"
        
        # Register and resolve
        service = TestService()
        Container.register_singleton(ITestService, service)
        
        resolved = Container.resolve(ITestService)
        assert resolved.do_work() == "work done"
        
        # Cleanup
        Container.unregister(ITestService)
