"""Tests for orchestrator sub-agent mixin."""

import pytest
from unittest.mock import MagicMock, patch, call

from core.orchestrator_subagents import SubAgentMixin


class MockOrchestrator(SubAgentMixin):
    """Mock orchestrator for testing."""
    
    def __init__(self):
        super().__init__()
        self.memory_store = MagicMock()


class TestSubAgentMixinBasics:
    """Test SubAgentMixin initialization."""

    def test_initialization(self):
        orch = MockOrchestrator()
        assert orch._subagents is not None
        assert orch._subagent_metrics["iota_invocations"] == 0
        assert orch._subagent_metrics["kappa_recordings"] == 0

    def test_metrics_tracking(self):
        orch = MockOrchestrator()
        metrics = orch.get_subagent_metrics()
        assert "metrics" in metrics
        assert "availability" in metrics


class TestIOTAMixin:
    """Test IOTA integration in orchestrator."""

    def test_attempt_error_resolution_when_unavailable(self):
        orch = MockOrchestrator()
        orch._subagents.is_agent_available = MagicMock(return_value=False)
        
        result = orch._attempt_error_resolution("error")
        assert result is None

    def test_attempt_error_resolution_success(self):
        orch = MockOrchestrator()
        orch._subagents.is_agent_available = MagicMock(return_value=True)
        orch._subagents.resolve_error = MagicMock(return_value={
            "resolved": True,
            "fix": "suggested fix",
        })
        
        result = orch._attempt_error_resolution("error", {"context": "value"})
        
        assert result["resolved"] == True
        assert orch._subagent_metrics["iota_invocations"] == 1
        orch._subagents.resolve_error.assert_called_once()

    def test_should_retry_with_iota_unavailable(self):
        orch = MockOrchestrator()
        orch._subagents.is_agent_available = MagicMock(return_value=False)
        
        result = orch._should_retry_with_iota({"error_type": "syntax_error"})
        assert result == False

    def test_should_retry_with_iota_recoverable_error(self):
        orch = MockOrchestrator()
        orch._subagents.is_agent_available = MagicMock(return_value=True)
        
        result = orch._should_retry_with_iota({"error_type": "syntax_error"})
        assert result == True

    def test_should_retry_with_iota_non_recoverable_error(self):
        orch = MockOrchestrator()
        orch._subagents.is_agent_available = MagicMock(return_value=True)
        
        result = orch._should_retry_with_iota({"error_type": "system_crash"})
        assert result == False


class TestKAPPAMixin:
    """Test KAPPA integration in orchestrator."""

    def test_record_workflow_if_enabled(self):
        orch = MockOrchestrator()
        orch._subagents.is_agent_available = MagicMock(return_value=True)
        orch._subagents.record_workflow = MagicMock(return_value={"recorded": True})
        
        orch._record_workflow_if_enabled("wf1", [{"step": 1}])
        
        assert orch._subagent_metrics["kappa_recordings"] == 1
        orch._subagents.record_workflow.assert_called_once_with("wf1", [{"step": 1}])

    def test_try_replay_workflow_success(self):
        orch = MockOrchestrator()
        orch._subagents.is_agent_available = MagicMock(return_value=True)
        orch._subagents.replay_workflow = MagicMock(return_value={
            "replayed": True,
            "result": "success",
        })
        
        result = orch._try_replay_workflow("wf1", {"var": "value"})
        
        assert result["replayed"] == True
        assert orch._subagent_metrics["kappa_replays"] == 1


class TestNUMixin:
    """Test NU integration in orchestrator."""

    def test_check_connectivity_before_cycle_unavailable(self):
        orch = MockOrchestrator()
        orch._subagents.is_agent_available = MagicMock(return_value=False)
        
        result = orch._check_connectivity_before_cycle()
        
        assert result["online"] == True
        assert result["mode"] == "normal"

    def test_check_connectivity_offline_mode(self):
        orch = MockOrchestrator()
        orch._subagents.is_agent_available = MagicMock(return_value=True)
        orch._subagents.check_connectivity = MagicMock(return_value={
            "online": False,
            "mode": "offline",
            "reason": "no_internet",
        })
        
        result = orch._check_connectivity_before_cycle()
        
        assert result["online"] == False
        assert orch._subagent_metrics["nu_offline_switches"] == 1


class TestPIMixin:
    """Test PI integration in orchestrator."""

    def test_load_secure_config_delegates(self):
        orch = MockOrchestrator()
        orch._subagents.load_encrypted_config = MagicMock(return_value={"key": "value"})
        
        result = orch._load_secure_config("/path", key_id="key1")
        
        assert result == {"key": "value"}
        orch._subagents.load_encrypted_config.assert_called_once_with("/path", "key1")

    def test_save_secure_config_delegates(self):
        orch = MockOrchestrator()
        orch._subagents.save_encrypted_config = MagicMock(return_value=True)
        
        result = orch._save_secure_config("/path", {"key": "value"}, key_id="key1")
        
        assert result == True
        orch._subagents.save_encrypted_config.assert_called_once_with(
            "/path", {"key": "value"}, "key1"
        )


class TestRHOMixin:
    """Test RHO integration in orchestrator."""

    def test_run_preflight_health_checks_unavailable(self):
        orch = MockOrchestrator()
        orch._subagents.is_agent_available = MagicMock(return_value=False)
        
        result = orch._run_preflight_health_checks()
        
        assert result["healthy"] == True

    def test_run_preflight_health_checks_with_failures(self):
        orch = MockOrchestrator()
        orch._subagents.is_agent_available = MagicMock(return_value=True)
        orch._subagents.run_health_checks = MagicMock(return_value={
            "healthy": False,
            "checks": {"memory": {"healthy": False, "error": "low memory"}},
        })
        
        result = orch._run_preflight_health_checks()
        
        assert result["healthy"] == False
        assert orch._subagent_metrics["rho_health_checks"] == 1

    def test_is_system_healthy_for_phase_unavailable(self):
        orch = MockOrchestrator()
        orch._subagents.is_agent_available = MagicMock(return_value=False)
        
        result = orch._is_system_healthy_for_phase("plan")
        assert result == True

    def test_is_system_healthy_for_phase_healthy(self):
        orch = MockOrchestrator()
        orch._subagents.is_agent_available = MagicMock(return_value=True)
        orch._subagents.check_component_health = MagicMock(return_value={"healthy": True})
        
        result = orch._is_system_healthy_for_phase("plan")
        assert result == True
        orch._subagents.check_component_health.assert_called()

    def test_is_system_healthy_for_phase_unhealthy(self):
        orch = MockOrchestrator()
        orch._subagents.is_agent_available = MagicMock(return_value=True)
        orch._subagents.check_component_health = MagicMock(return_value={"healthy": False})
        
        result = orch._is_system_healthy_for_phase("plan")
        assert result == False


class TestSIGMAMixin:
    """Test SIGMA integration in orchestrator."""

    def test_run_security_gate_unavailable(self):
        orch = MockOrchestrator()
        orch._subagents.is_agent_available = MagicMock(return_value=False)
        
        result = orch._run_security_gate([{"file": "test.py"}])
        
        assert result["allowed"] == True

    def test_run_security_gate_blocked(self):
        orch = MockOrchestrator()
        orch._subagents.is_agent_available = MagicMock(return_value=True)
        orch._subagents.security_gate_check = MagicMock(return_value={
            "allowed": False,
            "violations": [{"type": "secret"}],
        })
        
        result = orch._run_security_gate([{"file": "config.py"}])
        
        assert result["allowed"] == False
        assert orch._subagent_metrics["sigma_gate_blocks"] == 1

    def test_scan_content_for_secrets_delegates(self):
        orch = MockOrchestrator()
        orch._subagents.scan_for_secrets = MagicMock(return_value=[
            {"type": "api_key", "line": 5},
        ])
        
        result = orch._scan_content_for_secrets("content")
        
        assert len(result) == 1
        orch._subagents.scan_for_secrets.assert_called_once_with("content")


class TestTAUMixin:
    """Test TAU integration in orchestrator."""

    def test_schedule_background_task_delegates(self):
        orch = MockOrchestrator()
        orch._subagents.schedule_task = MagicMock(return_value={"scheduled": True})
        
        def dummy_task():
            return "done"
        
        result = orch._schedule_background_task("task1", dummy_task, "now")
        
        assert result["scheduled"] == True
        assert orch._subagent_metrics["tau_tasks_scheduled"] == 1

    def test_schedule_maintenance_tasks(self):
        orch = MockOrchestrator()
        orch._subagents.is_agent_available = MagicMock(return_value=True)
        orch._schedule_background_task = MagicMock(return_value={"scheduled": True})
        
        orch._schedule_maintenance_tasks()
        
        # Should schedule memory_cleanup and health_check
        assert orch._schedule_background_task.call_count == 2

    def test_cleanup_old_memories(self):
        orch = MockOrchestrator()
        orch.memory_store = MagicMock()
        
        orch._cleanup_old_memories()
        
        orch.memory_store.clear_old.assert_called_once_with(days=30)

    def test_cleanup_old_memories_no_store(self):
        orch = MockOrchestrator()
        orch.memory_store = None
        
        # Should not raise error
        orch._cleanup_old_memories()


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_full_cycle_with_subagents(self):
        """Simulate a full cycle with sub-agent involvement."""
        orch = MockOrchestrator()
        
        # Mock all subagents available
        orch._subagents.is_agent_available = MagicMock(return_value=True)
        orch._subagents.check_connectivity = MagicMock(return_value={"online": True})
        orch._subagents.run_health_checks = MagicMock(return_value={"healthy": True})
        orch._subagents.security_gate_check = MagicMock(return_value={"allowed": True})
        
        # Simulate cycle
        connectivity = orch._check_connectivity_before_cycle()
        assert connectivity["online"] == True
        
        health = orch._run_preflight_health_checks()
        assert health["healthy"] == True
        
        gate = orch._run_security_gate([{"file": "test.py"}])
        assert gate["allowed"] == True
        
        # Verify metrics tracked
        assert orch._subagent_metrics["rho_health_checks"] == 1

    def test_offline_scenario(self):
        """Test behavior when system goes offline."""
        orch = MockOrchestrator()
        orch._subagents.is_agent_available = MagicMock(return_value=True)
        orch._subagents.check_connectivity = MagicMock(return_value={
            "online": False,
            "mode": "offline",
        })
        
        result = orch._check_connectivity_before_cycle()
        
        assert result["online"] == False
        assert orch._subagent_metrics["nu_offline_switches"] == 1

    def test_security_blocked_scenario(self):
        """Test behavior when security gate blocks changes."""
        orch = MockOrchestrator()
        orch._subagents.is_agent_available = MagicMock(return_value=True)
        orch._subagents.security_gate_check = MagicMock(return_value={
            "allowed": False,
            "violations": [
                {"type": "secret", "file": "config.py", "line": 10},
            ],
        })
        
        changes = [{"file": "config.py", "content": "API_KEY=secret123"}]
        result = orch._run_security_gate(changes)
        
        assert result["allowed"] == False
        assert len(result["violations"]) == 1
        assert orch._subagent_metrics["sigma_gate_blocks"] == 1
