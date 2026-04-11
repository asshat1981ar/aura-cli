"""Tests for sub-agent integration layer."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from core.subagent_integration import SubAgentRegistry, get_subagent_registry


class TestSubAgentRegistry:
    """Test SubAgentRegistry functionality."""

    def test_creation(self):
        registry = SubAgentRegistry()
        assert registry is not None
        assert isinstance(registry._agents, dict)

    def test_get_agent_status(self):
        registry = SubAgentRegistry()
        status = registry.get_agent_status()
        assert "iota" in status
        assert "kappa" in status
        assert "nu" in status
        assert "pi" in status
        assert "rho" in status
        assert "sigma" in status
        assert "tau" in status

    def test_is_agent_available(self):
        registry = SubAgentRegistry()
        # Returns bool based on availability
        assert isinstance(registry.is_agent_available("iota"), bool)
        assert registry.is_agent_available("nonexistent") == False


class TestIOTAIntegration:
    """Test IOTA (Error Resolution) integration."""

    def test_resolve_error_with_mock_engine(self):
        """Test error resolution with mocked engine."""
        registry = SubAgentRegistry()
        mock_engine = MagicMock()
        mock_engine.resolve.return_value = {
            "resolved": True,
            "fix": "suggested fix",
        }

        # Mock the iota property to return our mock
        with patch.object(SubAgentRegistry, "iota", new_callable=PropertyMock) as mock_iota:
            mock_iota.return_value = mock_engine
            result = registry.resolve_error("test error", {"context": "value"})

            mock_engine.resolve.assert_called_once()
            assert result["resolved"] == True
            assert result["fix"] == "suggested fix"

    def test_resolve_error_engine_exception(self):
        """Test error handling when engine throws."""
        registry = SubAgentRegistry()
        mock_engine = MagicMock()
        mock_engine.resolve.side_effect = Exception("Engine failed")

        with patch.object(SubAgentRegistry, "iota", new_callable=PropertyMock) as mock_iota:
            mock_iota.return_value = mock_engine
            result = registry.resolve_error("test error")

            assert result["resolved"] == False
            assert "Engine failed" in result["reason"]


class TestKAPPAIntegration:
    """Test KAPPA (Recording) integration."""

    def test_record_workflow_with_mock_recorder(self):
        """Test workflow recording with mocked recorder."""
        registry = SubAgentRegistry()
        mock_recorder = MagicMock()
        mock_recorder.record.return_value = {"id": "rec1", "steps": 2}

        with patch.object(SubAgentRegistry, "kappa", new_callable=PropertyMock) as mock_kappa:
            mock_kappa.return_value = mock_recorder
            steps = [{"action": "step1"}, {"action": "step2"}]
            result = registry.record_workflow("wf1", steps)

            mock_recorder.record.assert_called_once_with(workflow_id="wf1", steps=steps)
            assert result["recorded"] == True

    def test_replay_workflow_unavailable(self):
        """Test workflow replay when KAPPA unavailable."""
        registry = SubAgentRegistry()

        with patch.object(SubAgentRegistry, "kappa", new_callable=PropertyMock) as mock_kappa:
            mock_kappa.return_value = None
            result = registry.replay_workflow("wf1", {"var": "value"})

            assert result["replayed"] == False
            assert "not available" in result["reason"]


class TestNUIntegration:
    """Test NU (Offline Mode) integration."""

    def test_check_connectivity_with_mock_monitor(self):
        """Test connectivity check with mocked monitor."""
        registry = SubAgentRegistry()
        mock_monitor = MagicMock()
        mock_monitor.check_connectivity.return_value = {
            "online": False,
            "mode": "offline",
        }

        with patch.object(SubAgentRegistry, "nu", new_callable=PropertyMock) as mock_nu:
            mock_nu.return_value = mock_monitor
            result = registry.check_connectivity()

            mock_monitor.check_connectivity.assert_called_once()
            assert result["online"] == False
            assert result["mode"] == "offline"


class TestPIIntegration:
    """Test PI (Config Encryption) integration."""

    def test_load_encrypted_config_with_mock_manager(self):
        """Test config loading with mocked manager."""
        registry = SubAgentRegistry()
        mock_manager = MagicMock()
        mock_manager.load.return_value = {"secret_key": "decrypted_value"}

        with patch.object(SubAgentRegistry, "pi", new_callable=PropertyMock) as mock_pi:
            mock_pi.return_value = mock_manager
            result = registry.load_encrypted_config("/path/to/config", key_id="key1")

            mock_manager.load.assert_called_once_with("/path/to/config", key_id="key1")
            assert result["secret_key"] == "decrypted_value"

    def test_load_encrypted_config_fallback(self):
        """Test fallback to JSON when PI unavailable."""
        registry = SubAgentRegistry()

        with patch.object(SubAgentRegistry, "pi", new_callable=PropertyMock) as mock_pi:
            mock_pi.return_value = None

            import json
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump({"key": "value"}, f)
                temp_path = f.name

            try:
                result = registry.load_encrypted_config(temp_path)
                assert result == {"key": "value"}
            finally:
                os.unlink(temp_path)

    def test_save_encrypted_config_with_mock_manager(self):
        """Test config saving with mocked manager."""
        registry = SubAgentRegistry()
        mock_manager = MagicMock()

        with patch.object(SubAgentRegistry, "pi", new_callable=PropertyMock) as mock_pi:
            mock_pi.return_value = mock_manager
            result = registry.save_encrypted_config("/path", {"key": "value"}, key_id="key1")

            mock_manager.save.assert_called_once_with("/path", {"key": "value"}, key_id="key1")
            assert result == True


class TestRHOIntegration:
    """Test RHO (Health Monitoring) integration."""

    def test_run_health_checks_with_mock_monitor(self):
        """Test health checks with mocked monitor."""
        registry = SubAgentRegistry()
        mock_monitor = MagicMock()
        mock_monitor.check_all.return_value = {
            "healthy": False,
            "checks": {"memory": {"healthy": False}},
        }

        with patch.object(SubAgentRegistry, "rho", new_callable=PropertyMock) as mock_rho:
            mock_rho.return_value = mock_monitor
            result = registry.run_health_checks()

            mock_monitor.check_all.assert_called_once()
            assert result["healthy"] == False

    def test_check_component_health_with_mock(self):
        """Test component health check with mocked monitor."""
        registry = SubAgentRegistry()
        mock_monitor = MagicMock()
        mock_monitor.check_component.return_value = {
            "healthy": True,
            "component": "memory",
        }

        with patch.object(SubAgentRegistry, "rho", new_callable=PropertyMock) as mock_rho:
            mock_rho.return_value = mock_monitor
            result = registry.check_component_health("memory")

            mock_monitor.check_component.assert_called_once_with("memory")
            assert result["healthy"] == True


class TestSIGMAIntegration:
    """Test SIGMA (Security Gate) integration."""

    def test_security_gate_check_with_mock_auditor(self):
        """Test security gate with mocked auditor."""
        registry = SubAgentRegistry()
        mock_auditor = MagicMock()
        mock_auditor.validate_changes.return_value = {
            "allowed": False,
            "violations": [{"type": "secret", "file": "config.py"}],
        }

        with patch.object(SubAgentRegistry, "sigma", new_callable=PropertyMock) as mock_sigma:
            mock_sigma.return_value = mock_auditor
            changes = [{"file": "config.py", "content": "API_KEY=xxx"}]
            result = registry.security_gate_check(changes)

            mock_auditor.validate_changes.assert_called_once_with(changes)
            assert result["allowed"] == False
            assert len(result["violations"]) == 1

    def test_scan_for_secrets_unavailable(self):
        """Test secret scanning when SIGMA unavailable."""
        registry = SubAgentRegistry()

        with patch.object(SubAgentRegistry, "sigma", new_callable=PropertyMock) as mock_sigma:
            mock_sigma.return_value = None
            result = registry.scan_for_secrets("content with secret")

            # Should return empty list when unavailable
            assert result == []


class TestTAUIntegration:
    """Test TAU (Task Scheduling) integration."""

    def test_schedule_task_with_mock_scheduler(self):
        """Test task scheduling with mocked scheduler."""
        registry = SubAgentRegistry()
        mock_scheduler = MagicMock()
        mock_scheduler.schedule.return_value = {"id": "job1", "task_id": "task1"}

        with patch.object(SubAgentRegistry, "tau", new_callable=PropertyMock) as mock_tau:
            mock_tau.return_value = mock_scheduler

            def dummy_task(x):
                return x * 2

            result = registry.schedule_task("task1", dummy_task, "*/5 * * * *", 5)

            assert result["scheduled"] == True
            assert result["job"]["id"] == "job1"

    def test_get_scheduled_tasks_with_mock(self):
        """Test getting scheduled tasks with mocked scheduler."""
        registry = SubAgentRegistry()
        mock_scheduler = MagicMock()
        mock_scheduler.list_tasks.return_value = [
            {"id": "job1", "task_id": "task1"},
            {"id": "job2", "task_id": "task2"},
        ]

        with patch.object(SubAgentRegistry, "tau", new_callable=PropertyMock) as mock_tau:
            mock_tau.return_value = mock_scheduler
            result = registry.get_scheduled_tasks()

            mock_scheduler.list_tasks.assert_called_once()
            assert len(result) == 2


class TestSingletonPattern:
    """Test global registry singleton."""

    def test_get_subagent_registry_singleton(self):
        registry1 = get_subagent_registry()
        registry2 = get_subagent_registry()
        assert registry1 is registry2

    def test_global_registry_preserved_across_calls(self):
        registry1 = get_subagent_registry()
        registry1._test_marker = "test_value"

        registry2 = get_subagent_registry()
        assert registry2._test_marker == "test_value"


class TestErrorHandling:
    """Test error handling in sub-agent integration."""

    def test_resolve_error_handles_exception(self):
        """Test that resolve_error handles engine exceptions gracefully."""
        registry = SubAgentRegistry()
        mock_engine = MagicMock()
        mock_engine.resolve.side_effect = Exception("Engine failed")

        with patch.object(SubAgentRegistry, "iota", new_callable=PropertyMock) as mock_iota:
            mock_iota.return_value = mock_engine
            result = registry.resolve_error("test error")

            assert result["resolved"] == False
            assert "Engine failed" in result["reason"]

    def test_health_checks_handles_exception(self):
        """Test that health checks handle monitor exceptions gracefully."""
        registry = SubAgentRegistry()
        mock_monitor = MagicMock()
        mock_monitor.check_all.side_effect = Exception("Monitor failed")

        with patch.object(SubAgentRegistry, "rho", new_callable=PropertyMock) as mock_rho:
            mock_rho.return_value = mock_monitor
            result = registry.run_health_checks()

            assert result["healthy"] == False
            assert "Monitor failed" in result["error"]
