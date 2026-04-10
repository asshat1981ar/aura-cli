import pytest
from core.orchestrator import LoopOrchestrator
from agents.registry import default_agents
from unittest.mock import MagicMock


def test_orchestrator_fallback_on_mcp_failure(monkeypatch):
    """Verify that if async path fails, it falls back to legacy path (M4-004, M5-007)."""
    # Enable canary
    from core.config_manager import config

    monkeypatch.setattr(config, "get", lambda k, default=None: True if k == "enable_new_orchestrator" else config.effective_config.get(k, default))

    # Mock legacy agent
    legacy_agent = MagicMock()
    legacy_agent.run.return_value = {"status": "legacy_success"}

    agents = {"mcp_discovery": legacy_agent}
    orch = LoopOrchestrator(agents=agents)

    # Mock _dispatch_task to FAIL
    async def mock_fail_dispatch(req):
        from core.types import TaskResult

        return TaskResult(task_id=req.task_id, status="error", output={}, error="Simulated MCP Failure")

    monkeypatch.setattr(orch, "_dispatch_task", mock_fail_dispatch)

    # Run phase
    result = orch._run_phase("mcp_discovery", {})

    # Should HAVE fallen back to legacy agent
    assert result["status"] == "legacy_success"
    assert legacy_agent.run.called


def test_orchestrator_force_legacy_bypass(monkeypatch):
    """Verify that force_legacy_orchestrator completely bypasses new logic."""
    from core.config_manager import config

    # Set BOTH enable_new and force_legacy
    def mock_get(k, default=None):
        if k == "enable_new_orchestrator":
            return True
        if k == "force_legacy_orchestrator":
            return True
        return config.effective_config.get(k, default)

    monkeypatch.setattr(config, "get", mock_get)

    orch = LoopOrchestrator(agents={"test": MagicMock()})

    # If it hit the canary logic, it would log "orchestrator_canary_routing"
    # But force_legacy comes BEFORE that.

    # We can check if it tries to import anyio (which is in the canary block)
    # Actually, let's just mock the legacy agent and verify it's called.
    legacy_agent = MagicMock()
    orch.agents["test"] = legacy_agent

    orch._run_phase("test", {"val": 1})

    assert legacy_agent.run.called
