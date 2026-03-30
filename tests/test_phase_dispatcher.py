"""Unit tests for core.phase_dispatcher.PhaseDispatcher.

Tests cover:
- Basic dispatch to registered agent
- No-agent fallback returns {}
- Pre-hook blocking returns sentinel
- Pre-hook input modification is forwarded to agent
- Post-hook is called after agent run
- force_legacy_orchestrator bypasses hooks
- Canary routing is skipped when enable_new_orchestrator is False
- Shadow-mode check in orchestrator._run_phase delegation
"""
import unittest
from unittest.mock import MagicMock, call, patch

from core.phase_dispatcher import PhaseDispatcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(return_value=None):
    agent = MagicMock()
    agent.run.return_value = return_value or {"status": "ok"}
    return agent


def _make_hook_engine(should_proceed=True, modified_input=None):
    """Return a mock HookEngine.

    run_pre_hooks returns (should_proceed, input_data).
    run_post_hooks returns None (observational).
    """
    hook_engine = MagicMock()
    hook_engine.run_pre_hooks.side_effect = lambda phase, data: (
        should_proceed,
        modified_input if modified_input is not None else data,
    )
    hook_engine.run_post_hooks.return_value = None
    return hook_engine


# ---------------------------------------------------------------------------
# Basic dispatch
# ---------------------------------------------------------------------------

class TestPhaseDispatcherBasic(unittest.TestCase):
    def test_dispatch_calls_registered_agent(self):
        agent = _make_agent({"result": "planned"})
        dispatcher = PhaseDispatcher(
            agents={"plan": agent},
            hook_engine=_make_hook_engine(),
        )
        result = dispatcher.dispatch("plan", {"goal": "test"})
        agent.run.assert_called_once_with({"goal": "test"})
        self.assertEqual(result, {"result": "planned"})

    def test_dispatch_returns_empty_dict_when_no_agent(self):
        dispatcher = PhaseDispatcher(
            agents={},
            hook_engine=_make_hook_engine(),
        )
        result = dispatcher.dispatch("act", {"task": "something"})
        self.assertEqual(result, {})

    def test_dispatch_with_no_hook_engine(self):
        """When hook_engine is None hooks are skipped gracefully."""
        agent = _make_agent({"done": True})
        dispatcher = PhaseDispatcher(agents={"verify": agent}, hook_engine=None)
        result = dispatcher.dispatch("verify", {"change_set": {}})
        agent.run.assert_called_once()
        self.assertEqual(result, {"done": True})


# ---------------------------------------------------------------------------
# Hook integration
# ---------------------------------------------------------------------------

class TestPhaseDispatcherHooks(unittest.TestCase):
    def test_pre_hook_blocking_returns_sentinel(self):
        agent = _make_agent()
        hook_engine = _make_hook_engine(should_proceed=False)
        dispatcher = PhaseDispatcher(
            agents={"plan": agent},
            hook_engine=hook_engine,
        )
        result = dispatcher.dispatch("plan", {"goal": "blocked"})
        agent.run.assert_not_called()
        self.assertEqual(result, {"_blocked_by_hook": True, "phase": "plan"})

    def test_pre_hook_can_modify_input_data(self):
        """Modified input from pre-hook is forwarded to the agent."""
        agent = _make_agent()
        modified = {"goal": "injected by hook"}
        hook_engine = _make_hook_engine(should_proceed=True, modified_input=modified)
        dispatcher = PhaseDispatcher(
            agents={"plan": agent},
            hook_engine=hook_engine,
        )
        dispatcher.dispatch("plan", {"goal": "original"})
        agent.run.assert_called_once_with(modified)

    def test_post_hook_called_with_agent_result(self):
        agent_result = {"status": "pass", "logs": "all ok"}
        agent = _make_agent(agent_result)
        hook_engine = _make_hook_engine()
        dispatcher = PhaseDispatcher(
            agents={"verify": agent},
            hook_engine=hook_engine,
        )
        dispatcher.dispatch("verify", {})
        hook_engine.run_post_hooks.assert_called_once_with("verify", agent_result)

    def test_post_hook_receives_empty_dict_when_agent_returns_non_dict(self):
        agent = MagicMock()
        agent.run.return_value = "not a dict"
        hook_engine = _make_hook_engine()
        dispatcher = PhaseDispatcher(
            agents={"reflect": agent},
            hook_engine=hook_engine,
        )
        dispatcher.dispatch("reflect", {})
        hook_engine.run_post_hooks.assert_called_once_with("reflect", {})

    def test_post_hook_not_called_when_blocked(self):
        hook_engine = _make_hook_engine(should_proceed=False)
        dispatcher = PhaseDispatcher(
            agents={"plan": _make_agent()},
            hook_engine=hook_engine,
        )
        dispatcher.dispatch("plan", {})
        hook_engine.run_post_hooks.assert_not_called()


# ---------------------------------------------------------------------------
# Legacy bypass
# ---------------------------------------------------------------------------

class TestPhaseDispatcherLegacyBypass(unittest.TestCase):
    def test_force_legacy_skips_hooks(self):
        agent = _make_agent({"legacy": True})
        hook_engine = _make_hook_engine()
        dispatcher = PhaseDispatcher(
            agents={"act": agent},
            hook_engine=hook_engine,
            config={"force_legacy_orchestrator": True},
        )
        result = dispatcher.dispatch("act", {"task": "x"})
        hook_engine.run_pre_hooks.assert_not_called()
        hook_engine.run_post_hooks.assert_not_called()
        self.assertEqual(result, {"legacy": True})

    def test_force_legacy_no_agent_returns_empty(self):
        dispatcher = PhaseDispatcher(
            agents={},
            hook_engine=_make_hook_engine(),
            config={"force_legacy_orchestrator": True},
        )
        result = dispatcher.dispatch("missing", {})
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# Canary routing (unit-level — canary disabled)
# ---------------------------------------------------------------------------

class TestPhaseDispatcherCanary(unittest.TestCase):
    def test_canary_not_triggered_without_flag(self):
        """CANARY_PHASES fall through to synchronous agent when flag is off."""
        agent = _make_agent({"discovered": True})
        dispatcher = PhaseDispatcher(
            agents={"mcp_discovery": agent},
            hook_engine=_make_hook_engine(),
            config={},  # enable_new_orchestrator absent
        )
        result = dispatcher.dispatch("mcp_discovery", {})
        agent.run.assert_called_once()
        self.assertEqual(result, {"discovered": True})

    def test_non_canary_phase_never_routes_to_async(self):
        """Non-canary phases are never routed async even when flag is set."""
        agent = _make_agent({"planned": True})
        dispatcher = PhaseDispatcher(
            agents={"plan": agent},
            hook_engine=_make_hook_engine(),
            config={"enable_new_orchestrator": True},
        )
        result = dispatcher.dispatch("plan", {})
        agent.run.assert_called_once()
        self.assertEqual(result, {"planned": True})


# ---------------------------------------------------------------------------
# Config propagation
# ---------------------------------------------------------------------------

class TestPhaseDispatcherConfig(unittest.TestCase):
    def test_default_config_is_empty_dict(self):
        dispatcher = PhaseDispatcher(agents={})
        self.assertEqual(dispatcher._config, {})

    def test_project_root_defaults_to_dot(self):
        dispatcher = PhaseDispatcher(agents={})
        self.assertEqual(dispatcher.project_root, ".")

    def test_custom_project_root_stored(self):
        dispatcher = PhaseDispatcher(agents={}, project_root="/my/project")
        self.assertEqual(dispatcher.project_root, "/my/project")


# ---------------------------------------------------------------------------
# Orchestrator delegation smoke test
# ---------------------------------------------------------------------------

class TestOrchestratorDelegation(unittest.TestCase):
    """Verify that LoopOrchestrator._run_phase delegates to PhaseDispatcher."""

    def _make_orchestrator(self):
        from pathlib import Path
        from core.orchestrator import LoopOrchestrator

        agents = {
            "ingest": _make_agent({"context": "ok"}),
            "plan": _make_agent({"steps": ["step1"]}),
            "critique": _make_agent({"issues": []}),
            "synthesize": _make_agent({"tasks": []}),
            "act": _make_agent({"changes": []}),
            "verify": _make_agent({"status": "pass", "failures": [], "logs": ""}),
            "reflect": _make_agent({"summary": "done"}),
        }
        orc = LoopOrchestrator(agents=agents, project_root=Path("/tmp"))
        return orc

    def test_run_phase_delegates_to_phase_dispatcher(self):
        orc = self._make_orchestrator()
        # Patch the dispatcher's dispatch method
        orc._phase_dispatcher.dispatch = MagicMock(return_value={"patched": True})
        result = orc._run_phase("plan", {"goal": "test"})
        orc._phase_dispatcher.dispatch.assert_called_once_with("plan", {"goal": "test"})
        self.assertEqual(result, {"patched": True})

    def test_run_phase_shadow_check_skipped_when_blocked_by_hook(self):
        orc = self._make_orchestrator()
        orc._phase_dispatcher.dispatch = MagicMock(
            return_value={"_blocked_by_hook": True, "phase": "plan"}
        )
        with patch.object(orc, "_run_shadow_check") as mock_shadow:
            orc._run_phase("plan", {})
            mock_shadow.assert_not_called()

    def test_phase_dispatcher_instantiated_on_orchestrator(self):
        orc = self._make_orchestrator()
        self.assertIsInstance(orc._phase_dispatcher, PhaseDispatcher)

    def test_phase_dispatcher_uses_same_agents(self):
        orc = self._make_orchestrator()
        self.assertIs(orc._phase_dispatcher.agents, orc.agents)

    def test_phase_dispatcher_uses_same_hook_engine(self):
        orc = self._make_orchestrator()
        self.assertIs(orc._phase_dispatcher.hook_engine, orc.hook_engine)


if __name__ == "__main__":
    unittest.main()
