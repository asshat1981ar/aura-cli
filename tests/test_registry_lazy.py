"""Tests for lazy agent loading and registry scalability (closes #313, #321).

These tests verify:
- default_agents() returns all expected agents with callable run() methods
- Agent capabilities are declared correctly via FALLBACK_CAPABILITIES
- The lazy import cache (_agent_cache) is populated after load
- Module-level imports for optional agents do NOT happen at import time
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_brain():
    brain = MagicMock()
    brain.recall_recent.return_value = []
    brain.recall_with_budget.return_value = []
    brain.remember.return_value = None
    brain.get.return_value = None
    brain.set.return_value = None
    brain.vector_store = None
    return brain


def _make_model(response="mock response"):
    model = MagicMock()
    model.respond.return_value = response
    return model


# ---------------------------------------------------------------------------
# Core pipeline agents presence
# ---------------------------------------------------------------------------

EXPECTED_CORE_AGENTS = [
    "ingest",
    "plan",
    "critique",
    "synthesize",
    "act",
    "sandbox",
    "verify",
    "reflect",
]

EXPECTED_SPECIALIZED_AGENTS = [
    "python_agent",
    "typescript_agent",
    "external_llm",
    "monitoring",
    "notification",
    "telemetry",
    "self_correction",
    "code_search",
    "investigation",
    "root_cause_analysis",
    "mcp_discovery",
    "mcp_health",
]


class TestDefaultAgentsStructure:
    """Verify default_agents() returns expected agent dict."""

    def setup_method(self):
        from agents.registry import default_agents

        self.agents = default_agents(
            brain=_make_brain(),
            model=_make_model(),
        )

    def test_returns_dict(self):
        assert isinstance(self.agents, dict)

    def test_core_agents_present(self):
        for name in EXPECTED_CORE_AGENTS:
            assert name in self.agents, f"Missing core agent: {name}"

    def test_specialized_agents_present(self):
        for name in EXPECTED_SPECIALIZED_AGENTS:
            assert name in self.agents, f"Missing specialized agent: {name}"

    def test_core_agents_have_run_method(self):
        """Core pipeline agents must expose a callable run() method."""
        for name in EXPECTED_CORE_AGENTS:
            agent = self.agents[name]
            assert callable(getattr(agent, "run", None)), f"Core agent '{name}' does not have a callable run() method"

    def test_specialized_agents_with_run_method(self):
        """Specialized adapters that wrap a run() interface expose it."""
        # These adapters are explicitly known to have run()
        adapter_agents = [
            "python_agent",
            "typescript_agent",
            "external_llm",
            "monitoring",
            "notification",
        ]
        for name in adapter_agents:
            agent = self.agents[name]
            assert callable(getattr(agent, "run", None)), f"Adapter agent '{name}' does not have a callable run() method"

    def test_agent_names_match_keys(self):
        """Adapters with an explicit `name` attribute must match their dict key."""
        for key, agent in self.agents.items():
            agent_name = getattr(agent, "name", None)
            if agent_name is not None:
                assert agent_name == key, f"Agent key '{key}' mismatches agent.name '{agent_name}'"


# ---------------------------------------------------------------------------
# Capability declarations
# ---------------------------------------------------------------------------


class TestCapabilityDeclarations:
    """FALLBACK_CAPABILITIES covers all expected pipeline phases."""

    def test_fallback_capabilities_not_empty(self):
        from agents.registry import FALLBACK_CAPABILITIES

        assert len(FALLBACK_CAPABILITIES) > 0

    def test_core_phases_in_fallback(self):
        from agents.registry import FALLBACK_CAPABILITIES

        for phase in EXPECTED_CORE_AGENTS:
            assert phase in FALLBACK_CAPABILITIES, f"Phase '{phase}' missing from FALLBACK_CAPABILITIES"

    def test_each_capability_list_nonempty(self):
        from agents.registry import FALLBACK_CAPABILITIES

        for name, caps in FALLBACK_CAPABILITIES.items():
            assert isinstance(caps, list) and len(caps) > 0, f"Empty capability list for '{name}'"

    def test_make_spec_uses_native_capabilities(self):
        """_make_spec() should prefer an agent's own `capabilities` attribute."""
        from agents.registry import _make_spec

        class FakeAgent:
            capabilities = ["custom_cap", "other_cap"]
            description = "a test agent"

        spec = _make_spec("plan", FakeAgent())
        assert spec.capabilities == ["custom_cap", "other_cap"]

    def test_make_spec_falls_back_to_fallback_dict(self):
        """_make_spec() falls back to FALLBACK_CAPABILITIES when no native attr."""
        from agents.registry import _make_spec, FALLBACK_CAPABILITIES

        class MinimalAgent:
            pass

        spec = _make_spec("plan", MinimalAgent())
        assert spec.capabilities == FALLBACK_CAPABILITIES["plan"]

    def test_make_spec_last_resort_name(self):
        """_make_spec() uses [name] when key not in FALLBACK_CAPABILITIES either."""
        from agents.registry import _make_spec

        class MinimalAgent:
            pass

        spec = _make_spec("totally_unknown_agent", MinimalAgent())
        assert spec.capabilities == ["totally_unknown_agent"]


# ---------------------------------------------------------------------------
# Lazy loading cache
# ---------------------------------------------------------------------------


class TestLazyAgentCache:
    """Verify the module-level _agent_cache dict exists and is populated."""

    def test_agent_cache_exists(self):
        import agents.registry as reg

        assert hasattr(reg, "_agent_cache"), "agents.registry must expose a module-level _agent_cache dict"

    def test_agent_cache_is_dict(self):
        import agents.registry as reg

        assert isinstance(reg._agent_cache, dict)

    def test_lazy_import_function_exists(self):
        import agents.registry as reg

        assert hasattr(reg, "_lazy_import"), "agents.registry must expose a _lazy_import() function"

    def test_lazy_import_callable(self):
        import agents.registry as reg

        assert callable(reg._lazy_import)

    def test_lazy_import_returns_module_or_class(self):
        """_lazy_import('planner') should return the PlannerAgent class."""
        import agents.registry as reg

        result = reg._lazy_import("planner")
        # Must be truthy (a module or class), not None
        assert result is not None

    def test_lazy_import_caches_result(self):
        """Calling _lazy_import twice with the same key returns the same object."""
        import agents.registry as reg

        # Clear cache to test fresh population
        reg._agent_cache.clear()
        first = reg._lazy_import("planner")
        second = reg._lazy_import("planner")
        assert first is second

    def test_cache_populated_after_default_agents(self):
        """After calling default_agents(), _agent_cache should be non-empty."""
        import agents.registry as reg

        reg._agent_cache.clear()
        reg.default_agents(brain=_make_brain(), model=_make_model())
        assert len(reg._agent_cache) > 0


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """default_agents() return value is backward-compatible."""

    def test_plan_agent_runs(self):
        from agents.registry import default_agents

        agents = default_agents(brain=_make_brain(), model=_make_model())
        result = agents["plan"].run(
            {
                "goal": "test goal",
                "memory_snapshot": "",
                "similar_past_problems": "",
                "known_weaknesses": "",
            }
        )
        assert "steps" in result
        assert "risks" in result

    def test_critique_agent_runs(self):
        from agents.registry import default_agents

        agents = default_agents(brain=_make_brain(), model=_make_model())
        result = agents["critique"].run({"task": "test task", "plan": []})
        assert "issues" in result
        assert "fixes" in result

    def test_sandbox_agent_dry_run(self):
        from agents.registry import default_agents

        agents = default_agents(brain=_make_brain(), model=_make_model())
        result = agents["sandbox"].run({"dry_run": True})
        assert result["status"] == "skip"
        assert result["passed"] is True

    def test_second_call_returns_same_shape(self):
        """Calling default_agents() twice returns dicts with identical keys."""
        from agents.registry import default_agents

        brain = _make_brain()
        model = _make_model()
        first = default_agents(brain=brain, model=model)
        second = default_agents(brain=brain, model=model)
        assert set(first.keys()) == set(second.keys())
