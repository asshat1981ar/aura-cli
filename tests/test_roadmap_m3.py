import pytest
from core.mcp_agent_registry import TypedAgentRegistry
from core.types import AgentSpec, MCPServerConfig
from tests.fixtures.mcp_fixtures import MockMCPServer


def test_registry_local_registration():
    registry = TypedAgentRegistry()
    spec = AgentSpec(name="local_agent", description="desc", capabilities=["coding"])
    registry.register(spec)

    assert registry.get_agent("local_agent") == spec
    assert len(registry.resolve_by_capability("coding")) == 1


def test_registry_precedence():
    registry = TypedAgentRegistry()
    local_spec = AgentSpec(name="local_coder", description="local", capabilities=["code"], source="local")
    mcp_spec = AgentSpec(name="mcp_coder", description="mcp", capabilities=["code"], source="mcp")

    registry.register(mcp_spec)
    registry.register(local_spec)

    resolved = registry.resolve_by_capability("code")
    assert len(resolved) == 2
    # Local should be first
    assert resolved[0].source == "local"
    assert resolved[1].source == "mcp"


def test_registry_conflict():
    registry = TypedAgentRegistry()
    spec = AgentSpec(name="agent1", description="desc")
    registry.register(spec)

    with pytest.raises(ValueError):
        registry.register(spec)  # Duplicate name


@pytest.mark.anyio
async def test_registry_mcp_discovery():
    server = MockMCPServer(port=9006)
    server.start()

    registry = TypedAgentRegistry()
    config = MCPServerConfig(name="mock", command="none", port=9006)

    await registry.register_mcp_agents(config)

    agents = registry.list_agents()
    assert len(agents) == 1
    assert agents[0].name == "echo"
    assert agents[0].source == "mcp"
    assert agents[0].mcp_server == "mock"


# ---------------------------------------------------------------------------
# New tests: rich multi-capability registration (2026-03-25)
# ---------------------------------------------------------------------------

from agents.registry import FALLBACK_CAPABILITIES, _make_spec
from core.mcp_agent_registry import _MCP_CAPABILITY_GROUPS, _resolve_mcp_capabilities


def test_rich_capability_resolution():
    """python_agent should rank before act when resolving "python"."""
    registry = TypedAgentRegistry()
    act_spec = AgentSpec(
        name="act",
        description="generic coder",
        capabilities=["code_generation", "coding", "implement", "refactor", "python"],
        source="local",
    )
    python_spec = AgentSpec(
        name="python_agent",
        description="python specialist",
        capabilities=["python", "code_generation", "coding"],
        source="local",
    )
    registry.register(act_spec)
    registry.register(python_spec)

    resolved = registry.resolve_by_capability("python")
    assert len(resolved) == 2
    # python_agent has "python" as PRIMARY; act has it as secondary — python_agent first
    assert resolved[0].name == "python_agent"
    assert resolved[1].name == "act"


def test_primary_capability_sorting():
    """Agent whose primary capability matches should rank above one where it is secondary."""
    registry = TypedAgentRegistry()
    primary_match = AgentSpec(
        name="specialist",
        description="primary match",
        capabilities=["search", "retrieval"],
        source="local",
    )
    secondary_match = AgentSpec(
        name="generalist",
        description="secondary match",
        capabilities=["coding", "search"],
        source="local",
    )
    registry.register(secondary_match)
    registry.register(primary_match)

    resolved = registry.resolve_by_capability("search")
    assert resolved[0].name == "specialist"
    assert resolved[1].name == "generalist"


def test_local_before_mcp_sorting():
    """Local agents should still rank before MCP agents even when both have the same primary."""
    registry = TypedAgentRegistry()
    mcp_spec   = AgentSpec(name="mcp_coder",   description="mcp",   capabilities=["coding"], source="mcp")
    local_spec = AgentSpec(name="local_coder", description="local", capabilities=["coding"], source="local")
    registry.register(mcp_spec)
    registry.register(local_spec)

    resolved = registry.resolve_by_capability("coding")
    assert resolved[0].source == "local"
    assert resolved[1].source == "mcp"


def test_fallback_dict_used_for_legacy_agent():
    """Agents without a native capabilities attribute use FALLBACK_CAPABILITIES."""

    class LegacyAgent:
        description = "legacy"

    spec = _make_spec("act", LegacyAgent())
    assert spec.name == "act"
    assert spec.capabilities == FALLBACK_CAPABILITIES["act"]
    assert "code_generation" in spec.capabilities


def test_native_capabilities_override_fallback():
    """If agent declares native capabilities, they take precedence over fallback dict."""

    class SmartAgent:
        description = "smart"
        capabilities = ["my_special_cap", "also_this"]

    spec = _make_spec("act", SmartAgent())  # "act" exists in FALLBACK_CAPABILITIES
    assert spec.capabilities == ["my_special_cap", "also_this"]


def test_fallback_unknown_agent_defaults_to_name():
    """Agents not in FALLBACK_CAPABILITIES and without native capabilities fall back to [name]."""

    class UnknownAgent:
        description = "unknown"

    spec = _make_spec("brand_new_agent", UnknownAgent())
    assert spec.capabilities == ["brand_new_agent"]


def test_mcp_tool_grouping():
    """MCP tools in a known group register under their name AND the group."""
    caps = _resolve_mcp_capabilities("git_log")
    assert "git_log" in caps   # own name (primary)
    assert "git" in caps       # group name (secondary)
    assert caps[0] == "git_log"  # own name is always primary


def test_mcp_tool_no_group():
    """MCP tools not in any group register under their own name only."""
    caps = _resolve_mcp_capabilities("some_unknown_tool")
    assert caps == ["some_unknown_tool"]


def test_mcp_tool_grouping_registered_in_registry():
    """End-to-end: git_log should resolve via both 'git_log' and 'git' after MCP discovery."""
    registry = TypedAgentRegistry()
    spec = AgentSpec(
        name="git_log",
        description="git log tool",
        capabilities=_resolve_mcp_capabilities("git_log"),
        source="mcp",
        mcp_server="mock_git_server",
    )
    registry.register(spec)

    by_own_name = registry.resolve_by_capability("git_log")
    by_group    = registry.resolve_by_capability("git")

    assert len(by_own_name) == 1
    assert by_own_name[0].name == "git_log"
    assert len(by_group) == 1
    assert by_group[0].name == "git_log"


# ---------------------------------------------------------------------------
# Tests: health-check gating (2026-03-25)
# ---------------------------------------------------------------------------

def test_unhealthy_agent_excluded_from_resolution():
    """Agents marked unhealthy should not appear in resolve_by_capability results."""
    registry = TypedAgentRegistry()
    spec = AgentSpec(name="flaky_agent", description="flaky", capabilities=["coding"], source="local")
    registry.register(spec)

    registry.mark_unhealthy("flaky_agent")
    results = registry.resolve_by_capability("coding")
    assert len(results) == 0


def test_unhealthy_agent_included_when_skip_disabled():
    """With skip_unhealthy=False, unhealthy agents ARE returned."""
    registry = TypedAgentRegistry()
    spec = AgentSpec(name="flaky_agent", description="flaky", capabilities=["coding"], source="local")
    registry.register(spec)

    registry.mark_unhealthy("flaky_agent")
    results = registry.resolve_by_capability("coding", skip_unhealthy=False)
    assert len(results) == 1
    assert results[0].name == "flaky_agent"


def test_mark_healthy_restores_agent():
    """mark_healthy() should make a previously-excluded agent visible again."""
    registry = TypedAgentRegistry()
    spec = AgentSpec(name="recovering_agent", description="recovering", capabilities=["coding"], source="local")
    registry.register(spec)

    registry.mark_unhealthy("recovering_agent")
    assert len(registry.resolve_by_capability("coding")) == 0

    registry.mark_healthy("recovering_agent")
    assert len(registry.resolve_by_capability("coding")) == 1


def test_healthy_fallback_used_when_primary_unhealthy():
    """When the primary agent is unhealthy, the next available agent is returned."""
    registry = TypedAgentRegistry()
    primary = AgentSpec(name="primary_coder", description="primary", capabilities=["coding"], source="local")
    fallback = AgentSpec(name="backup_coder",  description="backup",  capabilities=["coding"], source="local")
    registry.register(primary)
    registry.register(fallback)

    registry.mark_unhealthy("primary_coder")
    results = registry.resolve_by_capability("coding")
    assert len(results) == 1
    assert results[0].name == "backup_coder"


# ---------------------------------------------------------------------------
# Tests: classify_goal_smart & resolve_agent_for_goal (2026-03-25)
# ---------------------------------------------------------------------------

from core.skill_dispatcher import (
    classify_goal_smart,
    resolve_agent_for_goal,
    GOAL_TYPE_TO_CAPABILITY,
)


def test_classify_goal_smart_no_model_uses_keywords():
    """Without a model adapter, classify_goal_smart falls back to keyword classification."""
    assert classify_goal_smart("fix the login crash") == "bug_fix"
    assert classify_goal_smart("add a new payments feature") == "feature"
    assert classify_goal_smart("refactor the auth module") == "refactor"


def test_classify_goal_smart_with_failing_model_falls_back():
    """If the model adapter raises, classify_goal_smart falls back to keywords."""

    class BrokenAdapter:
        def respond(self, prompt):
            raise RuntimeError("no model available")

    result = classify_goal_smart("fix the crash in login", model_adapter=BrokenAdapter())
    assert result == "bug_fix"


def test_resolve_agent_for_goal_python():
    """A Python-specific goal should resolve to python_agent over act."""
    from agents.registry import FALLBACK_CAPABILITIES, _make_spec
    registry = TypedAgentRegistry()

    class Fake:
        description = "fake"

    for name in ("act", "python_agent"):
        registry.register(_make_spec(name, Fake()), overwrite=True)

    # Patch the global singleton temporarily
    import core.mcp_agent_registry as reg_mod
    original = reg_mod.agent_registry
    reg_mod.agent_registry = registry
    try:
        result = resolve_agent_for_goal("Fix the bug in the python script", goal_type="bug_fix")
        assert result is not None
        assert result.name == "python_agent"
    finally:
        reg_mod.agent_registry = original


def test_resolve_agent_for_goal_returns_none_when_empty_registry():
    """resolve_agent_for_goal returns None when no agents are registered."""
    import core.mcp_agent_registry as reg_mod
    original = reg_mod.agent_registry
    reg_mod.agent_registry = TypedAgentRegistry()
    try:
        result = resolve_agent_for_goal("add a feature")
        assert result is None
    finally:
        reg_mod.agent_registry = original


def test_goal_type_to_capability_covers_all_types():
    """All classify_goal return values should have a mapping in GOAL_TYPE_TO_CAPABILITY."""
    from core.skill_dispatcher import SKILL_MAP
    for goal_type in SKILL_MAP:
        assert goal_type in GOAL_TYPE_TO_CAPABILITY, f"Missing: {goal_type}"

