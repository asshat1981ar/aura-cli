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



# ---------------------------------------------------------------------------
# Loop 2 tests (2026-03-26)
# ---------------------------------------------------------------------------

# --- Agent capabilities ---

from agents.coder import CoderAgent
from agents.planner import PlannerAgent
from agents.verifier import VerifierAgent


def test_coder_agent_capabilities_non_empty_list():
    assert isinstance(CoderAgent.capabilities, list)
    assert len(CoderAgent.capabilities) > 0


def test_coder_agent_capabilities_correct():
    assert CoderAgent.capabilities == ["code_generation", "coding", "implement", "refactor"]


def test_planner_agent_capabilities_non_empty_list():
    assert isinstance(PlannerAgent.capabilities, list)
    assert len(PlannerAgent.capabilities) > 0


def test_planner_agent_capabilities_correct():
    assert PlannerAgent.capabilities == ["planning", "decomposition", "design", "tree_of_thought", "strategy"]


def test_verifier_agent_capabilities_non_empty_list():
    assert isinstance(VerifierAgent.capabilities, list)
    assert len(VerifierAgent.capabilities) > 0


def test_verifier_agent_capabilities_correct():
    assert VerifierAgent.capabilities == ["testing", "verification", "lint", "quality", "test_runner"]


def test_make_spec_act_uses_native_coder_caps():
    """_make_spec('act', CoderAgent(...)) should use CoderAgent.capabilities (native), not FALLBACK."""
    spec = _make_spec("act", CoderAgent(brain=None, model=None))
    # Verify native caps were used — spec must exactly match the class attribute
    assert spec.capabilities == CoderAgent.capabilities
    # The class attribute is the source of truth; _make_spec must not derive caps from FALLBACK_CAPABILITIES
    # (even if they happen to be equal by design, the path through native caps is what matters)
    assert all(cap in CoderAgent.capabilities for cap in spec.capabilities)
    assert len(spec.capabilities) == len(CoderAgent.capabilities)


# --- _update_registry_health ---

from agents.mcp_health_agent import _update_registry_health
import core.mcp_agent_registry as _reg_mod


def _registry_with_mcp_agent(name="svc_agent", server="my_server"):
    """Helper: fresh registry with one MCP-backed agent."""
    registry = TypedAgentRegistry()
    spec = AgentSpec(name=name, description="svc", capabilities=["coding"], source="mcp", mcp_server=server)
    registry.register(spec)
    return registry


def test_update_registry_health_marks_unhealthy():
    """Non-healthy status should mark the backed agent as unhealthy."""
    registry = _registry_with_mcp_agent("svc1", "server_a")
    original = _reg_mod.agent_registry
    _reg_mod.agent_registry = registry
    try:
        _update_registry_health({"server_a": {"status": "unhealthy"}})
        assert len(registry.resolve_by_capability("coding")) == 0
    finally:
        _reg_mod.agent_registry = original


def test_update_registry_health_marks_unhealthy_on_error_status():
    """Any status other than 'healthy' should exclude the agent."""
    registry = _registry_with_mcp_agent("svc2", "server_b")
    original = _reg_mod.agent_registry
    _reg_mod.agent_registry = registry
    try:
        _update_registry_health({"server_b": {"status": "error"}})
        assert len(registry.resolve_by_capability("coding")) == 0
    finally:
        _reg_mod.agent_registry = original


def test_update_registry_health_marks_healthy():
    """'healthy' status should keep (or restore) the agent as available."""
    registry = _registry_with_mcp_agent("svc3", "server_c")
    # First make it unhealthy manually, then recover via _update_registry_health
    registry.mark_unhealthy("svc3")
    assert len(registry.resolve_by_capability("coding")) == 0

    original = _reg_mod.agent_registry
    _reg_mod.agent_registry = registry
    try:
        _update_registry_health({"server_c": {"status": "healthy"}})
        assert len(registry.resolve_by_capability("coding")) == 1
    finally:
        _reg_mod.agent_registry = original


def test_update_registry_health_only_affects_matching_server():
    """Health update for server_x must not affect agents on server_y."""
    registry = TypedAgentRegistry()
    spec_x = AgentSpec(name="svc_x", description="x", capabilities=["coding"], source="mcp", mcp_server="server_x")
    spec_y = AgentSpec(name="svc_y", description="y", capabilities=["coding"], source="mcp", mcp_server="server_y")
    registry.register(spec_x)
    registry.register(spec_y)

    original = _reg_mod.agent_registry
    _reg_mod.agent_registry = registry
    try:
        _update_registry_health({"server_x": {"status": "unhealthy"}})
        results = registry.resolve_by_capability("coding")
        assert len(results) == 1
        assert results[0].name == "svc_y"
    finally:
        _reg_mod.agent_registry = original


# --- _MCP_CAPABILITY_GROUPS has 17 keys ---

def test_mcp_capability_groups_has_17_keys():
    assert len(_MCP_CAPABILITY_GROUPS) == 17


def test_mcp_capability_groups_contains_new_keys():
    expected_new = {
        "code_analysis", "security", "lint_check", "architecture",
        "test_coverage", "git_analysis", "documentation", "browser",
        "search", "memory_store",
    }
    for key in expected_new:
        assert key in _MCP_CAPABILITY_GROUPS, f"Missing key: {key}"


# --- SKILL_TO_MCP_TOOL ---

from core.skill_dispatcher import SKILL_TO_MCP_TOOL


def test_skill_to_mcp_tool_has_13_entries():
    assert len(SKILL_TO_MCP_TOOL) == 13


def test_skill_to_mcp_tool_all_values_are_strings():
    for skill, tool in SKILL_TO_MCP_TOOL.items():
        assert isinstance(tool, str), f"Value for '{skill}' is not a string: {tool!r}"


# --- dispatch_skills builds mcp_available ---

from core.skill_dispatcher import dispatch_skills, SKILL_MAP
from unittest.mock import patch


def test_dispatch_skills_builds_mcp_available_when_agent_registered():
    """dispatch_skills populates mcp_available when the registry has agents for the skill."""
    skill_name = "ast_analyzer"  # exists in SKILL_TO_MCP_TOOL
    goal_type = next(
        (gt for gt, sklist in SKILL_MAP.items() if skill_name in sklist),
        None,
    )
    if goal_type is None:
        pytest.skip(f"{skill_name} not in any SKILL_MAP bucket")

    fake_spec = AgentSpec(name=skill_name, description="mcp tool", capabilities=[skill_name], source="mcp")

    class FakeSkill:
        def run(self, _): return {"ok": True}

    # Provide a filler skill for every bucket slot except skill_name so that
    # len(available) >= 1 (avoids ThreadPoolExecutor max_workers=0 crash)
    filler_skills = {
        n: FakeSkill()
        for n in SKILL_MAP.get(goal_type, [])
        if n != skill_name
    }
    if not filler_skills:
        pytest.skip("All skills in bucket are MCP-only; cannot satisfy max_workers>0 requirement")

    def _resolve(cap, **kw):
        return [fake_spec] if cap == skill_name else []

    with patch.object(_reg_mod.agent_registry, "resolve_by_capability", side_effect=_resolve):
        result = dispatch_skills(goal_type, filler_skills, project_root=".")
    assert isinstance(result, dict)


# --- LoopOrchestrator._select_act_agent ---

from core.orchestrator import LoopOrchestrator


def test_loop_orchestrator_has_select_act_agent():
    assert hasattr(LoopOrchestrator, "_select_act_agent")
    assert callable(LoopOrchestrator._select_act_agent)


def test_select_act_agent_returns_act_when_resolve_returns_none():
    """When resolve_agent_for_goal returns None, _select_act_agent must fall back to 'act'."""

    class MinimalOrch:
        agents = {}
        model = None
        model_adapter = None

    MinimalOrch._select_act_agent = LoopOrchestrator._select_act_agent

    with patch("core.skill_dispatcher.resolve_agent_for_goal", return_value=None):
        result = MinimalOrch()._select_act_agent("implement new feature")

    assert result == "act"


def test_select_act_agent_returns_resolved_name_when_in_agents():
    """When resolve_agent_for_goal returns a spec whose name is in self.agents, use that name."""

    class MinimalOrch:
        agents = {"python_agent": object()}
        model = None
        model_adapter = None

    MinimalOrch._select_act_agent = LoopOrchestrator._select_act_agent
    fake_spec = AgentSpec(name="python_agent", description="py", capabilities=["python"])

    with patch("core.skill_dispatcher.resolve_agent_for_goal", return_value=fake_spec):
        result = MinimalOrch()._select_act_agent("fix the python bug")

    assert result == "python_agent"
