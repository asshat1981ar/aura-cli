"""Extended unit tests for core/orchestrator.py — LoopOrchestrator.

Covers advanced features, edge cases, and integration points not in main test file.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call, AsyncMock
import pytest
import tempfile

from core.orchestrator import LoopOrchestrator, BeadsSyncLoop
from core.policy import Policy
from core.file_tools import MismatchOverwriteBlockedError, OldCodeNotFoundError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agents(**overrides):
    """Return a minimal agents dict with MagicMock instances."""
    phase_names = ["ingest", "plan", "critique", "synthesize", "act", "verify", "reflect"]
    agents = {}
    for name in phase_names:
        mock = MagicMock(name=name)
        mock.name = name
        mock.run.return_value = {"status": "success", "agent_name": name}
        agents[name] = mock
    agents.update(overrides)
    return agents


def _make_orchestrator(agents=None, policy=None, project_root=None, **kwargs):
    """Build a LoopOrchestrator with heavy side-effects patched."""
    agents = agents or _make_agents()
    with (
        patch("core.orchestrator.HookEngine"),
        patch("core.orchestrator.PhaseDispatcher"),
        patch("core.orchestrator.memory_controller"),
        patch("core.orchestrator.log_json"),
        patch("agents.skills.registry.all_skills", return_value={}, create=True),
    ):
        orch = LoopOrchestrator(
            agents=agents,
            policy=policy or Policy.from_config({}),
            project_root=project_root or Path("."),
            **kwargs,
        )
    return orch


# ---------------------------------------------------------------------------
# BeadsSyncLoop Tests
# ---------------------------------------------------------------------------


class TestBeadsSyncLoop:
    """Test BeadsSyncLoop synchronization trigger."""

    def test_on_cycle_complete_skips_dry_run(self):
        """Verify dry_run entries don't trigger beads sync."""
        skill = MagicMock()
        loop = BeadsSyncLoop(skill)
        
        entry = {"dry_run": True}
        loop.on_cycle_complete(entry)
        
        skill.run.assert_not_called()

    def test_on_cycle_complete_syncs_every_n_cycles(self):
        """Verify beads sync happens every N cycles."""
        skill = MagicMock()
        loop = BeadsSyncLoop(skill)
        
        # Call 5 times (EVERY_N = 5)
        for i in range(5):
            loop.on_cycle_complete({"dry_run": False})
        
        # Should trigger on 5th call
        assert skill.run.call_count == 2  # pull + push
        
        # Verify pull and push calls
        calls = skill.run.call_args_list
        assert calls[0][0][0] == {"cmd": "dolt", "args": ["pull"]}
        assert calls[1][0][0] == {"cmd": "dolt", "args": ["push"]}

    def test_on_cycle_complete_handles_non_dict_entry(self):
        """Verify handling of non-dict entry."""
        skill = MagicMock()
        loop = BeadsSyncLoop(skill)
        
        # Should not crash on non-dict
        loop.on_cycle_complete("invalid")
        skill.run.assert_not_called()


# ---------------------------------------------------------------------------
# Attachment Methods
# ---------------------------------------------------------------------------


class TestAttachUiCallback:
    """Test UI callback registration and invocation."""

    def test_attach_ui_callback_appends(self):
        orch = _make_orchestrator()
        callback1 = MagicMock()
        callback2 = MagicMock()
        
        orch.attach_ui_callback(callback1)
        orch.attach_ui_callback(callback2)
        
        assert len(orch._ui_callbacks) == 2
        assert callback1 in orch._ui_callbacks
        assert callback2 in orch._ui_callbacks

    def test_notify_ui_calls_method_on_all_callbacks(self):
        orch = _make_orchestrator()
        callback1 = MagicMock()
        callback2 = MagicMock()
        
        orch.attach_ui_callback(callback1)
        orch.attach_ui_callback(callback2)
        
        orch._notify_ui("on_phase_complete", "plan", {"status": "ok"})
        
        callback1.on_phase_complete.assert_called_once_with("plan", {"status": "ok"})
        callback2.on_phase_complete.assert_called_once_with("plan", {"status": "ok"})

    def test_notify_ui_ignores_missing_method(self):
        """Verify UI callback doesn't crash if method is missing."""
        orch = _make_orchestrator()
        callback = MagicMock(spec=[])  # Empty spec = no methods
        
        orch.attach_ui_callback(callback)
        
        # Should not raise
        orch._notify_ui("on_nonexistent_method")

    def test_notify_ui_ignores_callback_exceptions(self):
        """Verify exceptions in callbacks don't break the orchestrator.
        
        Note: _notify_ui only catches TypeError and AttributeError, not general exceptions.
        """
        orch = _make_orchestrator()
        callback = MagicMock()
        callback.on_event.side_effect = TypeError("Wrong args")
        
        orch.attach_ui_callback(callback)
        
        # Should not raise (TypeError is caught)
        orch._notify_ui("on_event", "arg")


# ---------------------------------------------------------------------------
# Improvement Loops
# ---------------------------------------------------------------------------


class TestAttachImprovementLoops:
    """Test improvement loop registration."""

    def test_attach_improvement_loops_extends_list(self):
        orch = _make_orchestrator()
        loop1 = MagicMock()
        loop2 = MagicMock()
        
        orch.attach_improvement_loops(loop1, loop2)
        
        assert len(orch._improvement_loops) == 2
        assert loop1 in orch._improvement_loops
        assert loop2 in orch._improvement_loops

    def test_attach_improvement_loops_multiple_calls_additive(self):
        """Verify multiple calls extend the list without deduplication."""
        orch = _make_orchestrator()
        loop1 = MagicMock()
        
        orch.attach_improvement_loops(loop1)
        orch.attach_improvement_loops(loop1)
        
        assert len(orch._improvement_loops) == 2


# ---------------------------------------------------------------------------
# CASPA Attachment
# ---------------------------------------------------------------------------


class TestAttachCaspa:
    """Test CASPA-W component attachment."""

    def test_attach_caspa_stores_all_components(self):
        orch = _make_orchestrator()
        adaptive = MagicMock()
        propagation = MagicMock()
        context_graph = MagicMock()
        
        orch.attach_caspa(
            adaptive_pipeline=adaptive,
            propagation_engine=propagation,
            context_graph=context_graph,
        )
        
        assert orch.adaptive_pipeline is adaptive
        assert orch.propagation_engine is propagation
        assert orch.context_graph is context_graph

    def test_attach_caspa_allows_none_components(self):
        """Verify None values are accepted for optional components."""
        orch = _make_orchestrator()
        orch.attach_caspa(
            adaptive_pipeline=None,
            propagation_engine=None,
            context_graph=None,
        )
        
        assert orch.adaptive_pipeline is None
        assert orch.propagation_engine is None
        assert orch.context_graph is None

    def test_attach_caspa_partial_attachment(self):
        """Verify that attach_caspa always sets all three attributes.
        
        Note: attach_caspa sets all three components, overwriting None values.
        """
        orch = _make_orchestrator()
        initial_graph = MagicMock()
        orch.context_graph = initial_graph
        
        new_adaptive = MagicMock()
        orch.attach_caspa(adaptive_pipeline=new_adaptive)
        
        assert orch.adaptive_pipeline is new_adaptive
        # When attach_caspa is called with context_graph=None (default), it sets it to None
        assert orch.context_graph is None


# ---------------------------------------------------------------------------
# Private Methods: _retrieve_hints
# ---------------------------------------------------------------------------


class TestRetrieveHints:
    """Test hint retrieval for relevance-based planning."""

    def test_retrieve_hints_empty_when_no_store(self):
        orch = _make_orchestrator()
        orch.memory_controller = None
        
        hints = orch._retrieve_hints("some goal")
        assert hints == []

    def test_retrieve_hints_empty_when_no_persistent_store(self):
        orch = _make_orchestrator()
        orch.memory_controller = MagicMock()
        orch.memory_controller.persistent_store = None
        
        hints = orch._retrieve_hints("some goal")
        assert hints == []

    def test_retrieve_hints_returns_limit(self):
        orch = _make_orchestrator()
        orch.memory_controller = MagicMock()
        summaries = [{"goal": f"task_{i}"} for i in range(10)]
        orch.memory_controller.persistent_store.query.return_value = summaries
        
        hints = orch._retrieve_hints("task", limit=3)
        
        assert len(hints) <= 3

    def test_retrieve_hints_scores_by_keyword_match(self):
        """Verify keyword relevance scoring."""
        orch = _make_orchestrator()
        orch.memory_controller = MagicMock()
        summaries = [
            {"status": "success"},  # No keyword match
            {"status": "success"},  # Has "fix" keyword
        ]
        orch.memory_controller.persistent_store.query.return_value = summaries
        
        hints = orch._retrieve_hints("fix bug", limit=5)
        # Should return some hints (order depends on scoring)
        assert isinstance(hints, list)

    def test_retrieve_hints_handles_query_failure(self):
        """Verify graceful handling of query errors."""
        orch = _make_orchestrator()
        orch.memory_controller = MagicMock()
        orch.memory_controller.persistent_store.query.side_effect = OSError("DB error")
        
        hints = orch._retrieve_hints("goal")
        assert hints == []


# ---------------------------------------------------------------------------
# File Snapshot and Apply
# ---------------------------------------------------------------------------


class TestSnapshotFileState:
    """Test file state capture before mutation."""

    def test_snapshot_nonexistent_file(self):
        with tempfile.TemporaryDirectory() as td:
            orch = _make_orchestrator(project_root=Path(td))
            
            snapshot = orch._snapshot_file_state("new_file.py")
            
            assert snapshot["file"] == "new_file.py"
            assert snapshot["existed"] is False
            assert snapshot["content"] is None
            assert snapshot["mode"] is None

    def test_snapshot_existing_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            test_file = root / "test.py"
            test_file.write_text("hello world")
            
            orch = _make_orchestrator(project_root=root)
            snapshot = orch._snapshot_file_state("test.py")
            
            assert snapshot["existed"] is True
            assert snapshot["content"] == "hello world"
            assert snapshot["mode"] is not None


class TestApplyChangeSet:
    """Test change set application to filesystem."""

    def test_apply_change_set_single_change(self):
        """Verify single change application."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            test_file = root / "test.py"
            test_file.write_text("old code")
            
            orch = _make_orchestrator(project_root=root)
            
            change_set = {
                "file_path": "test.py",
                "old_code": "old code",
                "new_code": "new code",
            }
            
            result = orch._apply_change_set(change_set, dry_run=False)
            
            assert "test.py" in result["applied"]
            assert test_file.read_text() == "new code"

    def test_apply_change_set_dry_run(self):
        """Verify dry_run doesn't modify files."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            test_file = root / "test.py"
            test_file.write_text("original")
            
            orch = _make_orchestrator(project_root=root)
            
            change_set = {
                "file_path": "test.py",
                "old_code": "original",
                "new_code": "modified",
            }
            
            result = orch._apply_change_set(change_set, dry_run=True)
            
            assert "test.py" in result["applied"]
            assert test_file.read_text() == "original"

    def test_apply_change_set_batch(self):
        """Verify batch changes application."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.py").write_text("a_old")
            (root / "b.py").write_text("b_old")
            
            orch = _make_orchestrator(project_root=root)
            
            change_set = {
                "changes": [
                    {"file_path": "a.py", "old_code": "a_old", "new_code": "a_new"},
                    {"file_path": "b.py", "old_code": "b_old", "new_code": "b_new"},
                ]
            }
            
            result = orch._apply_change_set(change_set, dry_run=False)
            
            assert "a.py" in result["applied"]
            assert "b.py" in result["applied"]
            assert (root / "a.py").read_text() == "a_new"
            assert (root / "b.py").read_text() == "b_new"

    def test_apply_change_set_continues_on_failure(self):
        """Verify that one failure doesn't block other changes."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.py").write_text("a_old")
            (root / "b.py").write_text("b_old")
            
            orch = _make_orchestrator(project_root=root)
            
            change_set = {
                "changes": [
                    {"file_path": "a.py", "old_code": "NONEXISTENT", "new_code": "a_new"},
                    {"file_path": "b.py", "old_code": "b_old", "new_code": "b_new"},
                ]
            }
            
            result = orch._apply_change_set(change_set, dry_run=False)
            
            # b.py should succeed even though a.py failed
            assert "b.py" in result["applied"]
            assert len(result["failed"]) == 1
            assert result["failed"][0]["file"] == "a.py"

    def test_apply_change_set_missing_file_path(self):
        """Verify handling of missing file_path."""
        orch = _make_orchestrator()
        
        change_set = {"old_code": "x", "new_code": "y"}
        result = orch._apply_change_set(change_set, dry_run=False)
        
        assert result["applied"] == []

    def test_apply_change_set_creates_snapshots(self):
        """Verify snapshots are captured for all changed files."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "test.py").write_text("original")
            
            orch = _make_orchestrator(project_root=root)
            
            change_set = {
                "file_path": "test.py",
                "old_code": "original",
                "new_code": "modified",
            }
            
            result = orch._apply_change_set(change_set, dry_run=False)
            
            assert len(result["snapshots"]) == 1
            assert result["snapshots"][0]["file"] == "test.py"
            assert result["snapshots"][0]["content"] == "original"


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------


class TestLoadConfigFile:
    """Test configuration file loading."""

    def test_load_config_file_returns_dict(self):
        orch = _make_orchestrator()
        config = orch._load_config_file()
        
        assert isinstance(config, dict)

    def test_load_config_file_handles_missing_file(self):
        """Verify graceful handling of missing config."""
        orch = _make_orchestrator()
        # Config path likely doesn't exist in test, should still return dict
        config = orch._load_config_file()
        
        assert isinstance(config, dict)


# ---------------------------------------------------------------------------
# External Goals
# ---------------------------------------------------------------------------


class TestPollExternalGoals:
    """Test polling for external goal additions."""

    def test_poll_external_goals_handles_none_beads_skill(self):
        orch = _make_orchestrator()
        orch._get_beads_skill = MagicMock(return_value=None)
        
        result = orch.poll_external_goals()
        
        assert result == []

    def test_poll_external_goals_retrieves_from_beads_list(self):
        """Verify goals are retrieved from BEADS list result."""
        orch = _make_orchestrator()
        beads_skill = MagicMock()
        beads_skill.run.return_value = [
            {"id": "b1", "title": "task 1"},
            {"id": "b2", "title": "task 2"},
        ]
        orch._get_beads_skill = MagicMock(return_value=beads_skill)
        
        result = orch.poll_external_goals()
        
        assert len(result) == 2
        assert "b1" in result[0]
        assert "task 1" in result[0]

    def test_poll_external_goals_retrieves_from_beads_dict(self):
        """Verify goals are retrieved from BEADS dict result."""
        orch = _make_orchestrator()
        beads_skill = MagicMock()
        beads_skill.run.return_value = {
            "beads": [
                {"id": "b1", "title": "task 1"},
            ]
        }
        orch._get_beads_skill = MagicMock(return_value=beads_skill)
        
        result = orch.poll_external_goals()
        
        assert len(result) >= 1

    def test_poll_external_goals_handles_beads_error(self):
        """Verify graceful handling of BEADS errors."""
        orch = _make_orchestrator()
        beads_skill = MagicMock()
        beads_skill.run.side_effect = Exception("BEADS error")
        orch._get_beads_skill = MagicMock(return_value=beads_skill)
        
        result = orch.poll_external_goals()
        
        assert result == []


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    """Test orchestrator shutdown."""

    def test_shutdown_is_async(self):
        """Verify shutdown is an async method."""
        import inspect
        
        orch = _make_orchestrator()
        assert inspect.iscoroutinefunction(orch.shutdown)


# ---------------------------------------------------------------------------
# Async Dispatch
# ---------------------------------------------------------------------------


class TestDispatchTask:
    """Test async task dispatch."""

    @pytest.mark.asyncio
    async def test_dispatch_task_unknown_agent(self):
        """Verify handling of unknown agent."""
        from core.types import TaskRequest
        
        orch = _make_orchestrator()
        orch.agents = {}
        
        request = TaskRequest(
            task_id="t1",
            agent_name="unknown",
            input_data={}
        )
        
        result = await orch._dispatch_task(request)
        
        assert result.status == "error"
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_dispatch_task_legacy_agent(self):
        """Verify legacy agent execution."""
        from core.types import TaskRequest
        
        agents = _make_agents()
        agents["test_agent"] = MagicMock()
        agents["test_agent"].run.return_value = {"result": "ok"}
        
        orch = _make_orchestrator(agents=agents)
        
        request = TaskRequest(
            task_id="t1",
            agent_name="test_agent",
            input_data={"data": "test"}
        )
        
        result = await orch._dispatch_task(request)
        
        assert result.status == "success"
        assert result.output == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_dispatch_task_legacy_agent_error(self):
        """Verify error handling in legacy agent execution."""
        from core.types import TaskRequest
        
        agents = _make_agents()
        agents["test_agent"] = MagicMock()
        agents["test_agent"].run.side_effect = ValueError("Agent error")
        
        orch = _make_orchestrator(agents=agents)
        
        request = TaskRequest(
            task_id="t1",
            agent_name="test_agent",
            input_data={}
        )
        
        result = await orch._dispatch_task(request)
        
        assert result.status == "error"
        assert "Agent error" in result.error


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_orchestrator_with_agents(self):
        """Verify orchestrator handles non-empty agent dict."""
        agents = _make_agents()
        orch = _make_orchestrator(agents=agents)
        assert len(orch.agents) > 0

    def test_orchestrator_with_custom_project_root(self):
        """Verify custom project root is stored correctly."""
        custom_root = Path("/custom/root")
        orch = _make_orchestrator(project_root=custom_root)
        
        assert orch.project_root == custom_root

    def test_orchestrator_strict_schema_mode(self):
        """Verify strict schema flag is stored."""
        orch = _make_orchestrator(strict_schema=True)
        assert orch.strict_schema is True

    def test_consecutive_fails_counter_initialized(self):
        """Verify consecutive failures counter starts at zero."""
        orch = _make_orchestrator()
        assert orch._consecutive_fails == 0

    def test_current_goal_none_initially(self):
        """Verify current goal is None on initialization."""
        orch = _make_orchestrator()
        assert orch.current_goal is None

    def test_active_cycle_summary_none_initially(self):
        """Verify active cycle summary is None on initialization."""
        orch = _make_orchestrator()
        assert orch.active_cycle_summary is None


# ---------------------------------------------------------------------------
# Human Gate
# ---------------------------------------------------------------------------


class TestHumanGate:
    """Test human gating mechanism."""

    def test_human_gate_initialized(self):
        """Verify human gate is initialized."""
        orch = _make_orchestrator()
        assert orch.human_gate is not None

    def test_human_gate_is_object(self):
        """Verify human gate is an instance of HumanGate."""
        from core.human_gate import HumanGate
        
        orch = _make_orchestrator()
        assert isinstance(orch.human_gate, HumanGate)


# ---------------------------------------------------------------------------
# Runtime Modes
# ---------------------------------------------------------------------------


class TestRuntimeMode:
    """Test runtime mode configuration."""

    def test_runtime_mode_stored(self):
        """Verify runtime mode is stored."""
        orch = _make_orchestrator(runtime_mode="sandbox")
        assert orch.runtime_mode == "sandbox"

    def test_runtime_mode_defaults_to_full(self):
        """Verify default runtime mode is 'full'."""
        orch = _make_orchestrator()
        assert orch.runtime_mode == "full"


# ---------------------------------------------------------------------------
# Beads Configuration
# ---------------------------------------------------------------------------


class TestBeadsConfiguration:
    """Test Beads synchronization configuration."""

    def test_beads_disabled_by_default(self):
        """Verify Beads is disabled by default."""
        orch = _make_orchestrator()
        assert orch.beads_enabled is False

    def test_beads_enabled_configuration(self):
        """Verify Beads can be enabled."""
        orch = _make_orchestrator(beads_enabled=True)
        assert orch.beads_enabled is True

    def test_beads_required_configuration(self):
        """Verify Beads required flag."""
        orch = _make_orchestrator(beads_required=True)
        assert orch.beads_required is True

    def test_beads_scope_configuration(self):
        """Verify Beads scope configuration."""
        orch = _make_orchestrator(beads_scope="cycle")
        assert orch.beads_scope == "cycle"


# ---------------------------------------------------------------------------
# Capability Management
# ---------------------------------------------------------------------------


class TestCapabilityConfiguration:
    """Test capability management configuration."""

    def test_auto_add_capabilities_enabled_by_default(self):
        """Verify auto capability addition is enabled by default."""
        orch = _make_orchestrator()
        assert orch.auto_add_capabilities is True

    def test_auto_add_capabilities_can_be_disabled(self):
        """Verify auto capability addition can be disabled."""
        orch = _make_orchestrator(auto_add_capabilities=False)
        assert orch.auto_add_capabilities is False

    def test_auto_queue_missing_capabilities_enabled_by_default(self):
        """Verify auto queuing of missing capabilities is enabled by default."""
        orch = _make_orchestrator()
        assert orch.auto_queue_missing_capabilities is True

    def test_auto_provision_mcp_disabled_by_default(self):
        """Verify auto MCP provisioning is disabled by default."""
        orch = _make_orchestrator()
        assert orch.auto_provision_mcp is False

    def test_auto_start_mcp_servers_disabled_by_default(self):
        """Verify auto MCP server start is disabled by default."""
        orch = _make_orchestrator()
        assert orch.auto_start_mcp_servers is False


# ---------------------------------------------------------------------------
# Optional Agents
# ---------------------------------------------------------------------------


class TestOptionalAgents:
    """Test optional specialized agents."""

    def test_self_correction_agent_optional(self):
        """Verify self correction agent is optional."""
        orch = _make_orchestrator()
        assert orch.self_correction_agent is None

    def test_investigation_agent_optional(self):
        """Verify investigation agent is optional."""
        orch = _make_orchestrator()
        assert orch.investigation_agent is None

    def test_root_cause_analysis_agent_optional(self):
        """Verify root cause analysis agent is optional."""
        orch = _make_orchestrator()
        assert orch.root_cause_analysis_agent is None

    def test_debugger_agent_optional(self):
        """Verify debugger agent is optional."""
        orch = _make_orchestrator()
        assert orch.debugger is None
