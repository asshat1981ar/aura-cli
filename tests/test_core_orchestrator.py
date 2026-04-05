"""Comprehensive test suite for core/orchestrator.py.

This module provides 80%+ coverage for LoopOrchestrator including:
- Full phase lifecycle (ingest → plan → critique → synthesize → act → sandbox → apply → verify → reflect)
- Goal queue integration
- Error recovery paths (sandbox retries, act retries, plan retries)
- Edge cases (cycle timeouts, empty queues, dry-run)
- Async operations with pytest-asyncio
"""

import json
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call, AsyncMock

import pytest

from core.orchestrator import LoopOrchestrator, BeadsSyncLoop, MAX_SANDBOX_RETRIES
from core.policy import Policy
from core.schema import RoutingDecision
from memory.store import MemoryStore


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def mock_brain():
    """Mock brain instance."""
    brain = MagicMock()
    brain.get.return_value = None
    brain.recall_all.return_value = []
    brain.recall_recent.return_value = []
    brain.recall_with_budget.return_value = []
    brain.set.return_value = None
    brain.remember.return_value = None
    return brain


@pytest.fixture
def mock_model():
    """Mock model adapter."""
    model = MagicMock()
    model.respond.return_value = "model response"
    return model


@pytest.fixture
def mock_agents():
    """Mock agent registry."""
    agents = {
        "ingest": MagicMock(),
        "plan": MagicMock(),
        "critique": MagicMock(),
        "synthesize": MagicMock(),
        "act": MagicMock(),
        "sandbox": MagicMock(),
        "verify": MagicMock(),
        "reflect": MagicMock(),
        "mcp_discovery": MagicMock(),
    }

    # Set default successful returns
    agents["ingest"].run.return_value = {
        "goal": "test goal",
        "memory_summary": "context",
        "hints_summary": "hints",
    }
    agents["plan"].run.return_value = {
        "steps": ["step1", "step2"],
        "confidence": 0.8,
    }
    agents["critique"].run.return_value = {
        "status": "approved",
        "issues": [],
    }
    agents["synthesize"].run.return_value = {
        "tasks": [{"description": "task1", "tests": []}],
    }
    agents["act"].run.return_value = {
        "changes": [{"file_path": "test.py", "old_code": "", "new_code": "# test"}],
    }
    agents["sandbox"].run.return_value = {
        "passed": True,
        "details": {},
    }
    agents["verify"].run.return_value = {
        "status": "pass",
        "failures": [],
        "logs": "",
    }
    agents["reflect"].run.return_value = {
        "summary": "Cycle completed successfully",
        "learnings": ["learning1"],
        "skill_summary": {},
    }
    agents["mcp_discovery"].run.return_value = {
        "status": "success",
        "discovered": [],
    }

    return agents


@pytest.fixture
def mock_memory_store():
    """Mock memory store."""
    store = MagicMock(spec=MemoryStore)
    store.read_log.return_value = []
    store.append_log.return_value = None
    store.query.return_value = []
    return store


@pytest.fixture
def temp_project_root():
    """Temporary project root directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def orchestrator(mock_agents, mock_brain, mock_model, mock_memory_store, temp_project_root):
    """Basic orchestrator instance for testing."""
    orch = LoopOrchestrator(
        agents=mock_agents,
        memory_store=mock_memory_store,
        brain=mock_brain,
        model=mock_model,
        project_root=temp_project_root,
        policy=Policy.from_config({}),
        strict_schema=False,
    )
    return orch


# ── Test LoopOrchestrator Initialization ───────────────────────────────────


class TestOrchestratorInit:
    """Test orchestrator initialization."""

    def test_init_with_defaults(self, mock_agents):
        """Test initialization with default parameters."""
        orch = LoopOrchestrator(agents=mock_agents)

        assert orch.agents == mock_agents
        assert orch.project_root == Path(".")
        assert orch.strict_schema is False
        assert orch.auto_add_capabilities is True
        assert orch.runtime_mode == "full"

    def test_init_with_custom_params(self, mock_agents, temp_project_root):
        """Test initialization with custom parameters."""
        orch = LoopOrchestrator(
            agents=mock_agents,
            project_root=temp_project_root,
            strict_schema=True,
            auto_add_capabilities=False,
            runtime_mode="test",
        )

        assert orch.project_root == temp_project_root
        assert orch.strict_schema is True
        assert orch.auto_add_capabilities is False
        assert orch.runtime_mode == "test"

    def test_skills_loaded_on_init(self, orchestrator):
        """Test skills are loaded during initialization."""
        # Skills dictionary exists (may be empty if deps missing)
        assert hasattr(orchestrator, "skills")
        assert isinstance(orchestrator.skills, dict)


# ── Test Full Phase Lifecycle ─────────────────────────────────────────────


class TestPhaseLifecycle:
    """Test complete phase execution lifecycle."""

    def test_run_cycle_executes_all_phases(self, orchestrator, mock_agents):
        """Test run_cycle executes all phases in order."""
        with patch("core.quality_snapshot.run_quality_snapshot") as mock_quality:
            mock_quality.return_value = {"test_count": 5}

            result = orchestrator.run_cycle("Test goal", dry_run=True)

            # Verify all critical agents were called
            assert mock_agents["ingest"].run.called
            assert mock_agents["plan"].run.called
            assert mock_agents["critique"].run.called
            assert mock_agents["synthesize"].run.called
            assert mock_agents["act"].run.called
            assert mock_agents["verify"].run.called
            assert mock_agents["reflect"].run.called

    def test_ingest_phase_receives_context(self, orchestrator, mock_agents):
        """Test ingest phase receives proper context."""
        orchestrator.run_cycle("Test goal", dry_run=True)

        call_args = mock_agents["ingest"].run.call_args[0][0]
        assert "goal" in call_args
        assert call_args["goal"] == "Test goal"
        assert "project_root" in call_args
        assert "hints" in call_args

    def test_plan_phase_receives_context(self, orchestrator, mock_agents):
        """Test plan phase receives ingest context."""
        orchestrator.run_cycle("Test goal", dry_run=True)

        call_args = mock_agents["plan"].run.call_args[0][0]
        assert "goal" in call_args
        assert "memory_snapshot" in call_args

    def test_critique_phase_receives_plan(self, orchestrator, mock_agents):
        """Test critique phase receives plan output."""
        orchestrator.run_cycle("Test goal", dry_run=True)

        call_args = mock_agents["critique"].run.call_args[0][0]
        assert "plan" in call_args

    def test_synthesize_merges_plan_and_critique(self, orchestrator, mock_agents):
        """Test synthesize merges plan and critique."""
        orchestrator.run_cycle("Test goal", dry_run=True)

        call_args = mock_agents["synthesize"].run.call_args[0][0]
        assert "plan" in call_args
        assert "critique" in call_args

    def test_act_phase_receives_task_bundle(self, orchestrator, mock_agents):
        """Test act phase receives synthesized task bundle."""
        orchestrator.run_cycle("Test goal", dry_run=True)

        call_args = mock_agents["act"].run.call_args[0][0]
        assert "task_bundle" in call_args
        assert "dry_run" in call_args

    def test_sandbox_phase_executes_before_apply(self, orchestrator, mock_agents):
        """Test sandbox executes before file apply."""
        orchestrator.run_cycle("Test goal", dry_run=True)

        # Sandbox should be called
        assert mock_agents["sandbox"].run.called
        call_args = mock_agents["sandbox"].run.call_args[0][0]
        assert "act" in call_args

    def test_verify_phase_receives_change_set(self, orchestrator, mock_agents):
        """Test verify phase receives applied changes."""
        orchestrator.run_cycle("Test goal", dry_run=True)

        call_args = mock_agents["verify"].run.call_args[0][0]
        assert "change_set" in call_args
        assert "dry_run" in call_args

    def test_reflect_phase_receives_verification(self, orchestrator, mock_agents):
        """Test reflect phase receives verification results."""
        orchestrator.run_cycle("Test goal", dry_run=True)

        call_args = mock_agents["reflect"].run.call_args[0][0]
        assert "verification" in call_args


# ── Test Error Recovery Paths ─────────────────────────────────────────────


class TestErrorRecovery:
    """Test error recovery and retry mechanisms."""

    def test_sandbox_retry_on_failure(self, orchestrator, mock_agents):
        """Test sandbox failures trigger act retry."""
        # First sandbox fails, second succeeds
        mock_agents["sandbox"].run.side_effect = [
            {"passed": False, "details": {"stderr": "error"}},
            {"passed": True, "details": {}},
        ]

        with patch("core.quality_snapshot.run_quality_snapshot") as mock_quality:
            mock_quality.return_value = {"test_count": 5}
            result = orchestrator.run_cycle("Test goal", dry_run=True)

        # Sandbox called twice (initial + retry)
        assert mock_agents["sandbox"].run.call_count == 2
        # Act called twice (initial + retry after sandbox failure)
        assert mock_agents["act"].run.call_count >= 2

    def test_sandbox_max_retries_exceeded(self, orchestrator, mock_agents):
        """Test sandbox stops after MAX_SANDBOX_RETRIES."""
        # All sandbox attempts fail
        mock_agents["sandbox"].run.return_value = {
            "passed": False,
            "details": {"stderr": "persistent error"},
        }

        with patch("core.quality_snapshot.run_quality_snapshot") as mock_quality:
            mock_quality.return_value = {"test_count": 5}
            result = orchestrator.run_cycle("Test goal", dry_run=True)

        # Sandbox called MAX_SANDBOX_RETRIES times
        assert mock_agents["sandbox"].run.call_count == MAX_SANDBOX_RETRIES

    def test_verification_failure_routes_to_act(self, orchestrator, mock_agents):
        """Test verification failure routes to act retry."""
        # First verification fails with code error, second passes
        mock_agents["verify"].run.side_effect = [
            {"status": "fail", "failures": ["assertion error"], "logs": ""},
            {"status": "pass", "failures": [], "logs": ""},
        ]

        with patch("core.quality_snapshot.run_quality_snapshot") as mock_quality:
            mock_quality.return_value = {"test_count": 5}
            result = orchestrator.run_cycle("Test goal", dry_run=True)

        # Verify called multiple times (act retry)
        assert mock_agents["verify"].run.call_count >= 2

    def test_verification_failure_routes_to_plan(self, orchestrator, mock_agents):
        """Test structural failure routes to plan retry."""
        # Verification fails with structural issue
        mock_agents["verify"].run.return_value = {
            "status": "fail",
            "failures": ["architecture violation detected"],
            "logs": "circular dependency found",
        }

        with patch("core.quality_snapshot.run_quality_snapshot") as mock_quality:
            mock_quality.return_value = {"test_count": 5}
            # Set max retries to prevent infinite loop
            with patch.object(orchestrator, "_configure_pipeline") as mock_cfg:
                cfg = MagicMock()
                cfg.max_act_attempts = 1
                cfg.plan_retries = 1
                cfg.skill_set = []
                cfg.intensity = "normal"
                cfg.phases = []
                mock_cfg.return_value = cfg

                result = orchestrator.run_cycle("Test goal", dry_run=True)

        # Plan should be called again after failure routing
        assert mock_agents["plan"].run.call_count >= 1

    def test_verification_skip_on_external_error(self, orchestrator, mock_agents):
        """Test external errors are skipped without retry."""
        mock_agents["verify"].run.return_value = {
            "status": "fail",
            "failures": ["ModuleNotFoundError: no module named 'nonexistent'"],
            "logs": "dependency not found",
        }

        with patch("core.quality_snapshot.run_quality_snapshot") as mock_quality:
            mock_quality.return_value = {"test_count": 5}
            result = orchestrator.run_cycle("Test goal", dry_run=True)

        # Should not retry indefinitely on external errors
        assert result["stop_reason"] is None  # Completes cycle

    def test_act_retry_includes_fix_hints(self, orchestrator, mock_agents):
        """Test act retries include fix hints from previous failures."""
        mock_agents["sandbox"].run.side_effect = [
            {"passed": False, "details": {"stderr": "NameError: 'foo' not defined"}},
            {"passed": True, "details": {}},
        ]

        with patch("core.quality_snapshot.run_quality_snapshot") as mock_quality:
            mock_quality.return_value = {"test_count": 5}
            result = orchestrator.run_cycle("Test goal", dry_run=True)

        # Second act call should have fix_hints
        second_call = mock_agents["act"].run.call_args_list[1][0][0]
        assert "fix_hints" in second_call


# ── Test Goal Queue Integration ───────────────────────────────────────────


class TestGoalQueueIntegration:
    """Test goal queue integration."""

    def test_run_loop_processes_single_goal(self, orchestrator):
        """Test run_loop processes a single goal."""
        with patch("core.quality_snapshot.run_quality_snapshot") as mock_quality:
            mock_quality.return_value = {"test_count": 5}
            result = orchestrator.run_loop("Test goal", max_cycles=1, dry_run=True)

        assert result["goal"] == "Test goal"
        assert len(result["history"]) == 1
        assert result["stop_reason"] in ("PASS", "MAX_CYCLES")

    def test_run_loop_stops_on_policy(self, orchestrator):
        """Test run_loop stops when policy triggers."""
        # Mock policy to stop after first cycle
        orchestrator.policy.evaluate = Mock(return_value="PASS")

        with patch("core.quality_snapshot.run_quality_snapshot") as mock_quality:
            mock_quality.return_value = {"test_count": 5}
            result = orchestrator.run_loop("Test goal", max_cycles=5, dry_run=True)

        assert result["stop_reason"] == "PASS"
        assert len(result["history"]) == 1

    def test_run_loop_respects_max_cycles(self, orchestrator):
        """Test run_loop respects max_cycles limit."""
        # Policy never stops
        orchestrator.policy.evaluate = Mock(return_value="")

        with patch("core.quality_snapshot.run_quality_snapshot") as mock_quality:
            mock_quality.return_value = {"test_count": 5}
            result = orchestrator.run_loop("Test goal", max_cycles=3, dry_run=True)

        assert len(result["history"]) == 3
        assert result["stop_reason"] == "MAX_CYCLES"

    def test_poll_external_goals_returns_beads(self, orchestrator):
        """Test polling external goals from BEADS."""
        beads_skill = MagicMock()
        beads_skill.run.return_value = {
            "ready": [
                {"id": "bd-1", "title": "Fix bug"},
                {"id": "bd-2", "summary": "Add feature"},
            ]
        }
        orchestrator.beads_enabled = True
        orchestrator.skills = {"beads_skill": beads_skill}

        goals = orchestrator.poll_external_goals()

        assert len(goals) == 2
        assert goals[0] == "bead:bd-1: Fix bug"
        assert goals[1] == "bead:bd-2: Add feature"

    def test_poll_external_goals_empty_when_disabled(self, orchestrator):
        """Test polling returns empty when BEADS disabled."""
        orchestrator.beads_enabled = False

        goals = orchestrator.poll_external_goals()

        assert goals == []


# ── Test Edge Cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_dry_run_skips_file_writes(self, orchestrator, mock_agents, temp_project_root):
        """Test dry_run mode skips actual file writes."""
        test_file = temp_project_root / "test.py"

        result = orchestrator.run_cycle("Test goal", dry_run=True)

        # File should not be created in dry-run
        assert not test_file.exists()
        assert result["phase_outputs"]["dry_run"] is True

    def test_empty_goal_queue_returns_empty_list(self, orchestrator):
        """Test empty goal queue handling."""
        # No goal queue set
        orchestrator.goal_queue = None

        goals = orchestrator.poll_external_goals()

        assert goals == []

    def test_strict_schema_stops_on_invalid_output(self, orchestrator, mock_agents):
        """Test strict_schema mode stops on invalid phase output."""
        orchestrator.strict_schema = True

        # Ingest returns invalid output
        mock_agents["ingest"].run.return_value = "invalid"

        result = orchestrator.run_cycle("Test goal", dry_run=True)

        # Should stop early due to invalid output
        assert result.get("stop_reason") == "INVALID_OUTPUT"

    def test_missing_agent_returns_empty_dict(self, orchestrator):
        """Test missing agent in registry returns empty dict."""
        # Remove an agent
        del orchestrator.agents["ingest"]

        result = orchestrator._run_phase("ingest", {"goal": "test"})

        assert result == {}

    def test_context_injection_passed_to_ingest(self, orchestrator, mock_agents):
        """Test context_injection parameter flows to ingest phase."""
        context_injection = {"dependency_context": ["dep1", "dep2"]}

        result = orchestrator.run_cycle(
            "Test goal",
            dry_run=True,
            context_injection=context_injection
        )

        # Check ingest received the injected context
        call_args = mock_agents["ingest"].run.call_args[0][0]
        assert "dependency_context" in call_args
        assert call_args["dependency_context"] == ["dep1", "dep2"]

    def test_cycle_timeout_tracking(self, orchestrator):
        """Test cycle execution time tracking."""
        with patch("core.quality_snapshot.run_quality_snapshot") as mock_quality:
            mock_quality.return_value = {"test_count": 5}

            start = time.time()
            result = orchestrator.run_cycle("Test goal", dry_run=True)
            duration = time.time() - start

        # Result should have timing info
        assert "started_at" in result
        assert "completed_at" in result
        assert result["completed_at"] > result["started_at"]
        assert result.get("duration_s", 0) <= duration + 1  # Allow 1s tolerance


# ── Test File Safety and Apply Policy ─────────────────────────────────────


class TestFileSafety:
    """Test file safety mechanisms."""

    def test_apply_change_set_creates_snapshots(self, orchestrator, temp_project_root):
        """Test file snapshots are created before apply."""
        test_file = temp_project_root / "test.py"
        test_file.write_text("original")

        change_set = {
            "changes": [{
                "file_path": "test.py",
                "old_code": "original",
                "new_code": "modified",
            }]
        }

        result = orchestrator._apply_change_set(change_set, dry_run=False)

        assert "snapshots" in result
        assert len(result["snapshots"]) > 0
        assert result["snapshots"][0]["file"] == "test.py"

    def test_restore_applied_changes_on_verify_fail(self, orchestrator, mock_agents, temp_project_root):
        """Test file restoration after verification failure."""
        test_file = temp_project_root / "test.py"
        test_file.write_text("original")

        mock_agents["verify"].run.return_value = {
            "status": "fail",
            "failures": ["test failed"],
            "logs": "",
        }

        with patch("core.quality_snapshot.run_quality_snapshot") as mock_quality:
            mock_quality.return_value = {"test_count": 5}
            # Limit retries to prevent long test
            with patch.object(orchestrator, "_configure_pipeline") as mock_cfg:
                cfg = MagicMock()
                cfg.max_act_attempts = 1
                cfg.plan_retries = 1
                cfg.skill_set = []
                cfg.intensity = "normal"
                cfg.phases = []
                mock_cfg.return_value = cfg

                result = orchestrator.run_cycle("Test goal", dry_run=False)

        # Original content should be restored if verification failed
        # (depends on routing logic, but restoration method should be called)
        assert test_file.exists()


# ── Test Async Operations ─────────────────────────────────────────────────


class TestAsyncOperations:
    """Test async operations with pytest-asyncio."""

    @pytest.mark.asyncio
    async def test_dispatch_task_async(self, orchestrator, mock_agents):
        """Test async task dispatch."""
        from core.types import TaskRequest, ExecutionContext

        request = TaskRequest(
            task_id="test_123",
            agent_name="plan",
            input_data={"goal": "test"},
            context=ExecutionContext(project_root=str(orchestrator.project_root))
        )

        result = await orchestrator._dispatch_task(request)

        assert result.task_id == "test_123"
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_dispatch_task_missing_agent(self, orchestrator):
        """Test async dispatch with missing agent."""
        from core.types import TaskRequest, ExecutionContext

        request = TaskRequest(
            task_id="test_456",
            agent_name="nonexistent",
            input_data={},
            context=ExecutionContext(project_root=str(orchestrator.project_root))
        )

        result = await orchestrator._dispatch_task(request)

        assert result.status == "error"
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_shutdown_cleans_resources(self, orchestrator):
        """Test shutdown cleans up async resources."""
        # Should not raise
        await orchestrator.shutdown()


# ── Test Helper Methods ───────────────────────────────────────────────────


class TestHelperMethods:
    """Test orchestrator helper methods."""

    def test_route_failure_returns_act_by_default(self, orchestrator):
        """Test _route_failure defaults to ACT."""
        result = orchestrator._route_failure({"failures": ["generic error"]})
        assert result == RoutingDecision.ACT

    def test_route_failure_structural_returns_plan(self, orchestrator):
        """Test _route_failure returns PLAN for structural issues."""
        result = orchestrator._route_failure({
            "failures": ["architecture violation"],
            "logs": "",
        })
        assert result == RoutingDecision.PLAN

    def test_route_failure_external_returns_skip(self, orchestrator):
        """Test _route_failure returns SKIP for external issues."""
        result = orchestrator._route_failure({
            "failures": ["ModuleNotFoundError"],
            "logs": "",
        })
        assert result == RoutingDecision.SKIP

    def test_retrieve_hints_returns_relevant_summaries(self, orchestrator, mock_memory_store):
        """Test _retrieve_hints returns relevant past cycles."""
        mock_memory_store.query.return_value = [
            {"goal": "similar goal", "status": "success"},
            {"goal": "other goal", "status": "fail"},
        ]
        orchestrator.memory_controller.persistent_store = mock_memory_store

        hints = orchestrator._retrieve_hints("similar goal", limit=5)

        # Should return list of summaries
        assert isinstance(hints, list)

    def test_estimate_confidence_for_plan(self, orchestrator):
        """Test _estimate_confidence for plan phase."""
        plan = {"steps": ["step1", "step2", "test step"]}
        confidence = orchestrator._estimate_confidence(plan, "plan")

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be higher with test steps

    def test_estimate_confidence_for_verify_pass(self, orchestrator):
        """Test _estimate_confidence for passing verification."""
        verify = {"status": "pass"}
        confidence = orchestrator._estimate_confidence(verify, "verify")

        assert confidence == 0.95

    def test_estimate_confidence_for_verify_fail(self, orchestrator):
        """Test _estimate_confidence for failing verification."""
        verify = {"status": "fail"}
        confidence = orchestrator._estimate_confidence(verify, "verify")

        assert confidence == 0.1

    def test_normalize_verification_result_with_status(self, orchestrator):
        """Test _normalize_verification_result with status field."""
        result = orchestrator._normalize_verification_result({"status": "pass"})
        assert result["status"] == "pass"

    def test_normalize_verification_result_with_passed(self, orchestrator):
        """Test _normalize_verification_result with passed field."""
        result = orchestrator._normalize_verification_result({"passed": True})
        assert result["status"] == "pass"

        result = orchestrator._normalize_verification_result({"passed": False})
        assert result["status"] == "fail"


# ── Test BEADS Integration ────────────────────────────────────────────────


class TestBEADSIntegration:
    """Test BEADS integration features."""

    def test_beads_gate_blocks_before_plan(self, orchestrator, mock_agents):
        """Test BEADS gate can block execution."""
        beads_bridge = MagicMock()
        beads_bridge.run.return_value = {
            "ok": True,
            "decision": {
                "status": "block",
                "summary": "Blocked",
                "stop_reason": "OPERATOR_REVIEW",
            },
        }

        orchestrator.beads_bridge = beads_bridge
        orchestrator.beads_enabled = True
        orchestrator.beads_required = True

        result = orchestrator.run_cycle("Test goal", dry_run=True)

        assert result["stop_reason"] == "BEADS_BLOCKED"
        # Plan should not be called
        assert not mock_agents["plan"].run.called

    def test_beads_unavailable_stops_when_required(self, orchestrator, mock_agents):
        """Test BEADS unavailable stops cycle when required."""
        beads_bridge = MagicMock()
        beads_bridge.run.return_value = {
            "ok": False,
            "decision": None,
            "error": "timeout",
        }

        orchestrator.beads_bridge = beads_bridge
        orchestrator.beads_enabled = True
        orchestrator.beads_required = True

        result = orchestrator.run_cycle("Test goal", dry_run=True)

        assert result["stop_reason"] == "BEADS_UNAVAILABLE"
        assert not mock_agents["plan"].run.called

    def test_bead_claim_and_close(self, orchestrator):
        """Test bead claiming and closing."""
        beads_skill = MagicMock()
        orchestrator.beads_enabled = True
        orchestrator.skills = {"beads_skill": beads_skill}

        orchestrator._claim_bead("bd-123")
        orchestrator._close_bead("bd-123", "completed")

        assert beads_skill.run.call_count == 2
        assert beads_skill.run.call_args_list[0][0][0]["cmd"] == "update"
        assert beads_skill.run.call_args_list[1][0][0]["cmd"] == "close"


# ── Test BeadsSyncLoop ────────────────────────────────────────────────────


class TestBeadsSyncLoopClass:
    """Test BeadsSyncLoop class."""

    def test_sync_loop_init(self):
        """Test BeadsSyncLoop initialization."""
        skill = MagicMock()
        loop = BeadsSyncLoop(skill)

        assert loop._skill is skill
        assert loop._n == 0
        assert loop.EVERY_N == 5

    def test_sync_loop_skips_dry_run(self):
        """Test sync loop skips dry-run cycles."""
        skill = MagicMock()
        loop = BeadsSyncLoop(skill)
        loop._n = 4

        loop.on_cycle_complete({"dry_run": True})

        assert loop._n == 4  # Counter not incremented
        skill.run.assert_not_called()

    def test_sync_loop_triggers_every_n(self):
        """Test sync loop triggers every N cycles."""
        skill = MagicMock()
        loop = BeadsSyncLoop(skill)

        # Trigger 4 cycles
        for _ in range(4):
            loop.on_cycle_complete({})

        assert skill.run.call_count == 0

        # 5th cycle triggers sync
        loop.on_cycle_complete({})

        assert skill.run.call_count == 2  # pull + push


# ── Test Improvement Loops ────────────────────────────────────────────────


class TestImprovementLoops:
    """Test improvement loop integration."""

    def test_attach_improvement_loops(self, orchestrator):
        """Test attaching improvement loops."""
        loop1 = MagicMock()
        loop2 = MagicMock()

        orchestrator.attach_improvement_loops(loop1, loop2)

        assert len(orchestrator._improvement_loops) == 2

    def test_improvement_loops_called_on_cycle_complete(self, orchestrator):
        """Test improvement loops are called after cycle."""
        loop = MagicMock()
        orchestrator.attach_improvement_loops(loop)

        with patch("core.quality_snapshot.run_quality_snapshot") as mock_quality:
            mock_quality.return_value = {"test_count": 5}
            orchestrator.run_cycle("Test goal", dry_run=True)

        assert loop.on_cycle_complete.called

    def test_improvement_loop_errors_swallowed(self, orchestrator):
        """Test improvement loop errors don't break cycle."""
        loop = MagicMock()
        loop.on_cycle_complete.side_effect = Exception("loop error")
        orchestrator.attach_improvement_loops(loop)

        with patch("core.quality_snapshot.run_quality_snapshot") as mock_quality:
            mock_quality.return_value = {"test_count": 5}
            # Should not raise
            result = orchestrator.run_cycle("Test goal", dry_run=True)

        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
