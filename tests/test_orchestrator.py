"""Unit tests for core/orchestrator.py — LoopOrchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

from core.orchestrator import LoopOrchestrator, BeadsSyncLoop
from core.policy import Policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agents(**overrides):
    """Return a minimal agents dict with MagicMock instances for every phase."""
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
    """Build a LoopOrchestrator with all heavy side-effects patched out."""
    agents = agents or _make_agents()

    # Patch everything that touches the filesystem or external services at
    # construction time so tests are hermetic.
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


def _patch_run_cycle(orchestrator, return_value):
    """Patch run_cycle on *orchestrator* to return a fixed value."""
    orchestrator.run_cycle = MagicMock(return_value=return_value)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestLoopOrchestratorInit:
    def test_agents_stored(self):
        agents = _make_agents()
        orch = _make_orchestrator(agents=agents)
        assert orch.agents is agents

    def test_default_policy_is_sliding_window(self):
        orch = _make_orchestrator()
        assert orch.policy is not None

    def test_project_root_stored(self):
        orch = _make_orchestrator(project_root=Path("/some/path"))
        assert orch.project_root == Path("/some/path")

    def test_strict_schema_default_false(self):
        orch = _make_orchestrator()
        assert orch.strict_schema is False

    def test_strict_schema_true(self):
        orch = _make_orchestrator(strict_schema=True)
        assert orch.strict_schema is True


# ---------------------------------------------------------------------------
# run_loop — stopping conditions
# ---------------------------------------------------------------------------


class TestRunLoop:
    def _basic_cycle_entry(self, **extra):
        entry = {
            "cycle_id": "cycle_abc",
            # verification status "fail" so SlidingWindowPolicy doesn't stop early
            "phase_outputs": {"verification": {"status": "fail"}},
        }
        entry.update(extra)
        return entry

    def test_run_loop_returns_goal_and_history(self):
        orch = _make_orchestrator()
        entry = self._basic_cycle_entry()
        _patch_run_cycle(orch, entry)

        result = orch.run_loop("fix bug", max_cycles=1, dry_run=True)

        assert result["goal"] == "fix bug"
        assert len(result["history"]) == 1

    def test_run_loop_max_cycles_respected(self):
        orch = _make_orchestrator()
        _patch_run_cycle(orch, self._basic_cycle_entry())

        result = orch.run_loop("fix bug", max_cycles=3, dry_run=True)

        assert orch.run_cycle.call_count == 3
        assert result["stop_reason"] == "MAX_CYCLES"

    def test_run_loop_stops_on_cycle_stop_reason(self):
        orch = _make_orchestrator()
        entry = self._basic_cycle_entry(stop_reason="INVALID_OUTPUT")
        _patch_run_cycle(orch, entry)

        result = orch.run_loop("fix bug", max_cycles=10, dry_run=True)

        assert orch.run_cycle.call_count == 1
        assert result["stop_reason"] == "INVALID_OUTPUT"

    def test_run_loop_stops_on_policy_pass(self):
        orch = _make_orchestrator()
        # Use a passing verification to trigger SlidingWindowPolicy.PASS
        entry = {
            "cycle_id": "cycle_abc",
            "phase_outputs": {"verification": {"status": "pass"}},
        }
        _patch_run_cycle(orch, entry)

        result = orch.run_loop("feat", max_cycles=5, dry_run=True)

        assert result["stop_reason"] == "PASS"
        assert orch.run_cycle.call_count == 1

    def test_run_loop_max_cycles_sets_reason_on_last_entry(self):
        orch = _make_orchestrator()
        _patch_run_cycle(orch, self._basic_cycle_entry())
        orch.policy = MagicMock()
        orch.policy.evaluate.return_value = ""  # never stop early

        result = orch.run_loop("task", max_cycles=2, dry_run=True)

        assert result["history"][-1]["stop_reason"] == "MAX_CYCLES"

    def test_run_loop_dry_run_passed_to_cycle(self):
        orch = _make_orchestrator()
        _patch_run_cycle(orch, self._basic_cycle_entry())

        orch.run_loop("task", max_cycles=1, dry_run=True)

        orch.run_cycle.assert_called_with("task", dry_run=True, context_injection=None)

    def test_run_loop_empty_cycles_stop_reason_max_cycles(self):
        orch = _make_orchestrator()
        _patch_run_cycle(orch, self._basic_cycle_entry())

        result = orch.run_loop("task", max_cycles=0, dry_run=True)

        assert result["stop_reason"] == "MAX_CYCLES"
        assert result["history"] == []


# ---------------------------------------------------------------------------
# Improvement loops
# ---------------------------------------------------------------------------


class TestImprovementLoops:
    def test_attach_improvement_loops_registers_loop(self):
        orch = _make_orchestrator()
        loop_mock = MagicMock()
        orch.attach_improvement_loops(loop_mock)
        assert loop_mock in orch._improvement_loops

    def test_improvement_loop_called_via_post_cycle_hooks(self):
        """_run_post_cycle_hooks triggers on_cycle_complete on each loop."""
        orch = _make_orchestrator()
        loop_mock = MagicMock()
        orch.attach_improvement_loops(loop_mock)

        entry = {"cycle_id": "x", "phase_outputs": {}, "stop_reason": ""}
        with patch("core.orchestrator_learn.log_json"):
            orch._run_post_cycle_hooks(entry, "goal", "x", {}, False)

        loop_mock.on_cycle_complete.assert_called_once_with(entry)

    def test_improvement_loop_error_does_not_crash(self):
        orch = _make_orchestrator()
        bad_loop = MagicMock()
        bad_loop.on_cycle_complete.side_effect = RuntimeError("boom")
        orch.attach_improvement_loops(bad_loop)

        entry = {"cycle_id": "x", "phase_outputs": {}, "stop_reason": ""}
        with patch("core.orchestrator_learn.log_json"):
            # Should not raise
            orch._run_post_cycle_hooks(entry, "goal", "x", {}, False)


# ---------------------------------------------------------------------------
# BeadsSyncLoop
# ---------------------------------------------------------------------------


class TestBeadsSyncLoop:
    def test_push_pull_called_every_n_cycles(self):
        skill = MagicMock()
        loop = BeadsSyncLoop(skill)
        for _ in range(BeadsSyncLoop.EVERY_N):
            loop.on_cycle_complete({"dry_run": False})
        assert skill.run.call_count == 2  # pull + push

    def test_dry_run_skips_sync(self):
        skill = MagicMock()
        loop = BeadsSyncLoop(skill)
        for _ in range(BeadsSyncLoop.EVERY_N):
            loop.on_cycle_complete({"dry_run": True})
        skill.run.assert_not_called()


# ---------------------------------------------------------------------------
# _estimate_confidence
# ---------------------------------------------------------------------------


class TestEstimateConfidence:
    def setup_method(self):
        self.orch = _make_orchestrator()

    def test_verify_pass_returns_high_confidence(self):
        c = self.orch._estimate_confidence({"status": "pass"}, "verify")
        assert c >= 0.9

    def test_verify_fail_returns_low_confidence(self):
        c = self.orch._estimate_confidence({"status": "fail"}, "verify")
        assert c <= 0.2

    def test_plan_with_steps_raises_confidence(self):
        c = self.orch._estimate_confidence({"steps": ["a", "b"]}, "plan")
        assert c > 0.5

    def test_non_dict_output_baseline(self):
        c = self.orch._estimate_confidence("not a dict", "plan")
        assert c == 0.3

    def test_verify_skip_returns_mid_confidence(self):
        c = self.orch._estimate_confidence({"status": "skip"}, "verify")
        assert c == 0.6

    def test_act_with_changes_and_file_paths(self):
        output = {"changes": [{"file_path": "a.py"}, {"file_path": "b.py"}]}
        c = self.orch._estimate_confidence(output, "act")
        assert c > 0.7

    def test_act_with_empty_changes(self):
        c = self.orch._estimate_confidence({"changes": []}, "act")
        assert c == 0.5

    def test_critique_with_issues(self):
        c = self.orch._estimate_confidence({"issues": ["something bad"]}, "critique")
        assert c > 0.5

    def test_plan_with_test_step_boosts_confidence(self):
        c = self.orch._estimate_confidence(
            {"steps": ["write tests for auth", "implement auth"], "estimated_complexity": "low"},
            "plan",
        )
        # steps >= 2: +0.2, has "test": +0.1, estimated_complexity: +0.05 → 0.85
        assert c >= 0.85


# ---------------------------------------------------------------------------
# _apply_change_set
# ---------------------------------------------------------------------------


class TestApplyChangeSet:
    def setup_method(self):
        self.orch = _make_orchestrator()

    def test_dry_run_records_applied_without_writing(self):
        change_set = {"file_path": "foo.py", "old_code": "x=1", "new_code": "x=2"}
        result = self.orch._apply_change_set(change_set, dry_run=True)
        assert "foo.py" in result["applied"]
        assert result["failed"] == []

    def test_dry_run_batch_changes(self):
        change_set = {
            "changes": [
                {"file_path": "a.py", "old_code": "", "new_code": "x=1"},
                {"file_path": "b.py", "old_code": "", "new_code": "x=2"},
            ]
        }
        result = self.orch._apply_change_set(change_set, dry_run=True)
        assert set(result["applied"]) == {"a.py", "b.py"}
        assert result["failed"] == []

    def test_missing_file_path_skipped(self):
        change_set = {"changes": [{"old_code": "x", "new_code": "y"}]}
        result = self.orch._apply_change_set(change_set, dry_run=True)
        assert result["applied"] == []
        assert result["failed"] == []

    def test_mismatch_overwrite_blocked_error_goes_to_failed(self):
        from core.file_tools import MismatchOverwriteBlockedError

        change_set = {"file_path": "f.py", "old_code": "x", "new_code": "y"}
        exc = MismatchOverwriteBlockedError("blocked")
        with patch(
            "core.orchestrator.apply_change_with_explicit_overwrite_policy",
            side_effect=exc,
        ):
            self.orch._snapshot_file_state = MagicMock(return_value={})
            result = self.orch._apply_change_set(change_set, dry_run=False)
        assert result["failed"][0]["file"] == "f.py"
        assert result["applied"] == []

    def test_old_code_not_found_goes_to_failed(self):
        from core.file_tools import OldCodeNotFoundError

        change_set = {"file_path": "f.py", "old_code": "x", "new_code": "y"}
        with patch(
            "core.orchestrator.apply_change_with_explicit_overwrite_policy",
            side_effect=OldCodeNotFoundError("not found"),
        ):
            self.orch._snapshot_file_state = MagicMock(return_value={})
            result = self.orch._apply_change_set(change_set, dry_run=False)
        assert result["failed"][0]["file"] == "f.py"

    def test_generic_exception_goes_to_failed(self):
        change_set = {"file_path": "f.py", "old_code": "x", "new_code": "y"}
        with patch(
            "core.orchestrator.apply_change_with_explicit_overwrite_policy",
            side_effect=RuntimeError("boom"),
        ):
            self.orch._snapshot_file_state = MagicMock(return_value={})
            result = self.orch._apply_change_set(change_set, dry_run=False)
        assert result["failed"][0]["file"] == "f.py"
        assert "boom" in result["failed"][0]["error"]

    def test_empty_change_set_returns_empty(self):
        result = self.orch._apply_change_set({}, dry_run=False)
        assert result["applied"] == []
        assert result["failed"] == []
        assert result["snapshots"] == []


# ---------------------------------------------------------------------------
# _load_config_file
# ---------------------------------------------------------------------------


class TestLoadConfigFile:
    def test_returns_empty_dict_when_file_missing(self):
        orch = _make_orchestrator(project_root=Path("/nonexistent/path"))
        result = orch._load_config_file()
        assert result == {}

    def test_returns_parsed_dict_when_file_exists(self, tmp_path):
        cfg = {"n_best_candidates": 3, "hooks": []}
        (tmp_path / "aura.config.json").write_text('{"n_best_candidates": 3, "hooks": []}')
        orch = _make_orchestrator(project_root=tmp_path)
        result = orch._load_config_file()
        assert result == cfg

    def test_returns_empty_dict_on_invalid_json(self, tmp_path):
        (tmp_path / "aura.config.json").write_text("not valid json{{{")
        orch = _make_orchestrator(project_root=tmp_path)
        result = orch._load_config_file()
        assert result == {}


# ---------------------------------------------------------------------------
# _retrieve_hints
# ---------------------------------------------------------------------------


class TestRetrieveHints:
    def test_returns_empty_list_when_no_persistent_store(self):
        orch = _make_orchestrator()
        orch.memory_controller.persistent_store = None
        result = orch._retrieve_hints("fix auth bug")
        assert result == []

    def test_returns_empty_list_when_memory_controller_is_none(self):
        orch = _make_orchestrator()
        orch.memory_controller = None
        result = orch._retrieve_hints("fix auth bug")
        assert result == []

    def test_returns_empty_when_no_summaries(self):
        orch = _make_orchestrator()
        orch.memory_controller.persistent_store.query.return_value = []
        result = orch._retrieve_hints("fix auth bug")
        assert result == []

    def test_returns_ranked_and_limited_results(self):
        orch = _make_orchestrator()
        summaries = [
            {"goal": "fix auth bug", "status": "success"},
            {"goal": "unrelated task", "status": "fail"},
            {"goal": "auth token refresh", "status": "success"},
            {"goal": "database migration", "status": "success"},
            {"goal": "fix auth login", "status": "success"},
            {"goal": "add logging", "status": "fail"},
        ]
        orch.memory_controller.persistent_store.query.return_value = summaries
        result = orch._retrieve_hints("auth bug fix", limit=3)
        assert len(result) <= 3

    def test_handles_query_exception(self):
        orch = _make_orchestrator()
        orch.memory_controller.persistent_store.query.side_effect = OSError("DB down")
        result = orch._retrieve_hints("goal")
        assert result == []


# ---------------------------------------------------------------------------
# poll_external_goals
# ---------------------------------------------------------------------------


class TestPollExternalGoals:
    def test_returns_empty_when_beads_disabled(self):
        orch = _make_orchestrator()
        orch.beads_enabled = False
        result = orch.poll_external_goals()
        assert result == []

    def test_parses_list_result(self):
        orch = _make_orchestrator()
        orch.beads_enabled = True
        skill = MagicMock()
        skill.run.return_value = [{"id": "b1", "title": "Fix login bug"}]
        orch.skills = {"beads_skill": skill}
        result = orch.poll_external_goals()
        assert result == ["bead:b1: Fix login bug"]

    def test_parses_dict_with_beads_key(self):
        orch = _make_orchestrator()
        orch.beads_enabled = True
        skill = MagicMock()
        skill.run.return_value = {"beads": [{"id": "b2", "title": "Add tests"}]}
        orch.skills = {"beads_skill": skill}
        result = orch.poll_external_goals()
        assert result == ["bead:b2: Add tests"]

    def test_parses_dict_with_ready_key(self):
        orch = _make_orchestrator()
        orch.beads_enabled = True
        skill = MagicMock()
        skill.run.return_value = {"ready": [{"id": "b3", "summary": "Refactor utils"}]}
        orch.skills = {"beads_skill": skill}
        result = orch.poll_external_goals()
        assert result == ["bead:b3: Refactor utils"]

    def test_handles_exception_silently(self):
        orch = _make_orchestrator()
        orch.beads_enabled = True
        skill = MagicMock()
        skill.run.side_effect = RuntimeError("connection refused")
        orch.skills = {"beads_skill": skill}
        result = orch.poll_external_goals()
        assert result == []

    def test_bead_without_title_skipped(self):
        orch = _make_orchestrator()
        orch.beads_enabled = True
        skill = MagicMock()
        skill.run.return_value = [{"id": "b4"}]  # no title
        orch.skills = {"beads_skill": skill}
        result = orch.poll_external_goals()
        assert result == []


# ---------------------------------------------------------------------------
# UI callbacks
# ---------------------------------------------------------------------------


class TestUICallbacks:
    def test_attach_ui_callback_stores_callback(self):
        orch = _make_orchestrator()
        cb = MagicMock()
        orch.attach_ui_callback(cb)
        assert cb in orch._ui_callbacks

    def test_notify_ui_calls_method_on_registered_callback(self):
        orch = _make_orchestrator()
        cb = MagicMock()
        orch.attach_ui_callback(cb)
        orch._notify_ui("on_phase_start", "plan")
        cb.on_phase_start.assert_called_once_with("plan")

    def test_notify_ui_silently_ignores_callback_error(self):
        orch = _make_orchestrator()
        cb = MagicMock()
        cb.on_phase_start.side_effect = TypeError("bad arg")
        orch.attach_ui_callback(cb)
        orch._notify_ui("on_phase_start", "plan")  # should not raise

    def test_notify_ui_skips_missing_method(self):
        orch = _make_orchestrator()

        class MinimalCallback:
            pass

        orch.attach_ui_callback(MinimalCallback())
        orch._notify_ui("on_phase_start", "plan")  # should not raise


# ---------------------------------------------------------------------------
# attach_caspa
# ---------------------------------------------------------------------------


class TestAttachCaspa:
    def test_stores_all_three_components(self):
        orch = _make_orchestrator()
        ap = MagicMock()
        pe = MagicMock()
        cg = MagicMock()
        orch.attach_caspa(adaptive_pipeline=ap, propagation_engine=pe, context_graph=cg)
        assert orch.adaptive_pipeline is ap
        assert orch.propagation_engine is pe
        assert orch.context_graph is cg

    def test_handles_none_components(self):
        orch = _make_orchestrator()
        orch.attach_caspa(adaptive_pipeline=None, propagation_engine=None, context_graph=None)
        assert orch.adaptive_pipeline is None
        assert orch.propagation_engine is None
        assert orch.context_graph is None


# ---------------------------------------------------------------------------
# _route_failure (VerifyMixin)
# ---------------------------------------------------------------------------


class TestRouteFailure:
    def setup_method(self):
        self.orch = _make_orchestrator()

    def test_structural_signals_return_plan(self):
        from core.schema import RoutingDecision

        result = self.orch._route_failure({"failures": ["circular dependency detected"], "logs": ""})
        assert result == RoutingDecision.PLAN

    def test_external_signals_return_skip(self):
        from core.schema import RoutingDecision

        result = self.orch._route_failure({"failures": ["no module named requests"], "logs": ""})
        assert result == RoutingDecision.SKIP

    def test_default_returns_act(self):
        from core.schema import RoutingDecision

        result = self.orch._route_failure({"failures": ["assertion failed on line 42"], "logs": ""})
        assert result == RoutingDecision.ACT

    def test_structural_signal_in_logs(self):
        from core.schema import RoutingDecision

        result = self.orch._route_failure({"failures": [], "logs": "breaking_change in public API"})
        assert result == RoutingDecision.PLAN


# ---------------------------------------------------------------------------
# _normalize_verification_result (VerifyMixin)
# ---------------------------------------------------------------------------


class TestNormalizeVerificationResult:
    def setup_method(self):
        self.orch = _make_orchestrator()

    def test_non_dict_becomes_fail_status(self):
        result = self.orch._normalize_verification_result("bad output")
        assert result["status"] == "fail"
        assert "invalid verification payload" in result["failures"]

    def test_dict_with_status_returned_unchanged(self):
        v = {"status": "pass", "failures": [], "logs": ""}
        result = self.orch._normalize_verification_result(v)
        assert result is v

    def test_dict_with_passed_true_becomes_pass(self):
        result = self.orch._normalize_verification_result({"passed": True})
        assert result["status"] == "pass"

    def test_dict_with_passed_false_becomes_fail(self):
        result = self.orch._normalize_verification_result({"passed": False})
        assert result["status"] == "fail"

    def test_dict_without_status_or_passed_returned_as_is(self):
        v = {"details": "some info"}
        result = self.orch._normalize_verification_result(v)
        assert result is v


# ---------------------------------------------------------------------------
# _analyze_error / _run_root_cause_analysis (VerifyMixin)
# ---------------------------------------------------------------------------


class TestAnalyzeError:
    def test_returns_none_without_agent(self):
        orch = _make_orchestrator()
        orch.self_correction_agent = None
        assert orch._analyze_error("error", {}) is None

    def test_delegates_to_agent(self):
        orch = _make_orchestrator()
        orch.self_correction_agent = MagicMock()
        orch.self_correction_agent.analyze_error.return_value = "fix imports"
        result = orch._analyze_error("ImportError", {"phase": "act"})
        assert result == "fix imports"

    def test_swallows_agent_exception(self):
        orch = _make_orchestrator()
        orch.self_correction_agent = MagicMock()
        orch.self_correction_agent.analyze_error.side_effect = RuntimeError("crash")
        result = orch._analyze_error("error")
        assert result is None


class TestRunRootCauseAnalysis:
    def test_returns_none_without_agent(self):
        orch = _make_orchestrator()
        orch.root_cause_analysis_agent = None
        result = orch._run_root_cause_analysis([], "", {})
        assert result is None

    def test_delegates_to_agent(self):
        orch = _make_orchestrator()
        orch.root_cause_analysis_agent = MagicMock()
        orch.root_cause_analysis_agent.run.return_value = {"patterns": ["null_pointer"]}
        result = orch._run_root_cause_analysis(["NullPointer"], "log text", {})
        assert result == {"patterns": ["null_pointer"]}

    def test_swallows_agent_exception(self):
        orch = _make_orchestrator()
        orch.root_cause_analysis_agent = MagicMock()
        orch.root_cause_analysis_agent.run.side_effect = RuntimeError("rca crash")
        result = orch._run_root_cause_analysis([], "")
        assert result is None
