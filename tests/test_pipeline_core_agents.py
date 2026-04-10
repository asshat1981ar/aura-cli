"""Unit tests for core pipeline agents: PlannerAgent, CoderAgent, ReflectorAgent, ApplicatorAgent.

Sprint 4 — s4-unit-tests-core-pipeline
Coverage target: planner.py, coder.py, reflector.py, applicator.py
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _mock_brain(memories: list[str] | None = None) -> MagicMock:
    brain = MagicMock()
    brain.recall_with_budget.return_value = memories or []
    return brain


def _mock_model(response: str = "[]") -> MagicMock:
    model = MagicMock()
    model.respond.return_value = response
    return model


# ===========================================================================
# PlannerAgent
# ===========================================================================


class TestPlannerAgentRun:
    """Tests for PlannerAgent.run() standard interface."""

    def _make(self, response="[]", vector_store=None):
        from agents.planner import PlannerAgent

        agent = PlannerAgent(
            brain=_mock_brain(),
            model=_mock_model(response),
            vector_store=vector_store,
        )
        agent.use_structured = False  # force deterministic legacy path
        return agent

    def test_run_returns_steps_key(self):
        agent = self._make(response='["Step 1: do thing"]')
        result = agent.run({"goal": "add feature"})
        assert "steps" in result

    def test_run_steps_is_list(self):
        agent = self._make(response='["Step 1: a", "Step 2: b"]')
        result = agent.run({"goal": "refactor"})
        assert isinstance(result["steps"], list)

    def test_run_missing_goal_defaults_empty_string(self):
        agent = self._make(response='["Step 1: x"]')
        result = agent.run({})
        assert "steps" in result  # should not raise

    def test_run_with_vector_store_hint(self):
        vs = MagicMock()
        vs.query.return_value = ["past reflection about performance"]
        agent = self._make(response='["Step 1: optimise"]', vector_store=vs)
        result = agent.run({"goal": "improve speed"})
        vs.query.assert_called_once()
        assert "steps" in result

    def test_run_vector_store_exception_handled(self):
        vs = MagicMock()
        vs.query.side_effect = RuntimeError("vector db unavailable")
        agent = self._make(response='["Step 1: fallback"]', vector_store=vs)
        result = agent.run({"goal": "anything"})
        assert "steps" in result  # should not propagate exception

    def test_run_already_returns_plan_dict(self):
        """If model returns a JSON object with 'plan' key, run() passes it through."""
        from agents.planner import PlannerAgent

        agent = PlannerAgent(brain=_mock_brain(), model=_mock_model(), vector_store=None)
        agent.use_structured = False

        # Patch plan() to return a dict with 'plan' key directly
        agent.plan = MagicMock(return_value={"plan": [{"step_number": 1, "description": "x"}], "steps": ["Step 1: x"]})
        result = agent.run({"goal": "g"})
        assert "plan" in result or "steps" in result


class TestPlannerAgentLegacy:
    """Tests for PlannerAgent legacy parsing path."""

    def _make(self, response: str) -> Any:
        from agents.planner import PlannerAgent

        agent = PlannerAgent(brain=_mock_brain(), model=_mock_model(response))
        agent.use_structured = False
        return agent

    def test_valid_json_array_returned(self):
        agent = self._make('["Step 1: write tests", "Step 2: run tests"]')
        steps = agent.plan("goal", "mem", "similar", "weakness")
        assert isinstance(steps, list)
        assert steps[0].startswith("Step 1")

    def test_invalid_json_returns_error_step(self):
        agent = self._make("this is not json")
        steps = agent.plan("goal", "mem", "similar", "weakness")
        assert isinstance(steps, list)
        assert any("ERROR" in s for s in steps)

    def test_non_list_json_returns_error(self):
        agent = self._make('{"key": "value"}')
        steps = agent.plan("goal", "mem", "similar", "weakness")
        assert any("ERROR" in s for s in steps)

    def test_backfill_context_included_in_prompt(self):
        from agents.planner import PlannerAgent

        agent = PlannerAgent(brain=_mock_brain(), model=_mock_model('["Step 1: x"]'))
        agent.use_structured = False
        captured = []
        original_respond = agent._respond

        def capture(p):
            captured.append(p)
            return original_respond(p)

        agent._respond = capture
        agent.plan("goal", "", "", "", backfill_context=[{"file": "foo.py", "coverage_pct": 5}])
        assert any("foo.py" in p for p in captured)

    def test_hints_included_in_prompt(self):
        from agents.planner import PlannerAgent

        agent = PlannerAgent(brain=_mock_brain(), model=_mock_model('["Step 1: x"]'))
        agent.use_structured = False
        captured = []
        agent._respond = lambda p: (captured.append(p), '["Step 1: x"]')[1]
        agent.plan("goal", "", "", "", hints=["past learning: use caching"])
        assert any("past learning" in p for p in captured)


class TestPlannerAgentUpdatePlan:
    """Tests for PlannerAgent._update_plan()."""

    def test_update_plan_returns_list(self):
        from agents.planner import PlannerAgent

        agent = PlannerAgent(brain=_mock_brain(), model=_mock_model('["Step 1: revised"]'))
        agent.use_structured = False
        result = agent._update_plan(["Step 1: old"], "make it better")
        assert isinstance(result, list)

    def test_update_plan_accepts_dict_input(self):
        from agents.planner import PlannerAgent

        agent = PlannerAgent(brain=_mock_brain(), model=_mock_model('["Step 1: new"]'))
        agent.use_structured = False
        result = agent._update_plan({"steps": ["Step 1: old"]}, "feedback")
        assert isinstance(result, list)

    def test_update_plan_falls_back_on_bad_json(self):
        from agents.planner import PlannerAgent

        agent = PlannerAgent(brain=_mock_brain(), model=_mock_model("not json"))
        agent.use_structured = False
        original = ["Step 1: keep this"]
        result = agent._update_plan(original, "bad feedback")
        assert result == original  # falls back to original plan


class TestPlannerAgentRespond:
    """Tests for _respond() role-delegation."""

    def test_falls_back_to_model_respond_when_no_role_method(self):
        from agents.planner import PlannerAgent

        model = _mock_model('["Step 1: x"]')
        del model.respond_for_role  # ensure attribute doesn't exist
        agent = PlannerAgent(brain=_mock_brain(), model=model)
        agent.use_structured = False
        steps = agent.plan("goal", "", "", "")
        model.respond.assert_called_once()
        assert isinstance(steps, list)

    def test_uses_respond_for_role_when_available(self):
        from agents.planner import PlannerAgent

        model = MagicMock(spec=["respond", "respond_for_role"])
        # Must set as a real attribute so inspect.getattr_static finds it
        model.respond_for_role = MagicMock(return_value='["Step 1: role-routed"]')
        agent = PlannerAgent(brain=_mock_brain(), model=model)
        agent.use_structured = False
        steps = agent.plan("goal", "", "", "")
        model.respond_for_role.assert_called_once()


# ===========================================================================
# CoderAgent
# ===========================================================================

_VALID_CODE_BLOCK = "```python\n# AURA_TARGET: agents/new_feature.py\ndef hello():\n    return 'world'\n```"

_LEGACY_JSON = json.dumps({"aura_target": "agents/out.py", "code": "def foo(): pass"})


class TestCoderAgentImplement:
    """Tests for CoderAgent.implement()."""

    def _make(self, response: str = _LEGACY_JSON, tester=None) -> Any:
        from agents.coder import CoderAgent

        agent = CoderAgent(brain=_mock_brain(), model=_mock_model(response), tester=tester)
        agent.use_structured = False
        return agent

    def test_returns_string(self):
        agent = self._make()
        result = agent.implement("add feature")
        assert isinstance(result, str)

    def test_aura_target_directive_in_output(self):
        agent = self._make()
        result = agent.implement("add feature")
        assert "# AURA_TARGET:" in result

    def test_markdown_code_block_extraction(self):
        agent = self._make(response=_VALID_CODE_BLOCK)
        result = agent.implement("task")
        assert "def hello" in result

    def test_tester_pass_stops_at_first_iteration(self):
        tester = MagicMock()
        tester.generate_tests.return_value = "def test_foo(): pass"
        tester.evaluate_code.return_value = {"summary": "likely pass on all tests"}
        agent = self._make(tester=tester)
        result = agent.implement("feature")
        assert isinstance(result, str)
        tester.evaluate_code.assert_called_once()

    def test_tester_fail_retries(self):
        tester = MagicMock()
        tester.generate_tests.return_value = "def test_foo(): pass"
        tester.evaluate_code.return_value = {"summary": "tests fail: assertion error"}
        agent = self._make(tester=tester)
        result = agent.implement("feature")
        assert isinstance(result, str)
        assert tester.evaluate_code.call_count <= 3  # MAX_ITERATIONS

    def test_max_iterations_respected(self):
        from agents.coder import CoderAgent

        model = MagicMock()
        model.respond.return_value = _LEGACY_JSON
        tester = MagicMock()
        tester.generate_tests.return_value = "tests"
        tester.evaluate_code.return_value = {"summary": "fail always"}
        agent = CoderAgent(brain=_mock_brain(), model=model, tester=tester)
        agent.use_structured = False
        agent.implement("task")
        assert model.respond.call_count <= CoderAgent.MAX_ITERATIONS

    def test_error_result_on_first_iteration_returns_error_comment(self):
        from agents.coder import CoderAgent

        agent = CoderAgent(brain=_mock_brain(), model=_mock_model("not parseable at all"), tester=None)
        agent.use_structured = False
        # Even with bad JSON, the legacy extractor should produce something
        result = agent.implement("task")
        assert isinstance(result, str)


class TestCoderAgentLegacyParser:
    """Tests for _implement_legacy() JSON and markdown extraction."""

    def _make(self, response: str) -> Any:
        from agents.coder import CoderAgent

        agent = CoderAgent(brain=_mock_brain(), model=_mock_model(response))
        agent.use_structured = False
        return agent

    def test_json_object_with_target_and_code(self):
        payload = json.dumps({"aura_target": "core/thing.py", "code": "x = 1"})
        agent = self._make(payload)
        result = agent.implement("task")
        assert "core/thing.py" in result

    def test_json_embedded_in_prose(self):
        response = "Here is the code:\n" + json.dumps({"aura_target": "a.py", "code": "y=2"})
        agent = self._make(response)
        result = agent.implement("task")
        assert "AURA_TARGET" in result

    def test_markdown_block_fallback(self):
        agent = self._make(_VALID_CODE_BLOCK)
        result = agent.implement("task")
        assert "def hello" in result

    def test_unknown_format_still_returns_string(self):
        agent = self._make("completely unparseable raw text with no json or fences")
        result = agent.implement("task")
        assert isinstance(result, str)


class TestCoderAgentStructuredInfo:
    """Tests for CoderAgent.get_structured_info()."""

    def test_returns_dict_with_expected_keys(self):
        from agents.coder import CoderAgent

        agent = CoderAgent(brain=_mock_brain(), model=_mock_model())
        info = agent.get_structured_info()
        assert "structured_output_available" in info
        assert isinstance(info["structured_output_available"], bool)


# ===========================================================================
# ReflectorAgent
# ===========================================================================


class TestReflectorAgentRun:
    """Tests for ReflectorAgent.run() output contract."""

    @pytest.fixture
    def agent(self):
        from agents.reflector import ReflectorAgent

        return ReflectorAgent()

    def _base_input(self, **kwargs):
        defaults = {"verification": {"status": "pass", "failures": []}, "skill_context": {}}
        defaults.update(kwargs)
        return defaults

    def test_output_has_required_keys(self, agent):
        result = agent.run(self._base_input())
        for key in ("summary", "learnings", "next_actions", "skill_summary"):
            assert key in result, f"missing key: {key}"

    def test_summary_contains_status(self, agent):
        result = agent.run(self._base_input(verification={"status": "fail", "failures": []}))
        assert "fail" in result["summary"]

    def test_learnings_empty_on_no_failures(self, agent):
        result = agent.run(self._base_input())
        # skill_context is empty so no skill_learnings either
        assert result["learnings"] == []

    def test_learnings_include_failure_text(self, agent):
        inp = self._base_input(verification={"status": "fail", "failures": ["NameError: x"]})
        result = agent.run(inp)
        assert any("NameError" in l or "Failures" in l for l in result["learnings"])

    def test_next_actions_passed_through(self, agent):
        inp = self._base_input()
        inp["next_actions"] = ["deploy", "notify"]
        result = agent.run(inp)
        assert result["next_actions"] == ["deploy", "notify"]

    def test_pipeline_run_id_passed_through(self, agent):
        inp = self._base_input()
        inp["pipeline_run_id"] = "abc-123"
        result = agent.run(inp)
        assert result["pipeline_run_id"] == "abc-123"

    def test_missing_verification_key(self, agent):
        result = agent.run({})
        assert "summary" in result
        assert "skip" in result["summary"]

    def test_output_is_json_serialisable(self, agent):
        import json as _json

        result = agent.run(self._base_input())
        _json.dumps(result)  # must not raise


class TestReflectorSkillLearnings:
    """Tests for ReflectorAgent._extract_skill_learnings()."""

    @pytest.fixture
    def agent(self):
        from agents.reflector import ReflectorAgent

        return ReflectorAgent()

    def test_security_critical_count_generates_alert(self, agent):
        ctx = {"security_scanner": {"critical_count": 3, "findings": []}}
        learnings = agent._extract_skill_learnings(ctx)
        assert any("security_scanner" in l for l in learnings)

    def test_no_security_critical_no_alert(self, agent):
        ctx = {"security_scanner": {"critical_count": 0}}
        learnings = agent._extract_skill_learnings(ctx)
        assert not any("security_scanner" in l for l in learnings)

    def test_coupling_above_threshold_generates_alert(self, agent):
        ctx = {"architecture_validator": {"coupling_score": 2.5, "circular_deps": []}}
        learnings = agent._extract_skill_learnings(ctx)
        assert any("coupling" in l for l in learnings)

    def test_coupling_below_threshold_no_alert(self, agent):
        ctx = {"architecture_validator": {"coupling_score": 0.5}}
        learnings = agent._extract_skill_learnings(ctx)
        assert not any("coupling" in l for l in learnings)

    def test_coverage_below_target_generates_alert(self, agent):
        ctx = {"test_coverage_analyzer": {"meets_target": False, "coverage_pct": 12.0}}
        learnings = agent._extract_skill_learnings(ctx)
        assert any("coverage" in l for l in learnings)

    def test_coverage_meets_target_no_alert(self, agent):
        ctx = {"test_coverage_analyzer": {"meets_target": True, "coverage_pct": 85.0}}
        learnings = agent._extract_skill_learnings(ctx)
        assert not any("coverage" in l for l in learnings)

    def test_high_debt_score_generates_alert(self, agent):
        ctx = {"tech_debt_quantifier": {"debt_score": 75}}
        learnings = agent._extract_skill_learnings(ctx)
        assert any("debt" in l for l in learnings)

    def test_empty_context_returns_empty_learnings(self, agent):
        assert agent._extract_skill_learnings({}) == []


class TestReflectorContextAnalysis:
    """Tests for ReflectorAgent._analyze_context_quality()."""

    @pytest.fixture
    def agent(self):
        from agents.reflector import ReflectorAgent

        return ReflectorAgent()

    @pytest.mark.parametrize(
        "error,expected_signal",
        [
            ("NameError: name 'foo' is not defined", "NameError"),
            ("ImportError: cannot import name 'Bar'", "ImportError"),
            ("ModuleNotFoundError: No module named 'xyz'", "ModuleNotFoundError"),
            ("AttributeError: 'NoneType' object has no attribute 'run'", "AttributeError"),
            ("variable not defined in scope", "not defined"),
        ],
    )
    def test_detects_context_gap_signals(self, agent, error, expected_signal):
        gaps = agent._analyze_context_quality([error])
        assert len(gaps) > 0

    def test_no_signals_on_unrelated_error(self, agent):
        gaps = agent._analyze_context_quality(["ValueError: invalid literal for int()"])
        assert gaps == []

    def test_multiple_failures_each_analysed(self, agent):
        failures = ["NameError: a", "ImportError: b"]
        gaps = agent._analyze_context_quality(failures)
        assert len(gaps) == 2


class TestReflectorBuildSkillSummary:
    """Tests for ReflectorAgent._build_skill_summary()."""

    @pytest.fixture
    def agent(self):
        from agents.reflector import ReflectorAgent

        return ReflectorAgent()

    def test_empty_context_returns_empty_summary(self, agent):
        assert agent._build_skill_summary({}) == {}

    def test_security_summary_extracted(self, agent):
        ctx = {"security_scanner": {"critical_count": 2, "findings": ["a", "b"]}}
        summary = agent._build_skill_summary(ctx)
        assert "security_scanner" in summary
        assert summary["security_scanner"]["critical"] == 2
        assert summary["security_scanner"]["total"] == 2

    def test_coverage_summary_extracted(self, agent):
        ctx = {"test_coverage_analyzer": {"coverage_pct": 55.0, "meets_target": True}}
        summary = agent._build_skill_summary(ctx)
        assert summary["test_coverage_analyzer"]["coverage_pct"] == 55.0

    def test_unknown_skill_ignored(self, agent):
        ctx = {"my_custom_skill": {"value": 99}}
        summary = agent._build_skill_summary(ctx)
        assert "my_custom_skill" not in summary

    def test_broken_extractor_does_not_raise(self, agent):
        # coupling_score=None should cause division/comparison without raising
        ctx = {"architecture_validator": {"coupling_score": None, "circular_deps": []}}
        summary = agent._build_skill_summary(ctx)
        assert isinstance(summary, dict)  # must not raise


# ===========================================================================
# ApplicatorAgent
# ===========================================================================


@pytest.fixture
def tmp_workspace(tmp_path):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    return tmp_path, backup_dir


@pytest.fixture
def applicator(tmp_workspace):
    from agents.applicator import ApplicatorAgent

    _, backup_dir = tmp_workspace
    return ApplicatorAgent(brain=_mock_brain(), backup_dir=str(backup_dir))


_GOOD_OUTPUT = "```python\n# AURA_TARGET: agents/generated.py\ndef answer():\n    return 42\n```"

_NO_FENCE_OUTPUT = "def answer(): return 42"
_NO_TARGET_OUTPUT = "```python\ndef answer(): return 42\n```"


class TestApplicatorAgentApply:
    """Tests for ApplicatorAgent.apply()."""

    def test_happy_path_writes_file(self, applicator, tmp_workspace):
        tmp, _ = tmp_workspace
        target = tmp / "agents" / "generated.py"
        result = applicator.apply(_GOOD_OUTPUT, target_path=str(target))
        assert result.success
        assert target.read_text(encoding="utf-8").strip() == ("# AURA_TARGET: agents/generated.py\ndef answer():\n    return 42")

    def test_result_has_target_path(self, applicator, tmp_workspace):
        tmp, _ = tmp_workspace
        target = tmp / "out.py"
        result = applicator.apply(_GOOD_OUTPUT, target_path=str(target))
        assert result.target_path == str(target)

    def test_result_has_code(self, applicator, tmp_workspace):
        tmp, _ = tmp_workspace
        target = tmp / "out.py"
        result = applicator.apply(_GOOD_OUTPUT, target_path=str(target))
        assert result.code is not None
        assert "def answer" in result.code

    def test_no_code_block_returns_failure(self, applicator):
        result = applicator.apply(_NO_FENCE_OUTPUT)
        assert not result.success
        assert "code block" in result.error.lower()

    def test_no_target_path_uses_directive(self, applicator, tmp_workspace):
        tmp, _ = tmp_workspace
        # AURA_TARGET in code points to a relative path; set cwd so it resolves
        import os

        orig = os.getcwd()
        os.chdir(tmp)
        try:
            result = applicator.apply(_GOOD_OUTPUT)
        finally:
            os.chdir(orig)
        assert result.success
        assert "agents/generated.py" in result.target_path

    def test_no_target_no_directive_returns_failure(self, applicator):
        result = applicator.apply(_NO_TARGET_OUTPUT)
        assert not result.success
        assert result.target_path is None

    def test_allow_overwrite_false_blocks_existing_file(self, applicator, tmp_workspace):
        tmp, _ = tmp_workspace
        target = tmp / "existing.py"
        target.write_text("old code")
        result = applicator.apply(_GOOD_OUTPUT, target_path=str(target), allow_overwrite=False)
        assert not result.success
        assert "allow_overwrite" in result.error

    def test_allow_overwrite_true_overwrites_existing_file(self, applicator, tmp_workspace):
        tmp, _ = tmp_workspace
        target = tmp / "existing.py"
        target.write_text("old code")
        result = applicator.apply(_GOOD_OUTPUT, target_path=str(target), allow_overwrite=True)
        assert result.success

    def test_backup_created_for_existing_file(self, applicator, tmp_workspace):
        tmp, backup_dir = tmp_workspace
        target = tmp / "will_be_backed_up.py"
        target.write_text("original content")
        result = applicator.apply(_GOOD_OUTPUT, target_path=str(target))
        assert result.backup_path is not None
        assert Path(result.backup_path).exists()

    def test_no_backup_for_new_file(self, applicator, tmp_workspace):
        tmp, _ = tmp_workspace
        result = applicator.apply(_GOOD_OUTPUT, target_path=str(tmp / "brand_new.py"))
        assert result.backup_path is None

    def test_metadata_has_lines_and_timestamp(self, applicator, tmp_workspace):
        tmp, _ = tmp_workspace
        result = applicator.apply(_GOOD_OUTPUT, target_path=str(tmp / "meta.py"))
        assert "lines" in result.metadata
        assert "timestamp" in result.metadata
        assert result.metadata["lines"] >= 1

    def test_brain_remember_called_on_success(self, applicator, tmp_workspace):
        tmp, _ = tmp_workspace
        applicator.apply(_GOOD_OUTPUT, target_path=str(tmp / "brain.py"))
        applicator.brain.remember.assert_called_once()

    def test_parent_dirs_created_automatically(self, applicator, tmp_workspace):
        tmp, _ = tmp_workspace
        nested = tmp / "deep" / "nested" / "dir" / "out.py"
        result = applicator.apply(_GOOD_OUTPUT, target_path=str(nested))
        assert result.success
        assert nested.exists()

    def test_apply_result_str_on_success(self, applicator, tmp_workspace):
        tmp, _ = tmp_workspace
        result = applicator.apply(_GOOD_OUTPUT, target_path=str(tmp / "str.py"))
        s = str(result)
        assert "OK" in s

    def test_apply_result_str_on_failure(self, applicator):
        result = applicator.apply(_NO_FENCE_OUTPUT)
        s = str(result)
        assert "FAIL" in s


class TestApplicatorAgentRollback:
    """Tests for ApplicatorAgent.rollback()."""

    def test_rollback_restores_original_content(self, applicator, tmp_workspace):
        tmp, _ = tmp_workspace
        target = tmp / "rollback_me.py"
        target.write_text("original")
        result = applicator.apply(_GOOD_OUTPUT, target_path=str(target))
        assert result.success
        assert target.read_text() != "original"
        ok = applicator.rollback(result)
        assert ok
        assert target.read_text() == "original"

    def test_rollback_returns_false_when_no_backup(self, applicator, tmp_workspace):
        tmp, _ = tmp_workspace
        result = applicator.apply(_GOOD_OUTPUT, target_path=str(tmp / "fresh.py"))
        # No backup because the file was new
        assert result.backup_path is None
        ok = applicator.rollback(result)
        assert not ok

    def test_rollback_returns_false_when_backup_deleted(self, applicator, tmp_workspace):
        tmp, _ = tmp_workspace
        target = tmp / "del_backup.py"
        target.write_text("original")
        result = applicator.apply(_GOOD_OUTPUT, target_path=str(target))
        Path(result.backup_path).unlink()  # delete backup
        ok = applicator.rollback(result)
        assert not ok

    def test_rollback_brain_remember_called(self, applicator, tmp_workspace):
        tmp, _ = tmp_workspace
        target = tmp / "rb.py"
        target.write_text("v1")
        result = applicator.apply(_GOOD_OUTPUT, target_path=str(target))
        applicator.brain.remember.reset_mock()
        applicator.rollback(result)
        applicator.brain.remember.assert_called_once()


class TestApplicatorExtractCode:
    """Tests for ApplicatorAgent._extract_code() and _detect_target()."""

    @pytest.fixture
    def ag(self):
        from agents.applicator import ApplicatorAgent

        return ApplicatorAgent(brain=_mock_brain(), backup_dir=tempfile.mkdtemp())

    def test_extracts_python_fenced_block(self, ag):
        code = ag._extract_code("```python\nprint('hi')\n```")
        assert code == "print('hi')"

    def test_extracts_plain_fenced_block(self, ag):
        code = ag._extract_code("```\nprint('hi')\n```")
        assert code == "print('hi')"

    def test_returns_none_when_no_fence(self, ag):
        assert ag._extract_code("no fences here") is None

    def test_detects_aura_target_directive(self, ag):
        code = "# AURA_TARGET: core/foo.py\ndef bar(): pass"
        target = ag._detect_target(code)
        assert target == "core/foo.py"

    def test_detect_target_returns_none_when_absent(self, ag):
        assert ag._detect_target("def bar(): pass") is None

    def test_detect_target_handles_whitespace(self, ag):
        code = "#   AURA_TARGET:   agents/spaced.py  \ndef x(): pass"
        target = ag._detect_target(code)
        assert target == "agents/spaced.py"
