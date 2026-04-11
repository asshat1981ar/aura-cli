"""Tests for agents/registry.py — adapters, lazy imports, _make_spec, FALLBACK_CAPABILITIES."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# FALLBACK_CAPABILITIES
# ---------------------------------------------------------------------------


class TestFallbackCapabilities:
    def test_all_core_phases_present(self):
        from agents.registry import FALLBACK_CAPABILITIES

        for phase in ("ingest", "plan", "critique", "synthesize", "act", "sandbox", "verify", "reflect"):
            assert phase in FALLBACK_CAPABILITIES

    def test_each_entry_is_nonempty_list(self):
        from agents.registry import FALLBACK_CAPABILITIES

        for name, caps in FALLBACK_CAPABILITIES.items():
            assert isinstance(caps, list) and len(caps) > 0, f"{name} has empty capabilities"

    def test_plan_includes_planning(self):
        from agents.registry import FALLBACK_CAPABILITIES

        assert "planning" in FALLBACK_CAPABILITIES["plan"]

    def test_act_includes_code_generation(self):
        from agents.registry import FALLBACK_CAPABILITIES

        assert "code_generation" in FALLBACK_CAPABILITIES["act"]

    def test_debugging_entry_present(self):
        from agents.registry import FALLBACK_CAPABILITIES

        assert "debugging" in FALLBACK_CAPABILITIES


# ---------------------------------------------------------------------------
# _make_spec
# ---------------------------------------------------------------------------


class TestMakeSpec:
    def test_uses_agent_native_capabilities(self):
        from agents.registry import _make_spec

        agent = MagicMock()
        agent.capabilities = ["custom_cap", "extra_cap"]
        spec = _make_spec("myagent", agent)
        assert spec.capabilities == ["custom_cap", "extra_cap"]

    def test_falls_back_to_fallback_dict(self):
        from agents.registry import _make_spec

        agent = MagicMock(spec=[])  # no capabilities attr
        spec = _make_spec("plan", agent)
        assert "planning" in spec.capabilities

    def test_falls_back_to_name_list_if_unknown(self):
        from agents.registry import _make_spec

        agent = MagicMock(spec=[])
        spec = _make_spec("totally_unknown_agent", agent)
        assert spec.capabilities == ["totally_unknown_agent"]

    def test_spec_name_matches(self):
        from agents.registry import _make_spec

        agent = MagicMock(spec=[])
        spec = _make_spec("my_agent", agent)
        assert spec.name == "my_agent"

    def test_spec_source_is_local(self):
        from agents.registry import _make_spec

        agent = MagicMock(spec=[])
        spec = _make_spec("x", agent)
        assert spec.source == "local"

    def test_uses_agent_description_attribute(self):
        from agents.registry import _make_spec

        agent = MagicMock(spec=["description"])
        agent.description = "Does something cool"
        spec = _make_spec("x", agent)
        assert spec.description == "Does something cool"

    def test_fallback_description(self):
        from agents.registry import _make_spec

        agent = MagicMock(spec=[])
        spec = _make_spec("mything", agent)
        assert "mything" in spec.description


# ---------------------------------------------------------------------------
# _lazy_import
# ---------------------------------------------------------------------------


class TestLazyImport:
    def test_returns_none_for_unknown(self):
        from agents.registry import _lazy_import

        result = _lazy_import("not_a_real_agent_xyz")
        assert result is None

    def test_returns_class_for_known_agent(self):
        from agents.registry import _lazy_import, _agent_cache

        _agent_cache.clear()
        cls = _lazy_import("synthesizer")
        assert cls is not None
        from agents.synthesizer import SynthesizerAgent

        assert cls is SynthesizerAgent

    def test_caches_result_on_second_call(self):
        from agents.registry import _lazy_import, _agent_cache

        _agent_cache.clear()
        cls1 = _lazy_import("synthesizer")
        cls2 = _lazy_import("synthesizer")
        assert cls1 is cls2

    def test_ingest_importable(self):
        from agents.registry import _lazy_import, _agent_cache

        _agent_cache.clear()
        cls = _lazy_import("ingest")
        assert cls is not None

    def test_debugger_importable(self):
        from agents.registry import _lazy_import, _agent_cache

        _agent_cache.clear()
        cls = _lazy_import("debugger")
        assert cls is not None


# ---------------------------------------------------------------------------
# PlannerAdapter
# ---------------------------------------------------------------------------


class TestPlannerAdapter:
    @pytest.fixture
    def adapter(self):
        from agents.registry import PlannerAdapter

        agent = MagicMock()
        agent.plan.return_value = ["Step 1", "Step 2"]
        return PlannerAdapter(agent=agent)

    def test_name(self, adapter):
        assert adapter.name == "plan"

    def test_run_returns_steps(self, adapter):
        result = adapter.run({"goal": "do something", "memory_snapshot": "", "similar_past_problems": "", "known_weaknesses": ""})
        assert result["steps"] == ["Step 1", "Step 2"]

    def test_run_returns_empty_risks(self, adapter):
        result = adapter.run({"goal": "x"})
        assert result["risks"] == []

    def test_run_passes_goal_to_agent(self, adapter):
        adapter.run({"goal": "my goal"})
        call_args = adapter.agent.plan.call_args
        assert call_args[0][0] == "my goal"

    def test_run_passes_memory_snapshot(self, adapter):
        adapter.run({"goal": "x", "memory_snapshot": "recent mem"})
        call_args = adapter.agent.plan.call_args
        assert call_args[0][1] == "recent mem"

    def test_run_empty_input(self, adapter):
        result = adapter.run({})
        assert "steps" in result


# ---------------------------------------------------------------------------
# CriticAdapter
# ---------------------------------------------------------------------------


class TestCriticAdapter:
    @pytest.fixture
    def adapter(self):
        from agents.registry import CriticAdapter

        agent = MagicMock()
        agent.critique_plan.return_value = "This plan has issues."
        return CriticAdapter(agent=agent)

    def test_name(self, adapter):
        assert adapter.name == "critique"

    def test_run_wraps_critique_in_issues_list(self, adapter):
        result = adapter.run({"task": "build X", "plan": ["step1"]})
        assert result["issues"] == ["This plan has issues."]

    def test_run_returns_empty_fixes(self, adapter):
        result = adapter.run({"task": "x", "plan": []})
        assert result["fixes"] == []

    def test_run_passes_task_and_plan(self, adapter):
        adapter.run({"task": "my task", "plan": ["a", "b"]})
        adapter.agent.critique_plan.assert_called_once_with("my task", ["a", "b"])

    def test_run_empty_input(self, adapter):
        result = adapter.run({})
        assert "issues" in result


# ---------------------------------------------------------------------------
# ActAdapter — unit helpers
# ---------------------------------------------------------------------------


class TestActAdapterHelpers:
    @pytest.fixture
    def adapter(self):
        from agents.registry import ActAdapter

        agent = MagicMock()
        agent.AURA_TARGET_DIRECTIVE = "# AURA_TARGET:"
        agent.implement.return_value = "# generated code"
        return ActAdapter(agent=agent)

    def test_keywords_basic(self, adapter):
        kws = adapter._keywords("hello world foo")
        assert "hello" in kws
        assert "world" in kws

    def test_keywords_filters_short_tokens(self, adapter):
        kws = adapter._keywords("a bb ccc dddd")
        assert "a" not in kws
        assert "bb" not in kws
        assert "ccc" in kws

    def test_keywords_deduplicates(self, adapter):
        kws = adapter._keywords("word word word")
        assert kws.count("word") == 1

    def test_keywords_none_safe(self, adapter):
        kws = adapter._keywords(None, "hello")
        assert "hello" in kws

    def test_score_path_matches(self, adapter):
        score = adapter._score_path("core/planner.py", ["planner", "core"])
        assert score == 2

    def test_score_path_no_match(self, adapter):
        score = adapter._score_path("agents/verifier.py", ["planner"])
        assert score == 0

    def test_score_path_case_insensitive(self, adapter):
        score = adapter._score_path("agents/PLANNER.py", ["planner"])
        assert score == 1

    def test_choose_generated_name_new_file(self, adapter, tmp_path):
        name = adapter._choose_generated_name(tmp_path, ["mymodule"])
        assert "mymodule" in name
        assert not Path(name).exists()

    def test_choose_generated_name_collision(self, adapter, tmp_path):
        (tmp_path / "aura_mymod.py").write_text("x")
        name = adapter._choose_generated_name(tmp_path, ["mymod"])
        assert name != str(tmp_path / "aura_mymod.py")

    def test_choose_generated_name_no_keywords(self, adapter, tmp_path):
        name = adapter._choose_generated_name(tmp_path, [])
        assert "aura_generated" in name


class TestActAdapterRun:
    @pytest.fixture
    def adapter(self):
        from agents.registry import ActAdapter

        agent = MagicMock()
        agent.AURA_TARGET_DIRECTIVE = "# AURA_TARGET:"
        agent.implement.return_value = "def foo(): pass"
        return ActAdapter(agent=agent)

    def test_run_returns_changes_list(self, adapter, tmp_path):
        result = adapter.run({"task": "x", "task_bundle": {}, "project_root": str(tmp_path)})
        assert "changes" in result
        assert len(result["changes"]) == 1

    def test_run_change_has_required_keys(self, adapter, tmp_path):
        result = adapter.run({"task": "x", "task_bundle": {}, "project_root": str(tmp_path)})
        change = result["changes"][0]
        assert "file_path" in change
        assert "new_code" in change
        assert change["old_code"] == ""
        assert change["overwrite_file"] is True

    def test_run_strips_aura_target_directive(self, adapter, tmp_path):
        adapter.agent.implement.return_value = "# AURA_TARGET: core/foo.py\ndef bar(): pass"
        result = adapter.run({"task": "x", "task_bundle": {}, "project_root": str(tmp_path)})
        change = result["changes"][0]
        assert change["file_path"] == "core/foo.py"
        assert "# AURA_TARGET:" not in change["new_code"]

    def test_run_falls_back_to_choose_file_path(self, adapter, tmp_path):
        adapter.agent.implement.return_value = "def foo(): pass"
        (tmp_path / "core").mkdir()
        result = adapter.run({"task": "planner refactor", "task_bundle": {}, "project_root": str(tmp_path)})
        assert result["changes"][0]["file_path"] != ""


# ---------------------------------------------------------------------------
# SandboxAdapter
# ---------------------------------------------------------------------------


class TestSandboxAdapter:
    @pytest.fixture
    def sandbox_result(self):
        r = MagicMock()
        r.passed = True
        r.exit_code = 0
        r.stdout = "ok"
        r.stderr = ""
        r.timed_out = False
        r.summary.return_value = "passed"
        return r

    @pytest.fixture
    def adapter(self, sandbox_result):
        from agents.registry import SandboxAdapter

        agent = MagicMock()
        agent.run_code.return_value = sandbox_result
        return SandboxAdapter(agent=agent)

    def test_name(self, adapter):
        assert adapter.name == "sandbox"

    def test_dry_run_skips_execution(self, adapter):
        result = adapter.run({"dry_run": True})
        assert result["status"] == "skip"
        adapter.agent.run_code.assert_not_called()

    def test_no_act_output_skips(self, adapter):
        result = adapter.run({})
        assert result["status"] == "skip"

    def test_empty_changes_skips(self, adapter):
        result = adapter.run({"act": {"changes": []}})
        assert result["status"] == "skip"

    def test_no_new_code_skips(self, adapter):
        result = adapter.run({"act": {"changes": [{"new_code": ""}]}})
        assert result["status"] == "skip"

    def test_passing_snippet_returns_pass(self, adapter):
        result = adapter.run({"act": {"changes": [{"new_code": "x = 1"}]}})
        assert result["status"] == "pass"
        assert result["passed"] is True

    def test_failing_snippet_returns_fail(self, adapter, sandbox_result):
        sandbox_result.passed = False
        result = adapter.run({"act": {"changes": [{"new_code": "x = 1"}]}})
        assert result["status"] == "fail"
        assert result["passed"] is False

    def test_details_include_snippet_count(self, adapter):
        result = adapter.run({"act": {"changes": [{"new_code": "x = 1"}, {"new_code": "y = 2"}]}})
        assert result["details"]["snippet_count"] == 2

    def test_whitespace_only_code_skipped(self, adapter):
        result = adapter.run({"act": {"changes": [{"new_code": "   \n  "}]}})
        assert result["status"] == "skip"
