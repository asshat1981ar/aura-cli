"""Comprehensive tests for agents/handlers/ module.

Covers:
- agents/handlers/__init__.py  (HANDLER_MAP, PHASE_MAP, re-exports)
- agents/handlers/planner.py   (handle, _resolve_agent)
- agents/handlers/critic.py    (handle, _resolve_agent)
- agents/handlers/coder.py     (handle, _resolve_agent)
- agents/handlers/debugger.py  (handle, _resolve_agent)
- agents/handlers/reflector.py (handle, _resolve_agent)
- agents/handlers/applicator.py(handle, _handle_apply, _handle_rollback, _resolve_agent)
- agents/handlers/*_handler.py (run_*_phase wrappers)
"""

from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

os.environ.setdefault("AURA_TEST_MODE", "1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _brain():
    b = MagicMock()
    b.remember.return_value = None
    b.recall_with_budget.return_value = []
    return b


def _model():
    m = MagicMock()
    m.respond.return_value = '["step 1", "step 2"]'
    return m


def _make_apply_result(success=True, target_path="src/foo.py", backup_path=".aura/backups/foo.py", code="print('hello')", error=None, metadata=None):
    r = MagicMock()
    r.success = success
    r.target_path = target_path
    r.backup_path = backup_path
    r.code = code
    r.error = error
    r.metadata = metadata or {}
    return r


# ===========================================================================
# agents/handlers/__init__.py
# ===========================================================================


class TestHandlersInit:
    def test_handler_map_keys(self):
        from agents.handlers import HANDLER_MAP

        expected = {"plan", "code", "critique", "debug", "reflect", "apply", "planner", "coder", "critic", "debugger", "reflector", "applicator"}
        assert expected == set(HANDLER_MAP.keys())

    def test_phase_map_keys(self):
        from agents.handlers import PHASE_MAP

        expected = {"planner", "coder", "critic", "debugger", "reflector", "applicator", "plan", "code", "critique", "debug", "reflect", "apply"}
        assert expected == set(PHASE_MAP.keys())

    def test_handler_map_values_are_callable(self):
        from agents.handlers import HANDLER_MAP

        for key, fn in HANDLER_MAP.items():
            assert callable(fn), f"HANDLER_MAP[{key!r}] is not callable"

    def test_phase_map_values_are_callable(self):
        from agents.handlers import PHASE_MAP

        for key, fn in PHASE_MAP.items():
            assert callable(fn), f"PHASE_MAP[{key!r}] is not callable"

    def test_run_phase_functions_exported(self):
        from agents.handlers import (
            run_planner_phase,
            run_coder_phase,
            run_critic_phase,
            run_debugger_phase,
            run_reflector_phase,
            run_applicator_phase,
        )

        for fn in [run_planner_phase, run_coder_phase, run_critic_phase, run_debugger_phase, run_reflector_phase, run_applicator_phase]:
            assert callable(fn)

    def test_handler_map_aliases_match_canonical(self):
        from agents.handlers import HANDLER_MAP

        assert HANDLER_MAP["plan"] is HANDLER_MAP["planner"]
        assert HANDLER_MAP["code"] is HANDLER_MAP["coder"]
        assert HANDLER_MAP["critique"] is HANDLER_MAP["critic"]
        assert HANDLER_MAP["debug"] is HANDLER_MAP["debugger"]
        assert HANDLER_MAP["reflect"] is HANDLER_MAP["reflector"]
        assert HANDLER_MAP["apply"] is HANDLER_MAP["applicator"]

    def test_phase_map_aliases_match_canonical(self):
        from agents.handlers import PHASE_MAP

        assert PHASE_MAP["plan"] is PHASE_MAP["planner"]
        assert PHASE_MAP["code"] is PHASE_MAP["coder"]


# ===========================================================================
# agents/handlers/planner.py
# ===========================================================================


class TestPlannerHandle:
    def test_returns_steps_from_agent(self):
        from agents.handlers.planner import handle

        agent = MagicMock()
        agent.run.return_value = {"steps": ["s1", "s2"]}
        result = handle({"goal": "do something"}, {"agent": agent})
        assert result == {"steps": ["s1", "s2"]}
        agent.run.assert_called_once()

    def test_passes_all_task_keys_to_agent(self):
        from agents.handlers.planner import handle

        agent = MagicMock()
        agent.run.return_value = {"steps": []}
        task = {
            "goal": "fix bug",
            "memory_snapshot": "prev",
            "similar_past_problems": "similar",
            "known_weaknesses": "weak",
            "backfill_context": [{"file": "x.py"}],
        }
        handle(task, {"agent": agent})
        call_arg = agent.run.call_args[0][0]
        assert call_arg["goal"] == "fix bug"
        assert call_arg["memory_snapshot"] == "prev"
        assert call_arg["backfill_context"] == [{"file": "x.py"}]

    def test_missing_keys_default_to_empty(self):
        from agents.handlers.planner import handle

        agent = MagicMock()
        agent.run.return_value = {"steps": []}
        result = handle({}, {"agent": agent})
        call_arg = agent.run.call_args[0][0]
        assert call_arg["goal"] == ""
        assert call_arg["memory_snapshot"] == ""
        assert call_arg["backfill_context"] == []

    def test_resolve_agent_uses_injected_agent(self):
        from agents.handlers.planner import _resolve_agent

        agent = MagicMock()
        assert _resolve_agent({"agent": agent}) is agent

    def test_resolve_agent_constructs_from_brain_model(self):
        from agents.handlers.planner import _resolve_agent

        brain = _brain()
        model = _model()
        with patch("agents.planner.PlannerAgent", autospec=False) as MockPA:
            MockPA.return_value = MagicMock()
            agent = _resolve_agent({"brain": brain, "model": model})
        MockPA.assert_called_once_with(brain=brain, model=model)

    def test_resolve_agent_raises_when_context_empty(self):
        from agents.handlers.planner import _resolve_agent

        with pytest.raises(ValueError, match="brain"):
            _resolve_agent({})

    def test_resolve_agent_raises_missing_model(self):
        from agents.handlers.planner import _resolve_agent

        with pytest.raises(ValueError):
            _resolve_agent({"brain": _brain()})


# ===========================================================================
# agents/handlers/critic.py
# ===========================================================================


class TestCriticHandle:
    def _agent(self):
        a = MagicMock()
        a.critique_plan.return_value = {"verdict": "ok"}
        a.critique_code.return_value = {"verdict": "ok"}
        a.validate_mutation.return_value = {"verdict": "ok"}
        return a

    def test_mode_plan_calls_critique_plan(self):
        from agents.handlers.critic import handle

        agent = self._agent()
        result = handle({"mode": "plan", "goal": "g", "plan": ["s1"]}, {"agent": agent})
        agent.critique_plan.assert_called_once_with(task="g", plan=["s1"])
        assert result == {"verdict": "ok"}

    def test_mode_code_calls_critique_code(self):
        from agents.handlers.critic import handle

        agent = self._agent()
        result = handle({"mode": "code", "goal": "g", "code": "x=1", "requirements": "req"}, {"agent": agent})
        agent.critique_code.assert_called_once_with(task="g", code="x=1", requirements="req")

    def test_mode_mutation_calls_validate_mutation(self):
        from agents.handlers.critic import handle

        agent = self._agent()
        result = handle({"mode": "mutation", "mutation_proposal": "add feature"}, {"agent": agent})
        agent.validate_mutation.assert_called_once_with(mutation_proposal="add feature")

    def test_default_mode_is_plan(self):
        from agents.handlers.critic import handle

        agent = self._agent()
        handle({"goal": "g", "plan": []}, {"agent": agent})
        agent.critique_plan.assert_called_once()

    def test_invalid_mode_raises_value_error(self):
        from agents.handlers.critic import handle

        agent = self._agent()
        result = handle({"mode": "bogus"}, {"agent": agent})
        assert "error" in result
        assert "unknown mode" in result["error"]

    def test_resolve_agent_uses_injected_agent(self):
        from agents.handlers.critic import _resolve_agent

        agent = MagicMock()
        assert _resolve_agent({"agent": agent}) is agent

    def test_resolve_agent_constructs_from_brain_model(self):
        from agents.handlers.critic import _resolve_agent

        brain = _brain()
        model = _model()
        with patch("agents.critic.CriticAgent", autospec=False) as MockCA:
            MockCA.return_value = MagicMock()
            _resolve_agent({"brain": brain, "model": model})
        MockCA.assert_called_once_with(brain=brain, model=model)

    def test_resolve_agent_raises_when_context_empty(self):
        from agents.handlers.critic import _resolve_agent

        with pytest.raises(ValueError):
            _resolve_agent({})


# ===========================================================================
# agents/handlers/coder.py
# ===========================================================================


class TestCoderHandle:
    def test_returns_code_key(self):
        from agents.handlers.coder import handle

        agent = MagicMock()
        agent.implement.return_value = "print('hello')"
        result = handle({"task": "write hello"}, {"agent": agent})
        assert result == {"code": "print('hello')"}

    def test_uses_goal_as_fallback_for_task(self):
        from agents.handlers.coder import handle

        agent = MagicMock()
        agent.implement.return_value = "x=1"
        handle({"goal": "implement x"}, {"agent": agent})
        agent.implement.assert_called_once_with("implement x")

    def test_task_takes_precedence_over_goal(self):
        from agents.handlers.coder import handle

        agent = MagicMock()
        agent.implement.return_value = "x=1"
        handle({"task": "task value", "goal": "goal value"}, {"agent": agent})
        agent.implement.assert_called_once_with("task value")

    def test_empty_task_passes_empty_string(self):
        from agents.handlers.coder import handle

        agent = MagicMock()
        agent.implement.return_value = ""
        result = handle({}, {"agent": agent})
        agent.implement.assert_called_once_with("")
        assert result == {"code": ""}

    def test_resolve_agent_uses_injected(self):
        from agents.handlers.coder import _resolve_agent

        agent = MagicMock()
        assert _resolve_agent({"agent": agent}) is agent

    def test_resolve_agent_constructs_with_tester(self):
        from agents.handlers.coder import _resolve_agent

        brain = _brain()
        model = _model()
        tester = MagicMock()
        with patch("agents.coder.CoderAgent", autospec=False) as MockCo:
            MockCo.return_value = MagicMock()
            _resolve_agent({"brain": brain, "model": model, "tester": tester})
        MockCo.assert_called_once_with(brain=brain, model=model, tester=tester)

    def test_resolve_agent_raises_without_brain_or_model(self):
        from agents.handlers.coder import _resolve_agent

        with pytest.raises(ValueError):
            _resolve_agent({})


# ===========================================================================
# agents/handlers/debugger.py
# ===========================================================================


class TestDebuggerHandle:
    def test_returns_diagnosis_result(self):
        from agents.handlers.debugger import handle

        agent = MagicMock()
        agent.diagnose.return_value = {"summary": "ok", "severity": "low"}
        result = handle({"error_message": "NullPointer"}, {"agent": agent})
        assert result["summary"] == "ok"
        assert result["severity"] == "low"

    def test_passes_all_task_keys_to_diagnose(self):
        from agents.handlers.debugger import handle

        agent = MagicMock()
        agent.diagnose.return_value = {}
        task = {
            "error_message": "err",
            "goal": "g",
            "context_text": "ctx",
            "improve_plan": "plan",
            "implement_details": {"step": 1},
        }
        handle(task, {"agent": agent})
        agent.diagnose.assert_called_once_with(
            error_message="err",
            current_goal="g",
            context="ctx",
            improve_plan="plan",
            implement_details={"step": 1},
        )

    def test_defaults_implement_details_to_empty_dict(self):
        from agents.handlers.debugger import handle

        agent = MagicMock()
        agent.diagnose.return_value = {}
        handle({}, {"agent": agent})
        _, kwargs = agent.diagnose.call_args
        assert kwargs["implement_details"] == {}

    def test_resolve_agent_uses_injected(self):
        from agents.handlers.debugger import _resolve_agent

        agent = MagicMock()
        assert _resolve_agent({"agent": agent}) is agent

    def test_resolve_agent_constructs_from_brain_model(self):
        from agents.handlers.debugger import _resolve_agent

        brain = _brain()
        model = _model()
        with patch("agents.debugger.DebuggerAgent", autospec=False) as MockD:
            MockD.return_value = MagicMock()
            _resolve_agent({"brain": brain, "model": model})
        MockD.assert_called_once_with(brain=brain, model=model)

    def test_resolve_agent_raises_when_empty(self):
        from agents.handlers.debugger import _resolve_agent

        with pytest.raises(ValueError):
            _resolve_agent({})


# ===========================================================================
# agents/handlers/reflector.py
# ===========================================================================


class TestReflectorHandle:
    def _agent(self):
        a = MagicMock()
        a.run.return_value = {"summary": "ok", "learnings": ["l1"], "next_actions": [], "skill_summary": {}}
        return a

    def test_returns_reflector_result(self):
        from agents.handlers.reflector import handle

        agent = self._agent()
        result = handle({"verification": {"status": "pass"}}, {"agent": agent})
        assert result["summary"] == "ok"
        assert result["learnings"] == ["l1"]

    def test_passes_all_task_keys(self):
        from agents.handlers.reflector import handle

        agent = self._agent()
        task = {
            "verification": {"status": "pass", "failures": []},
            "next_actions": ["commit"],
            "skill_context": {"key": "val"},
            "pipeline_run_id": "run-42",
        }
        handle(task, {"agent": agent})
        call_arg = agent.run.call_args[0][0]
        assert call_arg["verification"] == {"status": "pass", "failures": []}
        assert call_arg["next_actions"] == ["commit"]
        assert call_arg["skill_context"] == {"key": "val"}
        assert call_arg["pipeline_run_id"] == "run-42"

    def test_defaults_missing_task_keys(self):
        from agents.handlers.reflector import handle

        agent = self._agent()
        handle({}, {"agent": agent})
        call_arg = agent.run.call_args[0][0]
        assert call_arg["verification"] == {}
        assert call_arg["next_actions"] == []
        assert call_arg["skill_context"] == {}
        assert call_arg["pipeline_run_id"] is None

    def test_resolve_agent_uses_injected(self):
        from agents.handlers import reflector as reflector_mod

        agent = MagicMock()
        assert reflector_mod._resolve_agent({"agent": agent}) is agent

    def test_resolve_agent_uses_shared_singleton(self):
        import agents.handlers.reflector as reflector_mod

        reflector_mod._SHARED_AGENT = None  # reset
        with patch("agents.reflector.ReflectorAgent", autospec=False) as MockR:
            mock_instance = MagicMock()
            MockR.return_value = mock_instance
            agent1 = reflector_mod._resolve_agent({})
            agent2 = reflector_mod._resolve_agent({})
        MockR.assert_called_once()  # constructed once only
        assert agent1 is agent2


# ===========================================================================
# agents/handlers/applicator.py
# ===========================================================================


class TestApplicatorHandle:
    def test_apply_action_returns_success_dict(self):
        from agents.handlers.applicator import handle

        agent = MagicMock()
        agent.apply.return_value = _make_apply_result()
        result = handle({"llm_output": "```python\nprint('x')\n```"}, {"agent": agent})
        assert result["success"] is True
        assert result["target_path"] == "src/foo.py"
        assert "backup_path" in result
        assert "code" in result
        assert "error" in result
        assert "metadata" in result

    def test_apply_passes_correct_kwargs(self):
        from agents.handlers.applicator import handle

        agent = MagicMock()
        agent.apply.return_value = _make_apply_result()
        handle(
            {"llm_output": "code", "target_path": "dst.py", "allow_overwrite": False},
            {"agent": agent},
        )
        agent.apply.assert_called_once_with(llm_output="code", target_path="dst.py", allow_overwrite=False)

    def test_apply_default_allow_overwrite_is_true(self):
        from agents.handlers.applicator import handle

        agent = MagicMock()
        agent.apply.return_value = _make_apply_result()
        handle({"llm_output": "x"}, {"agent": agent})
        _, kwargs = agent.apply.call_args
        assert kwargs["allow_overwrite"] is True

    def test_rollback_action_calls_rollback(self):
        from agents.handlers.applicator import handle

        agent = MagicMock()
        agent.rollback.return_value = True
        task = {
            "action": "rollback",
            "apply_result": {
                "success": True,
                "target_path": "src/foo.py",
                "backup_path": ".aura/backups/foo.py",
                "code": "x=1",
                "error": None,
                "metadata": {},
            },
        }
        mock_ar = MagicMock()
        mock_ar.target_path = "src/foo.py"
        with patch("agents.applicator.ApplyResult", return_value=mock_ar):
            result = handle(task, {"agent": agent})
        assert result == {"rolled_back": True}

    def test_rollback_false_when_agent_returns_false(self):
        from agents.handlers.applicator import handle

        agent = MagicMock()
        agent.rollback.return_value = False
        task = {"action": "rollback", "apply_result": {"target_path": "x.py"}}
        mock_ar = MagicMock()
        mock_ar.target_path = "x.py"
        with patch("agents.applicator.ApplyResult", return_value=mock_ar):
            result = handle(task, {"agent": agent})
        assert result == {"rolled_back": False}

    def test_resolve_agent_uses_injected(self):
        from agents.handlers.applicator import _resolve_agent

        agent = MagicMock()
        assert _resolve_agent({"agent": agent}) is agent

    def test_resolve_agent_constructs_with_brain(self):
        from agents.handlers.applicator import _resolve_agent

        brain = _brain()
        with patch("agents.applicator.ApplicatorAgent", autospec=False) as MockA:
            MockA.return_value = MagicMock()
            _resolve_agent({"brain": brain})
        MockA.assert_called_once_with(brain=brain, backup_dir=".aura/backups")

    def test_resolve_agent_uses_custom_backup_dir(self):
        from agents.handlers.applicator import _resolve_agent

        brain = _brain()
        with patch("agents.applicator.ApplicatorAgent", autospec=False) as MockA:
            MockA.return_value = MagicMock()
            _resolve_agent({"brain": brain, "backup_dir": "custom/backups"})
        MockA.assert_called_once_with(brain=brain, backup_dir="custom/backups")

    def test_resolve_agent_raises_without_agent_or_brain(self):
        from agents.handlers.applicator import _resolve_agent

        with pytest.raises(ValueError, match="brain"):
            _resolve_agent({})


# ===========================================================================
# agents/handlers/planner_handler.py  (run_planner_phase)
# ===========================================================================


class TestRunPlannerPhase:
    def test_returns_result_from_planner_handle(self):
        from agents.handlers.planner_handler import run_planner_phase

        with patch("agents.handlers.planner_handler._planner_handler") as mock_mod:
            mock_mod.handle.return_value = {"steps": ["s1"]}
            result = run_planner_phase(context={"agent": MagicMock()}, goal="do it")
        assert result == {"steps": ["s1"]}

    def test_kwargs_forwarded_as_task(self):
        from agents.handlers.planner_handler import run_planner_phase

        with patch("agents.handlers.planner_handler._planner_handler") as mock_mod:
            mock_mod.handle.return_value = {}
            run_planner_phase(context={}, goal="g", memory_snapshot="snap")
        call_kwargs = mock_mod.handle.call_args
        assert call_kwargs[1]["task"]["goal"] == "g"
        assert call_kwargs[1]["task"]["memory_snapshot"] == "snap"

    def test_logs_phase_start(self):
        from agents.handlers.planner_handler import run_planner_phase

        with patch("agents.handlers.planner_handler._planner_handler") as mock_mod, patch("agents.handlers.planner_handler.log_json") as mock_log:
            mock_mod.handle.return_value = {}
            run_planner_phase(context={})
        calls = [c.args[1] for c in mock_log.call_args_list]
        assert "phase_start" in calls


# ===========================================================================
# agents/handlers/critic_handler.py  (run_critic_phase)
# ===========================================================================


class TestRunCriticPhase:
    def test_returns_result_from_critic_handle(self):
        from agents.handlers.critic_handler import run_critic_phase

        with patch("agents.handlers.critic_handler._critic_handler") as mock_mod:
            mock_mod.handle.return_value = {"verdict": "pass"}
            result = run_critic_phase(context={}, mode="plan", goal="g", plan=[])
        assert result == {"verdict": "pass"}

    def test_kwargs_forwarded_as_task(self):
        from agents.handlers.critic_handler import run_critic_phase

        with patch("agents.handlers.critic_handler._critic_handler") as mock_mod:
            mock_mod.handle.return_value = {}
            run_critic_phase(context={}, mode="code", code="x=1")
        call_kwargs = mock_mod.handle.call_args
        assert call_kwargs[1]["task"]["mode"] == "code"
        assert call_kwargs[1]["task"]["code"] == "x=1"


# ===========================================================================
# agents/handlers/coder_handler.py  (run_coder_phase)
# ===========================================================================


class TestRunCoderPhase:
    def test_returns_code_result(self):
        from agents.handlers.coder_handler import run_coder_phase

        with patch("agents.handlers.coder_handler._coder_handler") as mock_mod:
            mock_mod.handle.return_value = {"code": "x=1"}
            result = run_coder_phase(context={}, task="implement x")
        assert result == {"code": "x=1"}

    def test_kwargs_forwarded_as_task(self):
        from agents.handlers.coder_handler import run_coder_phase

        with patch("agents.handlers.coder_handler._coder_handler") as mock_mod:
            mock_mod.handle.return_value = {}
            run_coder_phase(context={}, task="do work", goal="goal")
        call_kwargs = mock_mod.handle.call_args
        assert call_kwargs[1]["task"]["task"] == "do work"


# ===========================================================================
# agents/handlers/debugger_handler.py  (run_debugger_phase)
# ===========================================================================


class TestRunDebuggerPhase:
    def test_returns_debug_result(self):
        from agents.handlers.debugger_handler import run_debugger_phase

        with patch("agents.handlers.debugger_handler._debugger_handler") as mock_mod:
            mock_mod.handle.return_value = {"fix_strategy": "add null check"}
            result = run_debugger_phase(context={}, error_message="NullPointer")
        assert result["fix_strategy"] == "add null check"

    def test_kwargs_forwarded_as_task(self):
        from agents.handlers.debugger_handler import run_debugger_phase

        with patch("agents.handlers.debugger_handler._debugger_handler") as mock_mod:
            mock_mod.handle.return_value = {}
            run_debugger_phase(context={}, error_message="err", goal="g")
        call_kwargs = mock_mod.handle.call_args
        assert call_kwargs[1]["task"]["error_message"] == "err"
        assert call_kwargs[1]["task"]["goal"] == "g"


# ===========================================================================
# agents/handlers/reflector_handler.py  (run_reflector_phase)
# ===========================================================================


class TestRunReflectorPhase:
    def test_returns_reflector_result(self):
        from agents.handlers.reflector_handler import run_reflector_phase

        with patch("agents.handlers.reflector_handler._reflector_handler") as mock_mod:
            mock_mod.handle.return_value = {"learnings": ["l1"]}
            result = run_reflector_phase(context={}, verification={"status": "pass"})
        assert result["learnings"] == ["l1"]

    def test_kwargs_forwarded_as_task(self):
        from agents.handlers.reflector_handler import run_reflector_phase

        with patch("agents.handlers.reflector_handler._reflector_handler") as mock_mod:
            mock_mod.handle.return_value = {}
            run_reflector_phase(context={}, next_actions=["deploy"], pipeline_run_id="run-1")
        call_kwargs = mock_mod.handle.call_args
        assert call_kwargs[1]["task"]["next_actions"] == ["deploy"]
        assert call_kwargs[1]["task"]["pipeline_run_id"] == "run-1"


# ===========================================================================
# agents/handlers/applicator_handler.py  (run_applicator_phase)
# ===========================================================================


class TestRunApplicatorPhase:
    def test_returns_applicator_result(self):
        from agents.handlers.applicator_handler import run_applicator_phase

        with patch("agents.handlers.applicator_handler._applicator_handler") as mock_mod:
            mock_mod.handle.return_value = {"success": True}
            result = run_applicator_phase(context={}, llm_output="code")
        assert result == {"success": True}

    def test_kwargs_forwarded_as_task(self):
        from agents.handlers.applicator_handler import run_applicator_phase

        with patch("agents.handlers.applicator_handler._applicator_handler") as mock_mod:
            mock_mod.handle.return_value = {}
            run_applicator_phase(context={}, llm_output="x", target_path="dst.py")
        call_kwargs = mock_mod.handle.call_args
        assert call_kwargs[1]["task"]["llm_output"] == "x"
        assert call_kwargs[1]["task"]["target_path"] == "dst.py"

    def test_logs_phase_start(self):
        from agents.handlers.applicator_handler import run_applicator_phase

        with patch("agents.handlers.applicator_handler._applicator_handler") as mock_mod, patch("agents.handlers.applicator_handler.log_json") as mock_log:
            mock_mod.handle.return_value = {}
            run_applicator_phase(context={})
        calls = [c.args[1] for c in mock_log.call_args_list]
        assert "phase_start" in calls


# ===========================================================================
# Integration: HANDLER_MAP dispatches to correct handle functions
# ===========================================================================


class TestHandlerMapDispatch:
    def test_plan_handler_dispatches(self):
        from agents.handlers import HANDLER_MAP

        agent = MagicMock()
        agent.run.return_value = {"steps": []}
        HANDLER_MAP["plan"]({"goal": "g"}, {"agent": agent})
        agent.run.assert_called_once()

    def test_code_handler_dispatches(self):
        from agents.handlers import HANDLER_MAP

        agent = MagicMock()
        agent.implement.return_value = "code"
        HANDLER_MAP["code"]({"task": "do it"}, {"agent": agent})
        agent.implement.assert_called_once()

    def test_critique_handler_dispatches(self):
        from agents.handlers import HANDLER_MAP

        agent = MagicMock()
        agent.critique_plan.return_value = {}
        HANDLER_MAP["critique"]({"goal": "g", "plan": []}, {"agent": agent})
        agent.critique_plan.assert_called_once()

    def test_debug_handler_dispatches(self):
        from agents.handlers import HANDLER_MAP

        agent = MagicMock()
        agent.diagnose.return_value = {}
        HANDLER_MAP["debug"]({"error_message": "err"}, {"agent": agent})
        agent.diagnose.assert_called_once()

    def test_reflect_handler_dispatches(self):
        from agents.handlers import HANDLER_MAP

        agent = MagicMock()
        agent.run.return_value = {"learnings": []}
        HANDLER_MAP["reflect"]({}, {"agent": agent})
        agent.run.assert_called_once()
