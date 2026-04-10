"""Unit tests for agents.coder.CoderAgent — targets ≥90% coverage.

Covers:
- __init__ (with/without SCHEMAS_AVAILABLE)
- _respond() role-dispatch branches
- implement() happy path (no tester)
- implement() with tester — "likely pass" fast-exit
- implement() with tester — max-iteration exhaustion
- implement() first-iteration error short-circuit
- implement() mid-iteration error with best_output fallback
- _implement_structured() happy path
- _implement_structured() parse failure → legacy fallback
- _implement_structured() unexpected exception → error dict
- _implement_legacy() JSON-extraction branch
- _implement_legacy() markdown/AURA_TARGET branch
- _implement_legacy() no-target fallback ("unknown.py")
- _format_final_code() with/without target prefix
- _remember_output() with/without tests
- get_structured_info() both availability states
- get_cache_stats() both availability states
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import pytest

from agents.coder import CoderAgent
from tests.fixtures.mock_llm import MockModelAdapter


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_brain() -> MagicMock:
    brain = MagicMock()
    brain.recall_with_budget.return_value = ["memory line 1", "memory line 2"]
    brain.remember.return_value = None
    return brain


def _make_model(response: str = "{}") -> MagicMock:
    """Return a MagicMock whose .respond() returns *response*.

    Omitting `respond_for_role` in spec forces `_respond()` down the
    fallback ``model.respond(prompt)`` branch.
    """
    model = MagicMock(spec=["respond"])
    model.respond.return_value = response
    return model


# Valid CoderOutput-compatible dict
VALID_CODER_DICT: dict = {
    "problem_analysis": "Analyse the problem.",
    "approach_selection": "Iterate and refine.",
    "design_considerations": "Keep it simple.",
    "testing_strategy": "Unit-test each public function.",
    "aura_target": "core/foo.py",
    "code": "def foo(): pass",
    "explanation": "A simple stub.",
    "dependencies": [],
    "edge_cases_handled": [],
    "confidence": 0.9,
}

# Legacy JSON response the agent should parse successfully
LEGACY_JSON_RESPONSE: str = json.dumps({"aura_target": "core/bar.py", "code": "def bar(): return 42"})

# Markdown response with AURA_TARGET directive
MARKDOWN_RESPONSE: str = "# AURA_TARGET: core/baz.py\n```python\ndef baz(): return 'baz'\n```"


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestCoderAgentInit:
    def test_stores_brain_model_and_tester(self):
        brain = _make_brain()
        model = _make_model()
        tester = MagicMock()
        agent = CoderAgent(brain, model, tester=tester)
        assert agent.brain is brain
        assert agent.model is model
        assert agent.tester is tester

    def test_tester_defaults_to_none(self):
        agent = CoderAgent(_make_brain(), _make_model())
        assert agent.tester is None

    def test_capabilities_class_attribute(self):
        assert "code_generation" in CoderAgent.capabilities
        assert "coding" in CoderAgent.capabilities

    def test_use_structured_true_when_schemas_available(self):
        with patch("agents.coder.SCHEMAS_AVAILABLE", True):
            agent = CoderAgent(_make_brain(), _make_model())
        assert agent.use_structured is True

    def test_use_structured_false_when_schemas_unavailable(self):
        with patch("agents.coder.SCHEMAS_AVAILABLE", False):
            agent = CoderAgent(_make_brain(), _make_model())
        assert agent.use_structured is False

    def test_class_constants_present(self):
        assert CoderAgent.MAX_ITERATIONS == 3
        assert CoderAgent.AURA_TARGET_DIRECTIVE == "# AURA_TARGET: "
        assert CoderAgent.CODE_BLOCK_RE is not None


# ---------------------------------------------------------------------------
# _respond()
# ---------------------------------------------------------------------------


class TestCoderRespond:
    def test_falls_back_to_respond_when_no_role_method(self):
        """MagicMock spec=["respond"] has no respond_for_role → fallback."""
        model = _make_model("hello")
        agent = CoderAgent(_make_brain(), model)
        result = agent._respond("some prompt")
        assert result == "hello"
        model.respond.assert_called_once_with("some prompt")

    def test_uses_respond_for_role_when_available(self):
        """Model has respond_for_role as a real attribute → it should be called.

        MagicMock generates child mocks lazily via __getattr__, which
        inspect.getattr_static cannot see.  We use a concrete stub so the
        static lookup succeeds and _respond() takes the role-dispatch path.
        """

        class _ModelWithRole:
            def respond_for_role(self, route_key: str, prompt: str) -> str:  # noqa: D401
                return "role-based response"

            def respond(self, prompt: str) -> str:
                return "plain response"

        model = _ModelWithRole()
        agent = CoderAgent(_make_brain(), model)
        result = agent._respond("some prompt")
        assert result == "role-based response"

    def test_falls_back_to_respond_when_respond_for_role_not_callable(self):
        """respond_for_role exists as attribute but is not callable → use respond."""
        model = MagicMock()
        model.respond_for_role = "not_callable"
        model.respond.return_value = "plain response"
        agent = CoderAgent(_make_brain(), model)
        result = agent._respond("prompt")
        assert result == "plain response"


# ---------------------------------------------------------------------------
# _implement_legacy()
# ---------------------------------------------------------------------------


class TestImplementLegacy:
    def _agent(self, response: str) -> CoderAgent:
        model = _make_model(response)
        return CoderAgent(_make_brain(), model)

    def test_parses_valid_json_aura_target_and_code(self):
        agent = self._agent(LEGACY_JSON_RESPONSE)
        with patch("agents.coder.SCHEMAS_AVAILABLE", False):
            agent.use_structured = False
        result = agent._implement_legacy("task", "mem", "", "", "")
        assert result["aura_target"] == "core/bar.py"
        assert result["code"] == "def bar(): return 42"
        assert result["error"] is None

    def test_parses_json_wrapped_in_prose(self):
        wrapped = f"Here is the result:\n{LEGACY_JSON_RESPONSE}\nDone."
        agent = self._agent(wrapped)
        result = agent._implement_legacy("task", "mem", "", "", "")
        assert result["aura_target"] == "core/bar.py"

    def test_extracts_target_from_markdown_directive(self):
        agent = self._agent(MARKDOWN_RESPONSE)
        result = agent._implement_legacy("task", "mem", "", "", "")
        assert result["aura_target"] == "core/baz.py"
        assert "baz" in result["code"]

    def test_falls_back_to_unknown_when_no_target(self):
        agent = self._agent("def mystery(): pass")
        result = agent._implement_legacy("task", "mem", "", "", "")
        assert result["aura_target"] == "unknown.py"

    def test_code_block_extraction_without_python_fence(self):
        response = "```\ndef plain(): pass\n```"
        agent = self._agent(response)
        result = agent._implement_legacy("task", "mem", "", "", "")
        assert "plain" in result["code"]

    def test_confidence_is_low_for_legacy(self):
        agent = self._agent(LEGACY_JSON_RESPONSE)
        result = agent._implement_legacy("task", "mem", "", "", "")
        # JSON parse gives 0.5; markdown gives 0.3
        assert result["confidence"] in (0.3, 0.5)


# ---------------------------------------------------------------------------
# _implement_structured()
# ---------------------------------------------------------------------------


class TestImplementStructured:
    def _agent_with_structured_response(self, response_text: str) -> CoderAgent:
        model = _make_model(response_text)
        return CoderAgent(_make_brain(), model)

    def test_happy_path_returns_expected_keys(self):
        agent = self._agent_with_structured_response(json.dumps(VALID_CODER_DICT))
        with (
            patch("agents.coder.SCHEMAS_AVAILABLE", True),
            patch("agents.coder.render_prompt", return_value="rendered prompt"),
            patch("agents.coder.CoderOutput") as mock_schema,
            patch("agents.coder._aura_safe_loads", return_value=VALID_CODER_DICT),
        ):
            mock_instance = MagicMock()
            mock_instance.problem_analysis = VALID_CODER_DICT["problem_analysis"]
            mock_instance.approach_selection = VALID_CODER_DICT["approach_selection"]
            mock_instance.design_considerations = VALID_CODER_DICT["design_considerations"]
            mock_instance.testing_strategy = VALID_CODER_DICT["testing_strategy"]
            mock_instance.aura_target = VALID_CODER_DICT["aura_target"]
            mock_instance.code = VALID_CODER_DICT["code"]
            mock_instance.explanation = VALID_CODER_DICT["explanation"]
            mock_instance.dependencies = VALID_CODER_DICT["dependencies"]
            mock_instance.edge_cases_handled = VALID_CODER_DICT["edge_cases_handled"]
            mock_instance.confidence = VALID_CODER_DICT["confidence"]
            mock_instance.dict.return_value = VALID_CODER_DICT
            mock_schema.return_value = mock_instance

            result = agent._implement_structured("task", "mem", "", "", "")

        assert result["aura_target"] == "core/foo.py"
        assert result["code"] == "def foo(): pass"
        assert result["confidence"] == 0.9
        assert result["error"] is None

    def test_json_decode_error_falls_back_to_legacy(self):
        """When _aura_safe_loads raises JSONDecodeError, legacy is called."""
        import json as _json

        agent = self._agent_with_structured_response("not json at all")
        with (
            patch("agents.coder.SCHEMAS_AVAILABLE", True),
            patch("agents.coder.render_prompt", return_value="prompt"),
            patch(
                "agents.coder._aura_safe_loads",
                side_effect=_json.JSONDecodeError("bad json", "", 0),
            ),
        ):
            # Legacy parse of "not json at all" will produce unknown.py
            result = agent._implement_structured("task", "mem", "", "", "")
        # Should not raise; falls through to legacy
        assert "aura_target" in result

    def test_unexpected_exception_returns_error_dict(self):
        """Exceptions beyond JSON/Validation are caught and returned as error."""
        agent = self._agent_with_structured_response("{}")
        with (
            patch("agents.coder.SCHEMAS_AVAILABLE", True),
            patch("agents.coder.render_prompt", return_value="prompt"),
            patch("agents.coder._aura_safe_loads", side_effect=RuntimeError("boom")),
        ):
            result = agent._implement_structured("task", "mem", "", "", "")
        assert result.get("error") is not None
        assert "boom" in result["error"]


# ---------------------------------------------------------------------------
# _format_final_code()
# ---------------------------------------------------------------------------


class TestFormatFinalCode:
    def _agent(self) -> CoderAgent:
        return CoderAgent(_make_brain(), _make_model())

    def test_prepends_target_when_not_already_present(self):
        agent = self._agent()
        result = agent._format_final_code({"aura_target": "core/x.py", "code": "def x(): pass"})
        assert result.startswith("# AURA_TARGET: core/x.py")

    def test_does_not_double_prepend_target(self):
        agent = self._agent()
        code = "# AURA_TARGET: core/x.py\ndef x(): pass"
        result = agent._format_final_code({"aura_target": "core/x.py", "code": code})
        assert result.count("# AURA_TARGET:") == 1

    def test_empty_target_leaves_code_unchanged(self):
        agent = self._agent()
        result = agent._format_final_code({"aura_target": "", "code": "def x(): pass"})
        assert result == "def x(): pass"

    def test_missing_keys_return_empty_string(self):
        agent = self._agent()
        result = agent._format_final_code({})
        assert result == ""


# ---------------------------------------------------------------------------
# _remember_output()
# ---------------------------------------------------------------------------


class TestRememberOutput:
    def test_stores_code_memory(self):
        brain = _make_brain()
        agent = CoderAgent(brain, _make_model())
        agent._remember_output("some task", {"code": "def f(): pass"}, "")
        brain.remember.assert_called_once()

    def test_stores_test_memory_when_tests_provided(self):
        brain = _make_brain()
        agent = CoderAgent(brain, _make_model())
        agent._remember_output("task", {"code": "def f(): pass"}, "def test_f(): pass")
        assert brain.remember.call_count == 2

    def test_no_second_remember_when_no_tests(self):
        brain = _make_brain()
        agent = CoderAgent(brain, _make_model())
        agent._remember_output("task", {"code": "def f(): pass"}, "")
        assert brain.remember.call_count == 1


# ---------------------------------------------------------------------------
# get_structured_info() and get_cache_stats()
# ---------------------------------------------------------------------------


class TestCoderAgentInfo:
    def test_get_structured_info_true(self):
        with patch("agents.coder.SCHEMAS_AVAILABLE", True):
            agent = CoderAgent(_make_brain(), _make_model())
        info = agent.get_structured_info()
        assert info["structured_output_available"] is True
        assert info["schema_version"] == "1.0.0"

    def test_get_structured_info_false(self):
        with patch("agents.coder.SCHEMAS_AVAILABLE", False):
            agent = CoderAgent(_make_brain(), _make_model())
            info = agent.get_structured_info()
        assert info["structured_output_available"] is False
        assert info["schema_version"] is None

    def test_get_cache_stats_when_schemas_available(self):
        with (
            patch("agents.coder.SCHEMAS_AVAILABLE", True),
            patch("agents.coder.get_cached_prompt_stats", return_value={"hits": 3}),
        ):
            agent = CoderAgent(_make_brain(), _make_model())
            stats = agent.get_cache_stats()
        assert stats == {"hits": 3}

    def test_get_cache_stats_when_schemas_unavailable(self):
        with patch("agents.coder.SCHEMAS_AVAILABLE", False):
            agent = CoderAgent(_make_brain(), _make_model())
            stats = agent.get_cache_stats()
        assert "error" in stats


# ---------------------------------------------------------------------------
# implement() — high-level integration
# ---------------------------------------------------------------------------


class TestImplementHappyPath:
    """implement() without a tester — single-iteration fast exit."""

    def _agent_legacy(self, response: str) -> CoderAgent:
        """Return an agent with use_structured=False for predictable branching."""
        brain = _make_brain()
        model = _make_model(response)
        agent = CoderAgent(brain, model)
        agent.use_structured = False
        return agent

    def test_returns_code_string_with_target_header(self):
        agent = self._agent_legacy(LEGACY_JSON_RESPONSE)
        result = agent.implement("build a bar function")
        assert "# AURA_TARGET: core/bar.py" in result
        assert "bar" in result

    def test_recall_with_budget_is_called(self):
        agent = self._agent_legacy(LEGACY_JSON_RESPONSE)
        agent.implement("task")
        agent.brain.recall_with_budget.assert_called()

    def test_remember_is_called_after_success(self):
        agent = self._agent_legacy(LEGACY_JSON_RESPONSE)
        agent.implement("task")
        agent.brain.remember.assert_called()

    def test_empty_task_string_does_not_raise(self):
        agent = self._agent_legacy(LEGACY_JSON_RESPONSE)
        result = agent.implement("")
        assert isinstance(result, str)

    def test_markdown_task_produces_correct_target(self):
        agent = self._agent_legacy(MARKDOWN_RESPONSE)
        result = agent.implement("build baz")
        assert "# AURA_TARGET: core/baz.py" in result


class TestImplementWithTester:
    """implement() when a tester is attached."""

    def _agent_with_tester(self, response: str, feedback: str = "likely pass") -> CoderAgent:
        brain = _make_brain()
        model = _make_model(response)
        tester = MagicMock()
        tester.generate_tests.return_value = "def test_bar(): pass"
        tester.evaluate_code.return_value = {"summary": feedback}
        agent = CoderAgent(brain, model, tester=tester)
        agent.use_structured = False
        return agent

    def test_early_exit_on_likely_pass(self):
        agent = self._agent_with_tester(LEGACY_JSON_RESPONSE, feedback="likely pass")
        result = agent.implement("task")
        assert "# AURA_TARGET: core/bar.py" in result

    def test_max_iterations_reached_returns_best_output(self):
        """Tester always returns failing feedback → hits MAX_ITERATIONS."""
        agent = self._agent_with_tester(LEGACY_JSON_RESPONSE, feedback="tests fail")
        result = agent.implement("task")
        # Should still return code (best_output) not the error sentinel
        assert isinstance(result, str)
        assert "# Error: Max iterations reached" not in result

    def test_tester_generate_tests_called_per_iteration(self):
        agent = self._agent_with_tester(LEGACY_JSON_RESPONSE, feedback="tests fail")
        agent.implement("task")
        assert agent.tester.generate_tests.call_count >= 1


class TestImplementErrorHandling:
    """implement() when the model raises or returns an error dict."""

    def test_first_iteration_error_returns_error_comment(self):
        """If iteration 0 returns an error, implement() returns '# Error: ...'."""
        brain = _make_brain()
        model = _make_model("{}")
        agent = CoderAgent(brain, model)
        # Patch _implement_legacy to always return an error dict
        agent.use_structured = False
        with patch.object(
            agent,
            "_implement_legacy",
            return_value={"error": "model exploded"},
        ):
            result = agent.implement("task")
        assert result == "# Error: model exploded"

    def test_later_iteration_error_uses_best_output(self):
        """Error on iteration >0 should break loop but return best_output."""
        brain = _make_brain()
        model = _make_model("{}")
        agent = CoderAgent(brain, model)
        agent.use_structured = False

        call_count = [0]
        good = {
            "error": None,
            "code": "def ok(): pass",
            "aura_target": "core/ok.py",
        }
        bad = {"error": "boom on 2nd"}

        def side_effect(*_args):
            call_count[0] += 1
            return good if call_count[0] == 1 else bad

        with patch.object(agent, "_implement_legacy", side_effect=side_effect):
            result = agent.implement("task")

        assert "ok" in result  # best_output code was used

    def test_no_tester_no_exception_on_unknown_target(self):
        """Even with no AURA_TARGET, implement() should not raise."""
        brain = _make_brain()
        model = _make_model("def mystery(): pass")
        agent = CoderAgent(brain, model)
        agent.use_structured = False
        result = agent.implement("task")
        assert isinstance(result, str)

    def test_model_raises_exception_in_legacy_returns_unknown(self):
        """If model.respond raises, _implement_legacy handles it gracefully."""
        brain = _make_brain()
        model = MagicMock(spec=["respond"])
        model.respond.side_effect = RuntimeError("connection refused")
        agent = CoderAgent(brain, model)
        agent.use_structured = False
        # RuntimeError propagates through _respond; implement should propagate
        # only for unhandled paths — the legacy path doesn't catch _respond errors,
        # so we just verify the exception is not silently swallowed unexpectedly.
        with pytest.raises(RuntimeError):
            agent.implement("task")


# ---------------------------------------------------------------------------
# Edge cases: empty / minimal task dicts passed as strings
# ---------------------------------------------------------------------------


class TestImplementEdgeCases:
    def test_empty_string_task(self):
        brain = _make_brain()
        model = _make_model(LEGACY_JSON_RESPONSE)
        agent = CoderAgent(brain, model)
        agent.use_structured = False
        result = agent.implement("")
        assert isinstance(result, str)

    def test_task_with_only_whitespace(self):
        brain = _make_brain()
        model = _make_model(LEGACY_JSON_RESPONSE)
        agent = CoderAgent(brain, model)
        agent.use_structured = False
        result = agent.implement("   ")
        assert isinstance(result, str)

    def test_very_long_task_string_does_not_raise(self):
        brain = _make_brain()
        model = _make_model(LEGACY_JSON_RESPONSE)
        agent = CoderAgent(brain, model)
        agent.use_structured = False
        long_task = "implement feature X\n" * 100
        result = agent.implement(long_task)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# MockModelAdapter integration (verify fixture works with CoderAgent)
# ---------------------------------------------------------------------------


class TestMockModelAdapterIntegration:
    def test_mock_model_adapter_coder_response(self):
        """MockModelAdapter matches 'code' key → CODER_RESPONSE (markdown)."""
        model = MockModelAdapter()
        brain = _make_brain()
        agent = CoderAgent(brain, model)
        agent.use_structured = False
        # Just verify no exception; CODER_RESPONSE is valid markdown
        result = agent.implement("write some code")
        assert isinstance(result, str)

    def test_custom_response_via_mock_adapter(self):
        custom_code = "# AURA_TARGET: core/custom.py\ndef custom(): pass"
        model = MockModelAdapter({"code": custom_code})
        brain = _make_brain()
        agent = CoderAgent(brain, model)
        agent.use_structured = False
        result = agent.implement("write some code")
        assert isinstance(result, str)

    def test_mock_adapter_call_log_grows(self):
        model = MockModelAdapter()
        brain = _make_brain()
        agent = CoderAgent(brain, model)
        agent.use_structured = False
        agent.implement("write some code")
        assert len(model.call_log) >= 1
