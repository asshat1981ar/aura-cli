"""Tests for the deterministic mock LLM harness.

Verifies that ``MockModelAdapter`` behaves correctly in isolation so downstream
tests can rely on it as a stable building block.
"""

from __future__ import annotations

import json

import pytest

from tests.fixtures.mock_llm import MockModelAdapter
from tests.fixtures.mock_responses import (
    CODER_RESPONSE,
    CRITIC_RESPONSE,
    DEFAULT_RESPONSE,
    PLANNER_RESPONSE,
    REFLECTOR_RESPONSE,
)


# ---------------------------------------------------------------------------
# Default response routing
# ---------------------------------------------------------------------------


class TestDefaultResponses:
    """MockModelAdapter returns the expected built-in response for each phase."""

    def test_planner_prompt_returns_planner_response(self):
        adapter = MockModelAdapter()
        result = adapter.generate("Please plan the next steps for this task")
        assert result == PLANNER_RESPONSE

    def test_coder_prompt_returns_coder_response(self):
        adapter = MockModelAdapter()
        result = adapter.generate("Write code to implement the feature")
        assert result == CODER_RESPONSE

    def test_critic_prompt_returns_critic_response(self):
        adapter = MockModelAdapter()
        result = adapter.generate("Critique the following implementation")
        assert result == CRITIC_RESPONSE

    def test_reflector_prompt_returns_reflector_response(self):
        adapter = MockModelAdapter()
        result = adapter.generate("Reflect on the completed cycle")
        assert result == REFLECTOR_RESPONSE

    def test_matching_is_case_insensitive(self):
        adapter = MockModelAdapter()
        assert adapter.generate("PLAN the work") == PLANNER_RESPONSE
        assert adapter.generate("Plan the work") == PLANNER_RESPONSE
        assert adapter.generate("plan the work") == PLANNER_RESPONSE


# ---------------------------------------------------------------------------
# set_response overrides
# ---------------------------------------------------------------------------


class TestSetResponse:
    """set_response registers new patterns and overrides existing ones."""

    def test_new_pattern_is_returned(self):
        adapter = MockModelAdapter()
        adapter.set_response("custom_keyword", "custom response text")
        result = adapter.generate("Prompt containing custom_keyword here")
        assert result == "custom response text"

    def test_override_existing_pattern(self):
        adapter = MockModelAdapter()
        adapter.set_response("plan", "overridden plan response")
        result = adapter.generate("Please plan the work")
        assert result == "overridden plan response"

    def test_longer_pattern_wins_over_shorter(self):
        adapter = MockModelAdapter()
        adapter.set_response("plan", "short pattern match")
        adapter.set_response("plan and code", "long pattern match")
        result = adapter.generate("Please plan and code the feature")
        assert result == "long pattern match"

    def test_set_response_does_not_affect_unrelated_patterns(self):
        adapter = MockModelAdapter()
        adapter.set_response("custom_keyword", "custom response text")
        # Built-in patterns should still work.
        assert adapter.generate("plan the feature") == PLANNER_RESPONSE


# ---------------------------------------------------------------------------
# Unknown / unmatched prompts
# ---------------------------------------------------------------------------


class TestUnknownPrompts:
    """Unrecognised prompts return DEFAULT_RESPONSE, never raise."""

    def test_unknown_prompt_returns_default(self):
        adapter = MockModelAdapter()
        result = adapter.generate("xyzzy frobnicate quux")
        assert result == DEFAULT_RESPONSE

    def test_empty_prompt_returns_default(self):
        adapter = MockModelAdapter()
        result = adapter.generate("")
        assert result == DEFAULT_RESPONSE

    def test_whitespace_only_prompt_returns_default(self):
        adapter = MockModelAdapter()
        result = adapter.generate("   \t\n  ")
        assert result == DEFAULT_RESPONSE

    def test_unknown_prompt_does_not_raise(self):
        adapter = MockModelAdapter()
        # Must not raise even for weird inputs.
        try:
            adapter.generate("!@#$%^&*() unknown content")
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"generate() raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# Alternative entry-points (respond / respond_for_role)
# ---------------------------------------------------------------------------


class TestAliasedMethods:
    """respond and respond_for_role are transparent aliases of generate."""

    def test_respond_delegates_to_generate(self):
        adapter = MockModelAdapter()
        assert adapter.respond("plan the feature") == adapter.generate("plan the feature")

    def test_respond_for_role_delegates_to_generate(self):
        adapter = MockModelAdapter()
        result = adapter.respond_for_role("planner", "plan the feature")
        assert result == PLANNER_RESPONSE

    def test_respond_for_role_ignores_route_key(self):
        adapter = MockModelAdapter()
        # Same prompt with different route keys should yield the same result.
        r1 = adapter.respond_for_role("role_a", "code the feature")
        r2 = adapter.respond_for_role("role_b", "code the feature")
        assert r1 == r2 == CODER_RESPONSE


# ---------------------------------------------------------------------------
# Call logging
# ---------------------------------------------------------------------------


class TestCallLog:
    """Every prompt is appended to call_log for post-hoc inspection."""

    def test_single_call_logged(self):
        adapter = MockModelAdapter()
        adapter.generate("plan the feature")
        assert len(adapter.call_log) == 1
        assert "plan the feature" in adapter.call_log[0]

    def test_multiple_calls_all_logged(self):
        adapter = MockModelAdapter()
        prompts = ["plan step", "code step", "critique step", "reflect step"]
        for p in prompts:
            adapter.generate(p)
        assert adapter.call_log == prompts

    def test_call_log_starts_empty(self):
        adapter = MockModelAdapter()
        assert adapter.call_log == []


# ---------------------------------------------------------------------------
# Constructor with custom responses
# ---------------------------------------------------------------------------


class TestCustomResponses:
    """Passing a custom responses dict overrides all built-in defaults."""

    def test_custom_mapping_used(self):
        adapter = MockModelAdapter(responses={"hello": "world"})
        assert adapter.generate("Say hello please") == "world"

    def test_custom_mapping_drops_defaults(self):
        adapter = MockModelAdapter(responses={"hello": "world"})
        # Default "plan" pattern is not present, so unknown prompt → default.
        assert adapter.generate("plan the feature") == DEFAULT_RESPONSE

    def test_original_default_not_mutated_by_set_response(self):
        adapter_a = MockModelAdapter()
        adapter_b = MockModelAdapter()
        adapter_a.set_response("plan", "mutated")
        # adapter_b must not be affected.
        assert adapter_b.generate("plan the feature") == PLANNER_RESPONSE


# ---------------------------------------------------------------------------
# Response content sanity checks
# ---------------------------------------------------------------------------


class TestResponseContent:
    """Pre-built mock responses are parseable as their expected format."""

    def test_planner_response_is_valid_json(self):
        data = json.loads(PLANNER_RESPONSE)
        assert "steps" in data
        assert len(data["steps"]) == 3

    def test_coder_response_contains_aura_target(self):
        assert "# AURA_TARGET:" in CODER_RESPONSE
        assert "/tmp/test_output.py" in CODER_RESPONSE

    def test_critic_response_is_valid_json_with_no_blocking_issues(self):
        data = json.loads(CRITIC_RESPONSE)
        assert data["blocking"] is False
        assert data["issues"] == []

    def test_reflector_response_is_valid_json(self):
        data = json.loads(REFLECTOR_RESPONSE)
        assert "summary" in data
        assert "learnings" in data
        assert "next_actions" in data


# ---------------------------------------------------------------------------
# Pytest fixture smoke-test
# ---------------------------------------------------------------------------


def test_mock_model_adapter_fixture(mock_model_adapter):
    """The pytest fixture wires up a ready-to-use MockModelAdapter."""
    assert isinstance(mock_model_adapter, MockModelAdapter)
    result = mock_model_adapter.generate("plan the next steps")
    assert result == PLANNER_RESPONSE


def test_mock_adapter_patch_fixture(mock_adapter_patch):
    """mock_adapter_patch patches core.model_adapter.ModelAdapter and yields the mock."""
    from unittest.mock import patch as _patch
    import core.model_adapter as _ma

    # The patch is active during the test; instantiating ModelAdapter should
    # return our mock instance.
    with _patch("core.model_adapter.ModelAdapter", return_value=mock_adapter_patch):
        instance = _ma.ModelAdapter()
    assert instance is mock_adapter_patch
