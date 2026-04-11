"""Tests for core/convergence_escape.py — OscillationDetector, ConvergenceEscapeLoop."""

import pytest
from unittest.mock import MagicMock

from core.convergence_escape import (
    OscillationDetector,
    ConvergenceEscapeLoop,
    STUCK_THRESHOLD,
    OSCILLATION_WINDOW,
)


# ---------------------------------------------------------------------------
# OscillationDetector
# ---------------------------------------------------------------------------

class TestOscillationDetectorRecord:
    def test_no_oscillation_empty(self):
        od = OscillationDetector()
        assert not od.is_oscillating()

    def test_no_oscillation_too_few_scores(self):
        od = OscillationDetector(min_alternations=3)
        od.record(0.8)
        od.record(0.2)
        assert not od.is_oscillating()

    def test_detects_alternation(self):
        od = OscillationDetector(min_alternations=3)
        # pass, fail, pass, fail
        for score in [0.9, 0.1, 0.8, 0.2, 0.7, 0.1]:
            od.record(score)
        assert od.is_oscillating()

    def test_monotone_pass_not_oscillating(self):
        od = OscillationDetector(min_alternations=3)
        for _ in range(6):
            od.record(0.9)
        assert not od.is_oscillating()

    def test_monotone_fail_not_oscillating(self):
        od = OscillationDetector(min_alternations=3)
        for _ in range(6):
            od.record(0.1)
        assert not od.is_oscillating()

    def test_window_caps_history(self):
        od = OscillationDetector(window=4, min_alternations=3)
        # Fill window with non-oscillating pattern
        for _ in range(10):
            od.record(0.9)
        assert not od.is_oscillating()

    def test_reset_clears_scores(self):
        od = OscillationDetector(min_alternations=3)
        for score in [0.9, 0.1, 0.8, 0.2, 0.7, 0.1]:
            od.record(score)
        od.reset()
        assert not od.is_oscillating()


class TestOscillationDetectorSuggestStrategy:
    def test_suggest_vary_prompt_when_last_pass(self):
        od = OscillationDetector()
        od.record(0.9)  # pass
        assert od.suggest_strategy() == "vary_prompt"

    def test_suggest_replan_when_last_fail(self):
        od = OscillationDetector()
        od.record(0.1)  # fail
        assert od.suggest_strategy() == "replan"

    def test_suggest_replan_when_empty(self):
        od = OscillationDetector()
        assert od.suggest_strategy() == "replan"


# ---------------------------------------------------------------------------
# ConvergenceEscapeLoop helpers
# ---------------------------------------------------------------------------

def _make_cycle_entry(goal: str, failure_sig: str = "some_failure") -> dict:
    return {
        "goal": goal,
        "phase_outputs": {
            "verification": {
                "status": "fail",
                "failures": [failure_sig],
            }
        },
    }


def _make_escape_loop(entries: list) -> ConvergenceEscapeLoop:
    memory = MagicMock()
    memory.read_log.return_value = entries
    goal_queue = MagicMock()
    return ConvergenceEscapeLoop(memory_store=memory, goal_queue=goal_queue)


# ---------------------------------------------------------------------------
# ConvergenceEscapeLoop._check
# ---------------------------------------------------------------------------

class TestConvergenceEscapeCheck:
    def test_returns_none_too_few_cycles(self):
        goal = "add feature"
        entries = [_make_cycle_entry(goal)] * 2  # less than MIN_HISTORY
        loop = _make_escape_loop(entries)
        result = loop.check_and_escape(goal, _make_cycle_entry(goal))
        assert result is None

    def test_returns_none_when_failures_differ(self):
        goal = "add feature"
        entries = [
            _make_cycle_entry(goal, "error_a"),
            _make_cycle_entry(goal, "error_b"),
            _make_cycle_entry(goal, "error_c"),
        ]
        loop = _make_escape_loop(entries)
        result = loop.check_and_escape(goal, _make_cycle_entry(goal))
        assert result is None

    def test_triggers_when_same_failure_repeated(self):
        goal = "add feature"
        entries = [_make_cycle_entry(goal, "old_code_not_found")] * STUCK_THRESHOLD
        loop = _make_escape_loop(entries)
        result = loop.check_and_escape(goal, _make_cycle_entry(goal))
        assert result is not None
        assert "strategy" in result

    def test_exception_returns_none(self):
        memory = MagicMock()
        memory.read_log.side_effect = RuntimeError("db down")
        loop = ConvergenceEscapeLoop(memory_store=memory, goal_queue=MagicMock())
        result = loop.check_and_escape("goal", {})
        assert result is None


# ---------------------------------------------------------------------------
# ConvergenceEscapeLoop._select_strategy
# ---------------------------------------------------------------------------

class TestSelectStrategy:
    def setup_method(self):
        self.loop = _make_escape_loop([])

    def test_overwrite_for_old_code_not_found(self):
        assert self.loop._select_strategy("old_code_not_found") == "overwrite"

    def test_overwrite_for_not_found(self):
        assert self.loop._select_strategy("file not found in repo") == "overwrite"

    def test_different_model_for_syntax_error(self):
        assert self.loop._select_strategy("syntax error on line 5") == "different_model"

    def test_different_model_for_parse_error(self):
        assert self.loop._select_strategy("parse failed: unexpected token") == "different_model"

    def test_skip_for_import_error(self):
        assert self.loop._select_strategy("import error: missing dependency") == "skip_and_log"

    def test_skip_for_module_error(self):
        assert self.loop._select_strategy("module not installed") == "skip_and_log"

    def test_decompose_for_unknown_failure(self):
        assert self.loop._select_strategy("some completely unknown failure") == "decompose"


# ---------------------------------------------------------------------------
# ConvergenceEscapeLoop._apply_strategy
# ---------------------------------------------------------------------------

class TestApplyStrategy:
    def setup_method(self):
        self.memory = MagicMock()
        self.queue = MagicMock()
        self.loop = ConvergenceEscapeLoop(memory_store=self.memory, goal_queue=self.queue)

    def test_overwrite_sets_force_overwrite(self):
        result = self.loop._apply_strategy("g", "overwrite", "sig")
        assert result["hint"]["force_overwrite"] is True

    def test_different_model_sets_model_override(self):
        result = self.loop._apply_strategy("g", "different_model", "sig")
        assert result["hint"]["model_override"] == "quality"

    def test_skip_and_log_calls_memory_put(self):
        result = self.loop._apply_strategy("g", "skip_and_log", "sig")
        self.memory.put.assert_called_once()
        assert result["hint"]["skip"] is True

    def test_decompose_queues_prefixed_goal(self):
        result = self.loop._apply_strategy("my goal", "decompose", "sig")
        self.queue.add.assert_called_once_with("[DECOMPOSE] my goal")
        assert result["hint"]["decomposed"] is True

    def test_result_includes_strategy_and_stuck_signature(self):
        result = self.loop._apply_strategy("g", "overwrite", "my_sig")
        assert result["strategy"] == "overwrite"
        assert result["stuck_signature"] == "my_sig"
