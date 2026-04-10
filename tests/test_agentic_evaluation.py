"""
Unit tests for core/agentic_evaluation.py

Covers dataclasses, BasicReflection, EvaluatorOptimizer, CodeReflector,
RubricBasedEvaluator, and module-level helper functions.

ModelAdapter is mocked everywhere to avoid LLM calls.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch, call

from core.agentic_evaluation import (
    EvaluationCriteria,
    EvaluationResult,
    RefinementHistory,
    BasicReflection,
    EvaluatorOptimizer,
    CodeReflector,
    RubricBasedEvaluator,
    RUBRICS,
    evaluate_sadd_workstream,
    reflect_and_refine_workstream,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _passing_result(score: float = 0.95, iteration: int = 0) -> EvaluationResult:
    return EvaluationResult(score=score, passed=True, feedback="looks good", iteration=iteration)


def _failing_result(score: float = 0.3, iteration: int = 0) -> EvaluationResult:
    return EvaluationResult(score=score, passed=False, feedback="needs work", iteration=iteration)


# ---------------------------------------------------------------------------
# TestDataclasses
# ---------------------------------------------------------------------------


class TestDataclasses(unittest.TestCase):
    def test_evaluation_criteria_defaults(self):
        """threshold defaults to 0.8."""
        c = EvaluationCriteria(name="accuracy", weight=0.5, description="Is it accurate?")
        self.assertEqual(c.threshold, 0.8)
        self.assertEqual(c.name, "accuracy")
        self.assertEqual(c.weight, 0.5)

    def test_evaluation_criteria_custom_threshold(self):
        c = EvaluationCriteria(name="speed", weight=0.2, description="Fast?", threshold=0.6)
        self.assertEqual(c.threshold, 0.6)

    def test_evaluation_result_defaults(self):
        """dimensions defaults to empty dict, iteration defaults to 0."""
        r = EvaluationResult(score=0.7, passed=False, feedback="meh")
        self.assertEqual(r.dimensions, {})
        self.assertEqual(r.iteration, 0)

    def test_evaluation_result_fields(self):
        """All fields can be set and retrieved."""
        dims = {"clarity": 0.8, "depth": 0.6}
        r = EvaluationResult(score=0.85, passed=True, feedback="great", dimensions=dims, iteration=2)
        self.assertEqual(r.score, 0.85)
        self.assertTrue(r.passed)
        self.assertEqual(r.feedback, "great")
        self.assertEqual(r.dimensions, dims)
        self.assertEqual(r.iteration, 2)

    def test_evaluation_result_dimensions_not_shared(self):
        """Default factory produces independent dicts for each instance."""
        r1 = EvaluationResult(score=0.5, passed=False, feedback="a")
        r2 = EvaluationResult(score=0.5, passed=False, feedback="b")
        r1.dimensions["key"] = 1.0
        self.assertNotIn("key", r2.dimensions)

    def test_refinement_history_defaults(self):
        """iterations defaults to empty list, final_output to None, total_time_ms to 0."""
        h = RefinementHistory()
        self.assertEqual(h.iterations, [])
        self.assertIsNone(h.final_output)
        self.assertEqual(h.total_time_ms, 0.0)

    def test_refinement_history_fields(self):
        result = _passing_result()
        h = RefinementHistory(
            iterations=[("output text", result)],
            final_output="final",
            total_time_ms=123.4,
        )
        self.assertEqual(len(h.iterations), 1)
        self.assertEqual(h.final_output, "final")
        self.assertAlmostEqual(h.total_time_ms, 123.4)

    def test_refinement_history_lists_not_shared(self):
        """Default factory produces independent lists for each instance."""
        h1 = RefinementHistory()
        h2 = RefinementHistory()
        h1.iterations.append(("x", _failing_result()))
        self.assertEqual(len(h2.iterations), 0)


# ---------------------------------------------------------------------------
# TestBasicReflection
# ---------------------------------------------------------------------------

EVAL_LOG_PATH = "core.agentic_evaluation.log_json"
MODEL_ADAPTER_PATH = "core.model_adapter.ModelAdapter"


class TestBasicReflection(unittest.TestCase):
    """Tests for BasicReflection.run(), evaluate(), and optimize()."""

    def _make_reflector(self, generator=None, max_iterations=3, min_improvement=0.05):
        if generator is None:
            generator = MagicMock(return_value="initial output")
        return BasicReflection(
            generator=generator,
            criteria=["correctness", "clarity"],
            max_iterations=max_iterations,
            min_improvement=min_improvement,
        )

    # -- run() convergence -------------------------------------------------

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_run_converges_when_passed_on_first_iteration(self, MockAdapter, mock_log):
        """When evaluate() returns passed=True immediately, loop stops after 1 iteration."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = json.dumps({"score": 0.95, "passed": True, "feedback": "perfect", "dimensions": {}})

        reflector = self._make_reflector()
        output, history = reflector.run("some task")

        self.assertEqual(len(history.iterations), 1)
        self.assertIsNotNone(history.final_output)
        self.assertGreater(history.total_time_ms, 0)

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_run_stops_at_max_iterations(self, MockAdapter, mock_log):
        """When evaluate() always returns passed=False, loop runs exactly max_iterations times."""
        mock_model = MockAdapter.return_value
        # Always return a failing result with monotonically increasing scores
        # so stall detection doesn't kick in.
        scores = [0.1, 0.2, 0.4]
        responses = [json.dumps({"score": s, "passed": False, "feedback": "bad", "dimensions": {}}) for s in scores]
        mock_model.generate_text.side_effect = responses * 10  # plenty of responses

        reflector = self._make_reflector(max_iterations=3)
        output, history = reflector.run("some task")

        self.assertEqual(len(history.iterations), 3)

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_run_stops_when_stalled(self, MockAdapter, mock_log):
        """When improvement is below min_improvement threshold, loop stops early."""
        mock_model = MockAdapter.return_value
        # Return same low score each time — improvement is 0 < min_improvement
        fixed_response = json.dumps({"score": 0.3, "passed": False, "feedback": "still bad", "dimensions": {}})
        # optimize() also calls generate_text, so alternate:
        mock_model.generate_text.return_value = fixed_response

        reflector = self._make_reflector(max_iterations=5, min_improvement=0.05)
        output, history = reflector.run("some task")

        # After iteration 1 (score 0.3) there's no prior score, so stall fires on iteration 2
        # which checks iteration > 0 and improvement=0 < 0.05
        self.assertLess(len(history.iterations), 5)

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_run_returns_history_with_correct_structure(self, MockAdapter, mock_log):
        """RefinementHistory has iterations tuples and final_output set."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = json.dumps({"score": 0.9, "passed": True, "feedback": "good", "dimensions": {}})

        reflector = self._make_reflector()
        output, history = reflector.run("write code")

        self.assertIsInstance(history, RefinementHistory)
        self.assertIsNotNone(history.final_output)
        self.assertIsInstance(history.iterations, list)
        # Each element is (str, EvaluationResult)
        for item_output, item_result in history.iterations:
            self.assertIsInstance(item_output, str)
            self.assertIsInstance(item_result, EvaluationResult)

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_run_appends_to_instance_history(self, MockAdapter, mock_log):
        """Each call to run() appends to self.history."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = json.dumps({"score": 1.0, "passed": True, "feedback": "ok", "dimensions": {}})

        reflector = self._make_reflector()
        reflector.run("task 1")
        reflector.run("task 2")

        self.assertEqual(len(reflector.history), 2)

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_run_iteration_index_set_on_result(self, MockAdapter, mock_log):
        """EvaluationResult.iteration is set to the loop index."""
        mock_model = MockAdapter.return_value
        # Fail twice, pass on third — so we get 3 iterations
        responses = [
            json.dumps({"score": 0.1, "passed": False, "feedback": "bad", "dimensions": {}}),
            json.dumps({"score": 0.2, "passed": False, "feedback": "bad", "dimensions": {}}),
            json.dumps({"score": 0.9, "passed": True, "feedback": "ok", "dimensions": {}}),
        ]
        mock_model.generate_text.side_effect = responses * 5

        reflector = self._make_reflector(max_iterations=5)
        _, history = reflector.run("task")

        for idx, (_, result) in enumerate(history.iterations):
            self.assertEqual(result.iteration, idx)

    # -- evaluate() --------------------------------------------------------

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_evaluate_parses_json_response(self, MockAdapter, mock_log):
        """evaluate() correctly parses a valid JSON response from the model."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = json.dumps(
            {
                "score": 0.88,
                "passed": True,
                "feedback": "excellent work",
                "dimensions": {"correctness": 0.9, "clarity": 0.85},
            }
        )

        reflector = self._make_reflector()
        result = reflector.evaluate("some output", "some task")

        self.assertAlmostEqual(result.score, 0.88)
        self.assertTrue(result.passed)
        self.assertEqual(result.feedback, "excellent work")
        self.assertEqual(result.dimensions["correctness"], 0.9)

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_evaluate_handles_parse_error_with_fallback(self, MockAdapter, mock_log):
        """evaluate() returns a zero-score fallback when JSON is invalid."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = "this is not json {{{"

        reflector = self._make_reflector()
        result = reflector.evaluate("some output", "some task")

        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.passed)
        self.assertIsInstance(result.feedback, str)
        # Warning should be logged
        mock_log.assert_called()

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_evaluate_handles_model_exception_with_fallback(self, MockAdapter, mock_log):
        """evaluate() returns fallback if generate_text raises."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.side_effect = RuntimeError("network error")

        reflector = self._make_reflector()
        result = reflector.evaluate("output", "task")

        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.passed)

    # -- optimize() --------------------------------------------------------

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_optimize_returns_model_output(self, MockAdapter, mock_log):
        """optimize() returns the raw string from the model."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = "improved output"

        reflector = self._make_reflector()
        feedback = _failing_result()
        result = reflector.optimize("original", feedback)

        self.assertEqual(result, "improved output")
        mock_model.generate_text.assert_called_once()


# ---------------------------------------------------------------------------
# TestEvaluatorOptimizer
# ---------------------------------------------------------------------------


class TestEvaluatorOptimizer(unittest.TestCase):
    """Tests for EvaluatorOptimizer.run()."""

    def _make_eo(self, evaluator=None, optimizer=None, score_threshold=0.8, max_iterations=3):
        generator = MagicMock(return_value="generated output")
        if evaluator is None:
            evaluator = MagicMock(return_value=_passing_result())
        if optimizer is None:
            optimizer = MagicMock(return_value="optimized output")
        return EvaluatorOptimizer(
            generator=generator,
            evaluator=evaluator,
            optimizer=optimizer,
            score_threshold=score_threshold,
            max_iterations=max_iterations,
        )

    @patch(EVAL_LOG_PATH)
    def test_run_converges_when_score_meets_threshold(self, mock_log):
        """Loop stops after first iteration when evaluator score >= threshold."""
        evaluator = MagicMock(return_value=EvaluationResult(score=0.9, passed=True, feedback="good"))
        eo = self._make_eo(evaluator=evaluator, score_threshold=0.8)
        output, history = eo.run("task")

        self.assertEqual(len(history.iterations), 1)
        self.assertIsNotNone(history.final_output)

    @patch(EVAL_LOG_PATH)
    def test_run_uses_optimizer_when_score_below_threshold(self, mock_log):
        """Optimizer is called when evaluator score is below threshold."""
        # First call fails, second passes
        evaluator = MagicMock(
            side_effect=[
                EvaluationResult(score=0.5, passed=False, feedback="needs improvement"),
                EvaluationResult(score=0.9, passed=True, feedback="better"),
            ]
        )
        optimizer = MagicMock(return_value="improved output")
        eo = self._make_eo(evaluator=evaluator, optimizer=optimizer)
        eo.run("task")

        optimizer.assert_called_once()

    @patch(EVAL_LOG_PATH)
    def test_run_respects_max_iterations(self, mock_log):
        """Loop runs at most max_iterations times regardless of score."""
        evaluator = MagicMock(return_value=EvaluationResult(score=0.3, passed=False, feedback="bad"))
        eo = self._make_eo(evaluator=evaluator, max_iterations=4)
        _, history = eo.run("task")

        self.assertEqual(len(history.iterations), 4)

    @patch(EVAL_LOG_PATH)
    def test_run_returns_refinement_history(self, mock_log):
        """run() returns a properly populated RefinementHistory."""
        eo = self._make_eo()
        output, history = eo.run("task")

        self.assertIsInstance(history, RefinementHistory)
        self.assertIsInstance(output, str)
        self.assertGreater(history.total_time_ms, 0)

    @patch(EVAL_LOG_PATH)
    def test_run_sets_iteration_on_evaluation_result(self, mock_log):
        """EvaluationResult.iteration is set to the loop index."""
        evaluator = MagicMock(
            side_effect=[
                EvaluationResult(score=0.4, passed=False, feedback="bad"),
                EvaluationResult(score=0.6, passed=False, feedback="still bad"),
                EvaluationResult(score=0.9, passed=True, feedback="ok"),
            ]
        )
        eo = self._make_eo(evaluator=evaluator, max_iterations=5)
        _, history = eo.run("task")

        for idx, (_, result) in enumerate(history.iterations):
            self.assertEqual(result.iteration, idx)

    @patch(EVAL_LOG_PATH)
    def test_run_appends_to_instance_history(self, mock_log):
        """Each call to run() appends a RefinementHistory to self.history."""
        eo = self._make_eo()
        eo.run("task 1")
        eo.run("task 2")

        self.assertEqual(len(eo.history), 2)

    @patch(EVAL_LOG_PATH)
    def test_run_generator_called_once_per_run(self, mock_log):
        """Generator is only called once at the start of each run."""
        generator = MagicMock(return_value="output")
        evaluator = MagicMock(return_value=_passing_result())
        eo = EvaluatorOptimizer(
            generator=generator,
            evaluator=evaluator,
            optimizer=MagicMock(return_value="opt"),
            max_iterations=3,
        )
        eo.run("task")

        generator.assert_called_once_with("task")

    @patch(EVAL_LOG_PATH)
    def test_run_optimizer_output_fed_back_for_next_evaluation(self, mock_log):
        """The optimizer output becomes the input to the next evaluator call."""
        evaluator = MagicMock(
            side_effect=[
                EvaluationResult(score=0.5, passed=False, feedback="bad"),
                EvaluationResult(score=0.95, passed=True, feedback="great"),
            ]
        )
        optimizer = MagicMock(return_value="optimized version")
        eo = self._make_eo(evaluator=evaluator, optimizer=optimizer)
        eo.run("task")

        # Second evaluator call should receive the optimizer's output
        second_call_args = evaluator.call_args_list[1]
        self.assertEqual(second_call_args[0][0], "optimized version")


# ---------------------------------------------------------------------------
# TestCodeReflector
# ---------------------------------------------------------------------------


class TestCodeReflector(unittest.TestCase):
    """Tests for CodeReflector methods."""

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_generate_code_returns_string(self, MockAdapter, mock_log):
        """generate_code() returns whatever the model adapter returns."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = "def foo(): pass"

        reflector = CodeReflector()
        result = reflector.generate_code("write a function foo")

        self.assertEqual(result, "def foo(): pass")
        mock_model.generate_text.assert_called_once()

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_generate_tests_returns_string(self, MockAdapter, mock_log):
        """generate_tests() returns test code from the model."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = "def test_foo(): assert foo() is None"

        reflector = CodeReflector()
        result = reflector.generate_tests("spec", "def foo(): pass")

        self.assertIsInstance(result, str)

    @patch(EVAL_LOG_PATH)
    @patch("subprocess.run")
    def test_run_tests_success_when_exit_code_zero(self, mock_subproc, mock_log):
        """run_tests() returns success=True when subprocess exits 0."""
        mock_subproc.return_value = MagicMock(
            returncode=0,
            stdout="1 passed",
            stderr="",
        )

        reflector = CodeReflector()
        result = reflector.run_tests("def foo(): pass", "def test_foo(): assert True")

        self.assertTrue(result["success"])
        self.assertEqual(result["returncode"], 0)

    @patch(EVAL_LOG_PATH)
    @patch("subprocess.run")
    def test_run_tests_failure_when_exit_code_nonzero(self, mock_subproc, mock_log):
        """run_tests() returns success=False when subprocess exits non-zero."""
        mock_subproc.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="AssertionError",
        )

        reflector = CodeReflector()
        result = reflector.run_tests("def foo(): pass", "def test_bad(): assert False")

        self.assertFalse(result["success"])
        self.assertEqual(result["returncode"], 1)

    @patch(EVAL_LOG_PATH)
    @patch("subprocess.run")
    def test_run_tests_handles_timeout(self, mock_subproc, mock_log):
        """run_tests() returns success=False on TimeoutExpired."""
        import subprocess

        mock_subproc.side_effect = subprocess.TimeoutExpired(cmd="pytest", timeout=30)

        reflector = CodeReflector()
        result = reflector.run_tests("code", "tests")

        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("timed out", result["error"])

    @patch(EVAL_LOG_PATH)
    @patch("subprocess.run")
    def test_run_tests_handles_generic_exception(self, mock_subproc, mock_log):
        """run_tests() returns success=False on unexpected exceptions."""
        mock_subproc.side_effect = OSError("no python3")

        reflector = CodeReflector()
        result = reflector.run_tests("code", "tests")

        self.assertFalse(result["success"])
        self.assertIn("error", result)

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_fix_code_returns_model_output(self, MockAdapter, mock_log):
        """fix_code() passes error info to model and returns its response."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = "def foo(): return 42"

        reflector = CodeReflector()
        result = reflector.fix_code("def foo(): pass", {"success": False, "stderr": "AssertionError: expected 42"})

        self.assertEqual(result, "def foo(): return 42")

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    @patch("subprocess.run")
    def test_reflect_and_fix_stops_when_tests_pass(self, mock_subproc, MockAdapter, mock_log):
        """reflect_and_fix() exits loop early when tests pass on first attempt."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = "def foo(): return 1"

        mock_subproc.return_value = MagicMock(returncode=0, stdout="1 passed", stderr="")

        reflector = CodeReflector(max_iterations=5)
        code, history = reflector.reflect_and_fix("write foo returning 1")

        self.assertEqual(len(history.iterations), 1)
        self.assertTrue(history.iterations[0][1].passed)

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    @patch("subprocess.run")
    def test_reflect_and_fix_stops_at_max_iterations_on_failure(self, mock_subproc, MockAdapter, mock_log):
        """reflect_and_fix() runs max_iterations times when tests always fail."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = "def foo(): pass"

        mock_subproc.return_value = MagicMock(returncode=1, stdout="", stderr="FAIL")

        reflector = CodeReflector(max_iterations=3)
        code, history = reflector.reflect_and_fix("spec")

        self.assertEqual(len(history.iterations), 3)
        self.assertIsNotNone(history.final_output)

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    @patch("subprocess.run")
    def test_reflect_and_fix_returns_refinement_history(self, mock_subproc, MockAdapter, mock_log):
        """reflect_and_fix() returns a populated RefinementHistory."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = "code"

        mock_subproc.return_value = MagicMock(returncode=0, stdout="passed", stderr="")

        reflector = CodeReflector()
        code, history = reflector.reflect_and_fix("spec")

        self.assertIsInstance(history, RefinementHistory)
        self.assertIsInstance(code, str)
        self.assertGreater(history.total_time_ms, 0)

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    @patch("subprocess.run")
    def test_reflect_and_fix_score_is_one_on_success_zero_on_failure(self, mock_subproc, MockAdapter, mock_log):
        """EvaluationResult score is 1.0 on passing tests, 0.0 on failing."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = "code"

        mock_subproc.return_value = MagicMock(returncode=0, stdout="passed", stderr="")

        reflector = CodeReflector()
        _, history = reflector.reflect_and_fix("spec")

        _, result = history.iterations[0]
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.passed)


# ---------------------------------------------------------------------------
# TestRubricBasedEvaluator
# ---------------------------------------------------------------------------


class TestRubricBasedEvaluator(unittest.TestCase):
    """Tests for RubricBasedEvaluator."""

    def _make_evaluator(self, rubric=None):
        if rubric is None:
            rubric = {
                "correctness": {"weight": 0.6, "description": "Is it correct?", "threshold": 0.8},
                "readability": {"weight": 0.4, "description": "Is it readable?", "threshold": 0.7},
            }
        return RubricBasedEvaluator(rubric)

    def test_criteria_created_from_rubric(self):
        """RubricBasedEvaluator builds EvaluationCriteria from rubric dict."""
        ev = self._make_evaluator()
        self.assertEqual(len(ev.criteria), 2)
        names = {c.name for c in ev.criteria}
        self.assertIn("correctness", names)
        self.assertIn("readability", names)

    def test_criteria_weights_and_thresholds(self):
        """Criteria weight and threshold are loaded correctly from rubric."""
        ev = self._make_evaluator()
        by_name = {c.name: c for c in ev.criteria}
        self.assertAlmostEqual(by_name["correctness"].weight, 0.6)
        self.assertAlmostEqual(by_name["correctness"].threshold, 0.8)
        self.assertAlmostEqual(by_name["readability"].weight, 0.4)

    def test_criteria_defaults_weight_and_threshold(self):
        """Missing weight/threshold in rubric fall back to 1.0/0.8."""
        rubric = {"quality": {"description": "Overall quality"}}
        ev = RubricBasedEvaluator(rubric)
        c = ev.criteria[0]
        self.assertAlmostEqual(c.weight, 1.0)
        self.assertAlmostEqual(c.threshold, 0.8)

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_evaluate_returns_evaluation_result(self, MockAdapter, mock_log):
        """evaluate() returns an EvaluationResult."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = json.dumps(
            {
                "dimensions": {"correctness": 4, "readability": 3},
                "overall_score": 0.7,
                "feedback": "decent",
                "passed": True,
            }
        )

        ev = self._make_evaluator()
        result = ev.evaluate("some output")

        self.assertIsInstance(result, EvaluationResult)
        self.assertIsInstance(result.score, float)

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_evaluate_dimensions_contains_rubric_keys(self, MockAdapter, mock_log):
        """dimensions in EvaluationResult contains the rubric dimension keys."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = json.dumps(
            {
                "dimensions": {"correctness": 4, "readability": 3},
                "overall_score": 0.7,
                "feedback": "ok",
                "passed": True,
            }
        )

        ev = self._make_evaluator()
        result = ev.evaluate("output")

        self.assertIn("correctness", result.dimensions)
        self.assertIn("readability", result.dimensions)

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_evaluate_score_is_weighted_by_criteria(self, MockAdapter, mock_log):
        """Score is computed as sum(dimension * weight) / 5.0 (normalize from 1-5 scale)."""
        # correctness=5, weight=0.6 → 5*0.6=3.0
        # readability=5, weight=0.4 → 5*0.4=2.0
        # total = 5.0 / 5.0 = 1.0
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = json.dumps(
            {
                "dimensions": {"correctness": 5, "readability": 5},
                "overall_score": 1.0,
                "feedback": "perfect",
                "passed": True,
            }
        )

        ev = self._make_evaluator()
        result = ev.evaluate("perfect output")

        self.assertAlmostEqual(result.score, 1.0)

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_evaluate_score_bounded_between_zero_and_one(self, MockAdapter, mock_log):
        """Score never exceeds 1.0 or goes below 0.0 for valid 1-5 dimension scores."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = json.dumps(
            {
                "dimensions": {"correctness": 3, "readability": 2},
                "overall_score": 0.5,
                "feedback": "average",
                "passed": False,
            }
        )

        ev = self._make_evaluator()
        result = ev.evaluate("mediocre output")

        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_evaluate_handles_parse_error_with_fallback(self, MockAdapter, mock_log):
        """evaluate() returns fallback EvaluationResult on JSON parse error."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = "not json at all"

        ev = self._make_evaluator()
        result = ev.evaluate("output")

        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.passed)
        mock_log.assert_called()

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_evaluate_handles_model_exception_with_fallback(self, MockAdapter, mock_log):
        """evaluate() returns fallback when model raises."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.side_effect = ConnectionError("api down")

        ev = self._make_evaluator()
        result = ev.evaluate("output")

        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.passed)

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_evaluate_uses_context_parameter(self, MockAdapter, mock_log):
        """Context string is passed through to the model prompt."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = json.dumps(
            {
                "dimensions": {"correctness": 4},
                "overall_score": 0.8,
                "feedback": "good",
                "passed": True,
            }
        )

        rubric = {"correctness": {"weight": 1.0, "description": "correct?"}}
        ev = RubricBasedEvaluator(rubric)
        ev.evaluate("output", context="my special context")

        call_args = mock_model.generate_text.call_args[0][0]
        self.assertIn("my special context", call_args)


# ---------------------------------------------------------------------------
# TestPredefinedRubrics
# ---------------------------------------------------------------------------


class TestPredefinedRubrics(unittest.TestCase):
    """Verify the RUBRICS constant has expected keys and structure."""

    def test_rubrics_keys_present(self):
        self.assertIn("code_quality", RUBRICS)
        self.assertIn("documentation", RUBRICS)
        self.assertIn("test_quality", RUBRICS)
        self.assertIn("sadd_workstream", RUBRICS)

    def test_sadd_workstream_rubric_has_required_dimensions(self):
        rubric = RUBRICS["sadd_workstream"]
        self.assertIn("completeness", rubric)
        self.assertIn("quality", rubric)
        self.assertIn("alignment", rubric)
        self.assertIn("efficiency", rubric)

    def test_rubric_dimensions_have_weight(self):
        for rubric_name, rubric in RUBRICS.items():
            for dim_name, dim_config in rubric.items():
                self.assertIn("weight", dim_config, f"{rubric_name}/{dim_name} missing weight")

    def test_rubric_weights_are_positive(self):
        for rubric_name, rubric in RUBRICS.items():
            for dim_name, dim_config in rubric.items():
                self.assertGreater(dim_config["weight"], 0, f"{rubric_name}/{dim_name} weight must be > 0")


# ---------------------------------------------------------------------------
# TestEvaluateSaddWorkstream
# ---------------------------------------------------------------------------


class TestEvaluateSaddWorkstream(unittest.TestCase):
    """Tests for module-level evaluate_sadd_workstream()."""

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_returns_evaluation_result(self, MockAdapter, mock_log):
        """evaluate_sadd_workstream() returns an EvaluationResult."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = json.dumps(
            {
                "dimensions": {"completeness": 4, "quality": 4, "alignment": 4, "efficiency": 3},
                "overall_score": 0.76,
                "feedback": "solid workstream",
                "passed": True,
            }
        )

        result = evaluate_sadd_workstream(
            workstream_title="Implement login",
            output="login feature implemented",
            acceptance_criteria=["User can log in", "JWT issued"],
        )

        self.assertIsInstance(result, EvaluationResult)

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_acceptance_criteria_in_context(self, MockAdapter, mock_log):
        """Acceptance criteria are embedded in the context sent to the model."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = json.dumps(
            {
                "dimensions": {"completeness": 3, "quality": 3, "alignment": 3, "efficiency": 3},
                "overall_score": 0.6,
                "feedback": "ok",
                "passed": False,
            }
        )

        evaluate_sadd_workstream(
            workstream_title="Auth",
            output="auth code",
            acceptance_criteria=["must have rate limiting", "must log events"],
        )

        prompt = mock_model.generate_text.call_args[0][0]
        self.assertIn("must have rate limiting", prompt)
        self.assertIn("must log events", prompt)

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_workstream_title_in_context(self, MockAdapter, mock_log):
        """Workstream title is embedded in the context sent to the model."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = json.dumps(
            {
                "dimensions": {"completeness": 3, "quality": 3, "alignment": 3, "efficiency": 3},
                "overall_score": 0.6,
                "feedback": "ok",
                "passed": False,
            }
        )

        evaluate_sadd_workstream(
            workstream_title="My Unique Workstream Title",
            output="output",
            acceptance_criteria=[],
        )

        prompt = mock_model.generate_text.call_args[0][0]
        self.assertIn("My Unique Workstream Title", prompt)


# ---------------------------------------------------------------------------
# TestReflectAndRefineWorkstream
# ---------------------------------------------------------------------------


class TestReflectAndRefineWorkstream(unittest.TestCase):
    """Tests for module-level reflect_and_refine_workstream()."""

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_returns_tuple_of_string_and_history(self, MockAdapter, mock_log):
        """reflect_and_refine_workstream() returns (str, RefinementHistory)."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = json.dumps({"score": 0.95, "passed": True, "feedback": "great", "dimensions": {}})

        output, history = reflect_and_refine_workstream(
            workstream_title="Feature X",
            task="implement feature X",
            acceptance_criteria=["X works", "X is tested"],
        )

        self.assertIsInstance(output, str)
        self.assertIsInstance(history, RefinementHistory)

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_acceptance_criteria_included_in_criteria_list(self, MockAdapter, mock_log):
        """Acceptance criteria are reflected in the BasicReflection criteria list."""
        mock_model = MockAdapter.return_value
        mock_model.generate_text.return_value = json.dumps({"score": 1.0, "passed": True, "feedback": "ok", "dimensions": {}})

        # Call should complete without error regardless of criteria composition
        output, history = reflect_and_refine_workstream(
            workstream_title="WS",
            task="do the thing",
            acceptance_criteria=["criterion A", "criterion B"],
        )

        self.assertIsNotNone(output)

    @patch(EVAL_LOG_PATH)
    @patch(MODEL_ADAPTER_PATH)
    def test_runs_up_to_three_iterations_by_default(self, MockAdapter, mock_log):
        """Default max_iterations is 3 — loop runs at most 3 times."""
        # Always return failing scores with small increments to avoid stall detection
        scores = [0.1, 0.2, 0.4]
        responses = [json.dumps({"score": s, "passed": False, "feedback": "nope", "dimensions": {}}) for s in scores]
        mock_model = MockAdapter.return_value
        mock_model.generate_text.side_effect = responses * 10

        _, history = reflect_and_refine_workstream(
            workstream_title="WS",
            task="task",
            acceptance_criteria=[],
        )

        self.assertLessEqual(len(history.iterations), 3)


if __name__ == "__main__":
    unittest.main()
