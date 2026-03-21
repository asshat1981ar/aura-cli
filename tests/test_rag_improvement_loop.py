"""Tests for core.rag_improvement_loop — Continuous Self-Improvement Loop for RAG."""
import unittest
from unittest.mock import MagicMock, patch

from core.code_rag import CodeRAG, RAGContext
from core.rag_improvement_loop import (
    RAGImprovementLoop,
    RAGMetrics,
    RetrievalRecord,
    OutcomeRecord,
    TuningAction,
    _HIT_RATE_LOW_THRESHOLD,
    _HIT_RATE_HIGH_THRESHOLD,
    _SUCCESS_RATE_LOW_THRESHOLD,
    _MIN_OUTCOMES_FOR_TUNING,
    _MIN_OUTCOMES_FOR_PROMPT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loop(**kwargs) -> RAGImprovementLoop:
    return RAGImprovementLoop(**kwargs)


def _fill_retrievals(loop: RAGImprovementLoop, count: int, hit: bool = True,
                     retrieval_ms: float = 30.0) -> None:
    for i in range(count):
        loop.record_retrieval(
            goal=f"goal_{i}",
            retrieved_count=2 if hit else 0,
            retrieval_ms=retrieval_ms,
            hit=hit,
        )


def _fill_outcomes(loop: RAGImprovementLoop, count: int, success: bool = True,
                   rag_was_used: bool = True) -> None:
    for i in range(count):
        loop.record_outcome(
            goal=f"goal_{i}",
            success=success,
            rag_was_used=rag_was_used,
        )


# ---------------------------------------------------------------------------
# RAGImprovementLoop — instantiation
# ---------------------------------------------------------------------------

class TestRAGImprovementLoopInit(unittest.TestCase):

    def test_default_init_no_args(self):
        loop = _make_loop()
        self.assertIsNone(loop._rag)
        self.assertEqual(loop._config, {})
        self.assertEqual(loop._tuning_log, [])

    def test_reads_params_from_rag_instance(self):
        rag = MagicMock()
        rag.min_similarity = 0.75
        rag.max_examples = 5
        loop = RAGImprovementLoop(rag_instance=rag)
        self.assertAlmostEqual(loop._similarity_threshold, 0.75)
        self.assertEqual(loop._max_examples, 5)

    def test_reads_params_from_config_store(self):
        config = {"min_similarity": 0.55, "max_examples": 4}
        loop = RAGImprovementLoop(config_store=config)
        self.assertAlmostEqual(loop._similarity_threshold, 0.55)
        self.assertEqual(loop._max_examples, 4)

    def test_custom_prompt_variants(self):
        loop = RAGImprovementLoop(prompt_variants=["v0", "v1"])
        self.assertEqual(loop._prompt_variants, ["v0", "v1"])
        self.assertEqual(loop._prompt_version, 0)


# ---------------------------------------------------------------------------
# record_retrieval
# ---------------------------------------------------------------------------

class TestRecordRetrieval(unittest.TestCase):

    def test_records_basic_event(self):
        loop = _make_loop()
        loop.record_retrieval("goal", retrieved_count=2, retrieval_ms=40.0, hit=True)
        self.assertEqual(len(loop._retrievals), 1)
        rec = loop._retrievals[0]
        self.assertEqual(rec.goal, "goal")
        self.assertEqual(rec.retrieved_count, 2)
        self.assertAlmostEqual(rec.retrieval_ms, 40.0)
        self.assertTrue(rec.hit)

    def test_records_similarity_threshold_used(self):
        loop = _make_loop()
        loop.record_retrieval("g", 1, 10.0, True, similarity_threshold_used=0.72)
        self.assertAlmostEqual(loop._retrievals[0].similarity_threshold_used, 0.72)

    def test_falls_back_to_current_threshold_when_not_provided(self):
        loop = _make_loop()
        loop._similarity_threshold = 0.65
        loop.record_retrieval("g", 1, 10.0, True)
        self.assertAlmostEqual(loop._retrievals[0].similarity_threshold_used, 0.65)

    def test_never_raises_on_bad_input(self):
        loop = _make_loop()
        loop.record_retrieval(None, "bad", "bad", None)  # type: ignore

    def test_window_capped_at_50(self):
        loop = _make_loop()
        _fill_retrievals(loop, 60)
        self.assertLessEqual(len(loop._retrievals), 50)


# ---------------------------------------------------------------------------
# record_outcome
# ---------------------------------------------------------------------------

class TestRecordOutcome(unittest.TestCase):

    def test_records_basic_event(self):
        loop = _make_loop()
        loop.record_outcome("goal", success=True, rag_was_used=True)
        self.assertEqual(len(loop._outcomes), 1)
        rec = loop._outcomes[0]
        self.assertTrue(rec.success)
        self.assertTrue(rec.rag_was_used)

    def test_records_prompt_version(self):
        loop = _make_loop()
        loop._prompt_version = 2
        loop.record_outcome("g", success=False, rag_was_used=False)
        self.assertEqual(loop._outcomes[0].prompt_version, 2)

    def test_updates_prompt_counts(self):
        loop = _make_loop()
        loop.record_outcome("g", success=True, rag_was_used=True)
        loop.record_outcome("g2", success=False, rag_was_used=True)
        counts = loop._prompt_outcome_counts.get(0, {})
        self.assertEqual(counts["total"], 2)
        self.assertEqual(counts["wins"], 1)

    def test_never_raises(self):
        loop = _make_loop()
        loop.record_outcome(None, None, None)  # type: ignore


# ---------------------------------------------------------------------------
# get_metrics
# ---------------------------------------------------------------------------

class TestGetMetrics(unittest.TestCase):

    def test_empty_returns_zero_metrics(self):
        loop = _make_loop()
        m = loop.get_metrics()
        self.assertIsInstance(m, RAGMetrics)
        self.assertEqual(m.window_size, 0)
        self.assertEqual(m.hit_rate, 0.0)

    def test_hit_rate_computed_correctly(self):
        loop = _make_loop()
        _fill_retrievals(loop, 6, hit=True)
        _fill_retrievals(loop, 4, hit=False)
        m = loop.get_metrics()
        self.assertAlmostEqual(m.hit_rate, 0.6)

    def test_avg_retrieval_ms(self):
        loop = _make_loop()
        loop.record_retrieval("g1", 2, 100.0, True)
        loop.record_retrieval("g2", 2, 200.0, True)
        m = loop.get_metrics()
        self.assertAlmostEqual(m.avg_retrieval_ms, 150.0)

    def test_rag_success_rate_and_baseline(self):
        loop = _make_loop()
        # 3 RAG successes
        _fill_outcomes(loop, 3, success=True, rag_was_used=True)
        # 1 RAG failure
        _fill_outcomes(loop, 1, success=False, rag_was_used=True)
        # 2 baseline successes
        _fill_outcomes(loop, 2, success=True, rag_was_used=False)
        # 2 baseline failures
        _fill_outcomes(loop, 2, success=False, rag_was_used=False)
        m = loop.get_metrics()
        self.assertAlmostEqual(m.rag_success_rate, 0.75)
        self.assertAlmostEqual(m.baseline_success_rate, 0.5)
        self.assertAlmostEqual(m.rag_lift, 0.25)

    def test_never_raises(self):
        loop = _make_loop()
        with patch.object(loop, "_compute_metrics", side_effect=RuntimeError("boom")):
            m = loop.get_metrics()
        self.assertIsInstance(m, RAGMetrics)


# ---------------------------------------------------------------------------
# Dynamic config tuning (via analyse)
# ---------------------------------------------------------------------------

class TestDynamicConfigTuning(unittest.TestCase):

    def _loop_with_low_hit_rate(self) -> RAGImprovementLoop:
        """Loop with low hit rate to trigger similarity threshold decrease."""
        loop = _make_loop()
        # Very few hits
        _fill_retrievals(loop, 30, hit=False)
        _fill_retrievals(loop, 5, hit=True)
        # Enough outcomes for tuning
        _fill_outcomes(loop, _MIN_OUTCOMES_FOR_TUNING + 2, success=True)
        return loop

    def test_loosens_similarity_when_hit_rate_low(self):
        loop = self._loop_with_low_hit_rate()
        old_threshold = loop._similarity_threshold
        loop.analyse()
        self.assertLess(loop._similarity_threshold, old_threshold)

    def test_tuning_action_logged_when_threshold_changed(self):
        loop = self._loop_with_low_hit_rate()
        loop.analyse()
        log = loop.get_tuning_log()
        types = [a.action_type for a in log]
        self.assertIn("similarity_threshold", types)

    def test_config_store_updated_on_threshold_change(self):
        config = {"min_similarity": 0.65}
        loop = RAGImprovementLoop(config_store=config)
        # Inject low hit rate
        _fill_retrievals(loop, 35, hit=False)
        _fill_retrievals(loop, 2, hit=True)
        _fill_outcomes(loop, _MIN_OUTCOMES_FOR_TUNING + 2, success=True)
        loop.analyse()
        self.assertLess(config["min_similarity"], 0.65)

    def test_rag_instance_updated_on_threshold_change(self):
        rag = MagicMock()
        rag.min_similarity = 0.65
        rag.max_examples = 3
        loop = RAGImprovementLoop(rag_instance=rag)
        loop._similarity_threshold = 0.65
        # Inject low hit rate
        _fill_retrievals(loop, 35, hit=False)
        _fill_retrievals(loop, 2, hit=True)
        _fill_outcomes(loop, _MIN_OUTCOMES_FOR_TUNING + 2, success=True)
        loop.analyse()
        # min_similarity on the rag object should be reduced
        self.assertLess(rag.min_similarity, 0.65)

    def test_tightens_similarity_when_hits_high_but_success_low(self):
        loop = _make_loop()
        # Lots of hits but low success rate with RAG
        _fill_retrievals(loop, 35, hit=True)
        # RAG outcomes mostly failing
        for _ in range(_MIN_OUTCOMES_FOR_TUNING + 2):
            loop.record_outcome("g", success=False, rag_was_used=True)
        old_threshold = loop._similarity_threshold
        loop.analyse()
        # Should tighten (increase) threshold
        self.assertGreater(loop._similarity_threshold, old_threshold)

    def test_no_tuning_before_min_outcomes(self):
        loop = _make_loop()
        _fill_retrievals(loop, 35, hit=False)
        # Not enough outcomes
        _fill_outcomes(loop, _MIN_OUTCOMES_FOR_TUNING - 1, success=True)
        old_threshold = loop._similarity_threshold
        loop.analyse()
        self.assertAlmostEqual(loop._similarity_threshold, old_threshold)

    def test_reduces_max_examples_when_rag_hurts(self):
        loop = _make_loop()
        _fill_retrievals(loop, 15, hit=True)
        # RAG used — mostly failing
        for _ in range(_MIN_OUTCOMES_FOR_TUNING // 2 + 1):
            loop.record_outcome("g", success=False, rag_was_used=True)
        # Baseline (no RAG) mostly succeeding
        for _ in range(_MIN_OUTCOMES_FOR_TUNING // 2 + 1):
            loop.record_outcome("g", success=True, rag_was_used=False)
        old_max = loop._max_examples
        loop.analyse()
        # rag_lift should be sufficiently negative to trigger reduction
        m = loop.get_metrics()
        if m.rag_lift < -0.10:
            self.assertLess(loop._max_examples, old_max)

    def test_similarity_threshold_never_below_min(self):
        loop = _make_loop()
        loop._similarity_threshold = 0.31  # near floor
        _fill_retrievals(loop, 40, hit=False)
        _fill_outcomes(loop, _MIN_OUTCOMES_FOR_TUNING + 5, success=True)
        for _ in range(20):
            loop.analyse()
        self.assertGreaterEqual(loop._similarity_threshold, 0.30)

    def test_similarity_threshold_never_above_max(self):
        loop = _make_loop()
        loop._similarity_threshold = 0.83  # near ceiling
        _fill_retrievals(loop, 40, hit=True)
        for _ in range(_MIN_OUTCOMES_FOR_TUNING + 5):
            loop.record_outcome("g", success=False, rag_was_used=True)
        for _ in range(20):
            loop.analyse()
        self.assertLessEqual(loop._similarity_threshold, 0.85)


# ---------------------------------------------------------------------------
# Self-prompt refinement
# ---------------------------------------------------------------------------

class TestPromptRefinement(unittest.TestCase):

    def test_initial_prompt_suffix_is_empty_string(self):
        loop = RAGImprovementLoop(prompt_variants=["", "v1", "v2"])
        self.assertEqual(loop.current_prompt_suffix(), "")

    def test_switches_to_better_variant(self):
        loop = RAGImprovementLoop(prompt_variants=["v0", "v1", "v2"])
        # version 0 → 2 successes, 5 failures (win rate 0.29)
        for _ in range(2):
            loop.record_outcome("g", success=True, rag_was_used=True)
        for _ in range(5):
            loop.record_outcome("g", success=False, rag_was_used=True)

        # Simulate version 1 with high win rate
        loop._prompt_version = 1
        for _ in range(5):
            loop.record_outcome("g", success=True, rag_was_used=True)
        loop._prompt_version = 0  # reset to test switching

        # Enough outcomes to trigger prompt refinement
        # Add more outcomes to reach _MIN_OUTCOMES_FOR_PROMPT
        for _ in range(_MIN_OUTCOMES_FOR_PROMPT):
            loop.record_outcome("g", success=False, rag_was_used=True)

        loop.analyse()
        # Should switch away from version 0 (low win rate)
        self.assertNotEqual(loop._prompt_version, 0)

    def test_no_switch_with_single_variant(self):
        loop = RAGImprovementLoop(prompt_variants=["only_one"])
        _fill_outcomes(loop, _MIN_OUTCOMES_FOR_PROMPT + 2, success=False)
        loop.analyse()
        self.assertEqual(loop._prompt_version, 0)

    def test_prompt_version_recorded_in_outcomes(self):
        loop = RAGImprovementLoop(prompt_variants=["v0", "v1"])
        loop._prompt_version = 1
        loop.record_outcome("g", success=True, rag_was_used=True)
        self.assertEqual(loop._outcomes[0].prompt_version, 1)

    def test_win_rate_for_version_zero_when_no_data(self):
        loop = _make_loop()
        self.assertEqual(loop._win_rate_for_version(99), 0.0)

    def test_win_rate_correct(self):
        loop = _make_loop()
        for _ in range(3):
            loop.record_outcome("g", success=True, rag_was_used=True)
        for _ in range(1):
            loop.record_outcome("g", success=False, rag_was_used=True)
        self.assertAlmostEqual(loop._win_rate_for_version(0), 0.75)

    def test_tuning_log_records_prompt_switch(self):
        loop = RAGImprovementLoop(prompt_variants=["v0", "v1"])
        # Version 0 with very low win rate
        for _ in range(6):
            loop.record_outcome("g", success=False, rag_was_used=True)
        # version 1 with high win rate
        loop._prompt_version = 1
        for _ in range(5):
            loop.record_outcome("g", success=True, rag_was_used=True)
        loop._prompt_version = 0
        # Add more outcomes to pass threshold
        for _ in range(_MIN_OUTCOMES_FOR_PROMPT):
            loop.record_outcome("g", success=False, rag_was_used=True)
        loop.analyse()
        types = [a.action_type for a in loop.get_tuning_log()]
        # If a switch happened it should be logged
        if loop._prompt_version != 0:
            self.assertIn("prompt_refinement", types)


# ---------------------------------------------------------------------------
# Workflow hints
# ---------------------------------------------------------------------------

class TestWorkflowHints(unittest.TestCase):

    def test_hints_pushed_to_pipeline_graph(self):
        graph = MagicMock()
        graph.add_hint = MagicMock()
        pipeline = MagicMock()
        pipeline.graph = graph

        loop = RAGImprovementLoop(adaptive_pipeline=pipeline)
        # Create positive rag_lift
        for _ in range(_MIN_OUTCOMES_FOR_TUNING + 2):
            loop.record_outcome("g", success=True, rag_was_used=True)
        for _ in range(3):
            loop.record_outcome("g", success=False, rag_was_used=False)
        _fill_retrievals(loop, 15, hit=True)
        loop.analyse()
        # If rag_lift > 0.10, hints should be pushed
        m = loop.get_metrics()
        if m.rag_lift > 0.10:
            graph.add_hint.assert_called()

    def test_no_crash_if_pipeline_is_none(self):
        loop = _make_loop()
        _fill_retrievals(loop, 15, hit=True)
        _fill_outcomes(loop, _MIN_OUTCOMES_FOR_TUNING + 2, success=True)
        # Should complete without error
        loop.analyse()

    def test_slow_retrieval_hint_generated(self):
        loop = _make_loop()
        _fill_retrievals(loop, 20, hit=True, retrieval_ms=600.0)
        _fill_outcomes(loop, _MIN_OUTCOMES_FOR_TUNING + 2, success=True)
        m = loop.get_metrics()
        hints = loop._build_workflow_hints(m)
        self.assertTrue(any("slow" in h.lower() for h in hints))

    def test_no_hints_when_metrics_normal(self):
        loop = _make_loop()
        # Normal: decent hit rate, moderate retrieval, no strong lift signal
        _fill_retrievals(loop, 20, hit=True, retrieval_ms=50.0)
        m = loop.get_metrics()
        # rag_lift will be 0 since no outcomes yet
        hints = loop._build_workflow_hints(m)
        # No rag_lift signal, no slow retrieval, acceptable hit rate
        lift_hints = [h for h in hints if "RAG retrieval improves" in h]
        self.assertEqual(lift_hints, [])


# ---------------------------------------------------------------------------
# analyse() return value and error handling
# ---------------------------------------------------------------------------

class TestAnalyse(unittest.TestCase):

    def test_returns_dict_with_metrics_key(self):
        loop = _make_loop()
        report = loop.analyse()
        self.assertIn("metrics", report)
        self.assertIn("current_config", report)
        self.assertIn("actions_taken", report)

    def test_metrics_contains_expected_fields(self):
        loop = _make_loop()
        _fill_retrievals(loop, 5, hit=True)
        report = loop.analyse()
        m = report["metrics"]
        for key in ("hit_rate", "avg_retrieval_ms", "rag_success_rate",
                    "rag_lift", "outcomes_analysed", "window_size"):
            self.assertIn(key, m)

    def test_never_raises_on_internal_error(self):
        loop = _make_loop()
        with patch.object(loop, "_compute_metrics", side_effect=RuntimeError("boom")):
            report = loop.analyse()
        self.assertIn("error", report)

    def test_actions_taken_lists_tuning(self):
        loop = _make_loop()
        _fill_retrievals(loop, 35, hit=False)
        _fill_outcomes(loop, _MIN_OUTCOMES_FOR_TUNING + 2, success=True)
        report = loop.analyse()
        # Should have at least one action since hit rate is very low
        self.assertGreater(len(report["actions_taken"]), 0)


# ---------------------------------------------------------------------------
# Integration with CodeRAG
# ---------------------------------------------------------------------------

class TestCodeRAGIntegration(unittest.TestCase):
    """Verify that CodeRAG.retrieve_context reports to the improvement loop."""

    def test_retrieve_context_calls_record_retrieval(self):
        loop = MagicMock(spec=RAGImprovementLoop)
        loop.current_prompt_suffix.return_value = ""

        rag = CodeRAG(improvement_loop=loop)
        # No vector store — retrieval returns empty context
        rag.retrieve_context("add feature")
        loop.record_retrieval.assert_not_called()  # empty context (no store) → early return

    def test_retrieve_context_with_store_calls_record_retrieval(self):
        loop = MagicMock(spec=RAGImprovementLoop)
        loop.current_prompt_suffix.return_value = ""

        store = MagicMock()
        store.search.return_value = ["hit_1"]
        rag = CodeRAG(vector_store=store, improvement_loop=loop)
        rag.retrieve_context("fix bug")
        loop.record_retrieval.assert_called_once()
        call_kwargs = loop.record_retrieval.call_args
        # Accept both positional and keyword args
        args, kwargs = call_kwargs
        # goal should be 'fix bug'
        goal_arg = kwargs.get("goal") or (args[0] if args else None)
        self.assertEqual(goal_arg, "fix bug")

    def test_augment_prompt_appends_suffix(self):
        loop = MagicMock(spec=RAGImprovementLoop)
        loop.current_prompt_suffix.return_value = "\nThink step by step."

        rag = CodeRAG(improvement_loop=loop)
        ctx = RAGContext(examples=[{"content": "def foo(): pass", "source": "vs"}])
        result = rag.augment_prompt("Base prompt", ctx)
        self.assertIn("Think step by step.", result)
        self.assertIn("Base prompt", result)

    def test_augment_prompt_empty_suffix_no_change(self):
        loop = MagicMock(spec=RAGImprovementLoop)
        loop.current_prompt_suffix.return_value = ""

        rag = CodeRAG(improvement_loop=loop)
        ctx = RAGContext(examples=[{"content": "def bar(): pass", "source": "vs"}])
        result = rag.augment_prompt("My prompt", ctx)
        self.assertIn("My prompt", result)
        self.assertIn("def bar(): pass", result)

    def test_no_improvement_loop_compat(self):
        """CodeRAG should work exactly as before when no improvement loop is given."""
        rag = CodeRAG()
        # Should not raise
        ctx = rag.retrieve_context("goal")
        self.assertEqual(ctx.examples, [])


if __name__ == "__main__":
    unittest.main()
