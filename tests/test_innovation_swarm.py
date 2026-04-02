"""
Unit tests for agents/innovation_swarm.py.

Mocks brainstorming bots and logging to test InnovationSwarm in isolation.
"""

import unittest
from unittest.mock import MagicMock, patch

from agents.schemas import Idea, InnovationOutput, TechniqueResult, InnovationPhase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ideas(n, technique="test_technique", novelty=0.7, feasibility=0.6, impact=0.5):
    """Return a list of n Idea objects with the given scores."""
    return [
        Idea(
            description=f"idea_{i}",
            technique=technique,
            novelty=novelty,
            feasibility=feasibility,
            impact=impact,
        )
        for i in range(n)
    ]


def _make_technique_result(technique_name, ideas):
    """Return a TechniqueResult wrapping the given ideas list."""
    return TechniqueResult(
        technique=technique_name,
        ideas=ideas,
        idea_count=len(ideas),
    )


def _make_mock_bot(technique_name, ideas):
    """Return a mock bot whose generate() returns *ideas*."""
    bot = MagicMock()
    bot.technique_name = technique_name
    bot.generate.return_value = ideas
    return bot


# ---------------------------------------------------------------------------
# TestInnovationSwarmInit
# ---------------------------------------------------------------------------

class TestInnovationSwarmInit(unittest.TestCase):

    def _make_swarm(self, **kwargs):
        from agents.innovation_swarm import InnovationSwarm
        return InnovationSwarm(**kwargs)

    def test_init_defaults(self):
        swarm = self._make_swarm()
        self.assertIsNone(swarm.brain)
        self.assertIsNone(swarm.model)
        self.assertIsNone(swarm.session_id)

    def test_init_with_args(self):
        brain = MagicMock()
        model = MagicMock()
        swarm = self._make_swarm(brain=brain, model=model)
        self.assertIs(swarm.brain, brain)
        self.assertIs(swarm.model, model)

    def test_capabilities_non_empty(self):
        from agents.innovation_swarm import InnovationSwarm
        self.assertTrue(len(InnovationSwarm.capabilities) > 0)

    def test_description_is_string(self):
        from agents.innovation_swarm import InnovationSwarm
        self.assertIsInstance(InnovationSwarm.description, str)
        self.assertTrue(len(InnovationSwarm.description) > 0)


# ---------------------------------------------------------------------------
# TestDivergencePhase
# ---------------------------------------------------------------------------

class TestDivergencePhase(unittest.TestCase):

    @patch("agents.innovation_swarm.log_json")
    @patch("agents.innovation_swarm.get_bot")
    def test_divergence_runs_each_technique(self, mock_get_bot, mock_log):
        from agents.innovation_swarm import InnovationSwarm

        bot_a = _make_mock_bot("tech_a", _make_ideas(3, technique="tech_a"))
        bot_b = _make_mock_bot("tech_b", _make_ideas(2, technique="tech_b"))

        def _get_bot_side_effect(name):
            return {"tech_a": bot_a, "tech_b": bot_b}[name]

        mock_get_bot.side_effect = _get_bot_side_effect

        swarm = InnovationSwarm()
        swarm.session_id = "test-sid"
        results = swarm._divergence_phase("test problem", ["tech_a", "tech_b"], "")

        bot_a.generate.assert_called_once_with("test problem", "")
        bot_b.generate.assert_called_once_with("test problem", "")
        self.assertIn("tech_a", results)
        self.assertIn("tech_b", results)
        self.assertEqual(results["tech_a"].idea_count, 3)
        self.assertEqual(results["tech_b"].idea_count, 2)

    @patch("agents.innovation_swarm.log_json")
    @patch("agents.innovation_swarm.get_bot")
    def test_divergence_handles_bot_failure(self, mock_get_bot, mock_log):
        """A failing bot is skipped; the other technique still produces results."""
        from agents.innovation_swarm import InnovationSwarm

        bad_bot = MagicMock()
        bad_bot.generate.side_effect = RuntimeError("bot exploded")

        good_bot = _make_mock_bot("good_tech", _make_ideas(4, technique="good_tech"))

        def _get_bot_side_effect(name):
            return {"bad_tech": bad_bot, "good_tech": good_bot}[name]

        mock_get_bot.side_effect = _get_bot_side_effect

        swarm = InnovationSwarm()
        swarm.session_id = "test-sid"
        results = swarm._divergence_phase(
            "test problem", ["bad_tech", "good_tech"], ""
        )

        # bad_tech should be absent, good_tech should be present
        self.assertNotIn("bad_tech", results)
        self.assertIn("good_tech", results)
        self.assertEqual(results["good_tech"].idea_count, 4)


# ---------------------------------------------------------------------------
# TestConvergencePhase
# ---------------------------------------------------------------------------

class TestConvergencePhase(unittest.TestCase):

    def _swarm(self):
        from agents.innovation_swarm import InnovationSwarm
        return InnovationSwarm()

    def test_convergence_filters_low_scoring_ideas(self):
        """Ideas below the default novelty/feasibility thresholds are removed."""
        swarm = self._swarm()
        low_ideas = _make_ideas(10, novelty=0.1, feasibility=0.1, impact=0.1)
        # Default min_novelty=0.5, min_feasibility=0.4 → all should be filtered out.
        # But _convergence_phase always returns at least 1 idea (max(1, ...)).
        # When scored_ideas is empty the slice is also empty, so we verify count < input.
        result = swarm._convergence_phase(low_ideas, {})
        # None of the low-scoring ideas pass thresholds; scored_ideas will be [].
        # num_to_select = min(max(1, int(10*0.2)), 20, 0) = min(2, 20, 0) = 0
        self.assertEqual(len(result), 0)

    def test_convergence_keeps_high_scoring_ideas(self):
        swarm = self._swarm()
        high_ideas = _make_ideas(10, novelty=0.9, feasibility=0.9, impact=0.8)
        result = swarm._convergence_phase(high_ideas, {})
        self.assertGreater(len(result), 0)
        for idea in result:
            self.assertGreaterEqual(idea.novelty, 0.5)
            self.assertGreaterEqual(idea.feasibility, 0.4)

    def test_convergence_respects_max_ideas(self):
        swarm = self._swarm()
        many_ideas = _make_ideas(100, novelty=0.9, feasibility=0.9, impact=0.8)
        result = swarm._convergence_phase(many_ideas, {"max_ideas": 5})
        self.assertLessEqual(len(result), 5)

    def test_convergence_empty_input_returns_empty(self):
        swarm = self._swarm()
        result = swarm._convergence_phase([], {})
        self.assertEqual(result, [])

    def test_convergence_composite_score_ordering(self):
        """Higher composite-score ideas are selected first."""
        from agents.innovation_swarm import InnovationSwarm
        swarm = InnovationSwarm()

        # idea_high: score = 0.4*0.9 + 0.3*0.9 + 0.3*0.9 = 0.9
        # idea_low:  score = 0.4*0.6 + 0.3*0.6 + 0.3*0.6 = 0.6
        idea_high = Idea(
            description="high", technique="t", novelty=0.9, feasibility=0.9, impact=0.9
        )
        idea_low = Idea(
            description="low", technique="t", novelty=0.6, feasibility=0.6, impact=0.6
        )
        # Use max_ideas=1 so only one idea is selected
        result = swarm._convergence_phase(
            [idea_low, idea_high], {"max_ideas": 1}
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].description, "high")


# ---------------------------------------------------------------------------
# TestMetricCalculation
# ---------------------------------------------------------------------------

class TestMetricCalculation(unittest.TestCase):

    def _swarm(self):
        from agents.innovation_swarm import InnovationSwarm
        return InnovationSwarm()

    def _tech_result(self, name, ideas):
        return TechniqueResult(
            technique=name, ideas=ideas, idea_count=len(ideas)
        )

    def test_calculate_diversity_all_produce(self):
        swarm = self._swarm()
        results = {
            "a": self._tech_result("a", _make_ideas(3)),
            "b": self._tech_result("b", _make_ideas(3)),
            "c": self._tech_result("c", _make_ideas(3)),
        }
        score = swarm._calculate_diversity(results)
        # 3 techniques, all produce equal idea counts → balance=1.0
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_calculate_diversity_some_empty(self):
        swarm = self._swarm()
        results_all = {
            "a": self._tech_result("a", _make_ideas(5)),
            "b": self._tech_result("b", _make_ideas(5)),
            "c": self._tech_result("c", _make_ideas(5)),
        }
        results_one = {
            "a": self._tech_result("a", _make_ideas(5)),
        }
        score_all = swarm._calculate_diversity(results_all)
        score_one = swarm._calculate_diversity(results_one)
        # More techniques → higher diversity
        self.assertGreater(score_all, score_one)

    def test_calculate_novelty(self):
        swarm = self._swarm()
        ideas = _make_ideas(4, novelty=0.8)
        score = swarm._calculate_novelty(ideas)
        self.assertAlmostEqual(score, 0.8, places=5)

    def test_calculate_feasibility(self):
        swarm = self._swarm()
        ideas = _make_ideas(4, feasibility=0.65)
        score = swarm._calculate_feasibility(ideas)
        self.assertAlmostEqual(score, 0.65, places=5)

    def test_empty_ideas_novelty_returns_zero(self):
        swarm = self._swarm()
        self.assertEqual(swarm._calculate_novelty([]), 0.0)

    def test_empty_ideas_feasibility_returns_zero(self):
        swarm = self._swarm()
        self.assertEqual(swarm._calculate_feasibility([]), 0.0)

    def test_empty_technique_results_diversity_returns_zero(self):
        swarm = self._swarm()
        self.assertEqual(swarm._calculate_diversity({}), 0.0)


# ---------------------------------------------------------------------------
# TestBrainstorm
# ---------------------------------------------------------------------------

class TestBrainstorm(unittest.TestCase):

    def _patched_brainstorm(self, techniques, ideas_per_bot=5):
        """Return (swarm, output) with bots mocked to return *ideas_per_bot* ideas each."""
        from agents.innovation_swarm import InnovationSwarm

        mock_bot = _make_mock_bot("mock_tech", _make_ideas(ideas_per_bot))

        bots_dict = {t: MagicMock() for t in techniques}

        with patch("agents.innovation_swarm.log_json"), \
             patch("agents.innovation_swarm.BRAINSTORMING_BOTS", bots_dict), \
             patch("agents.innovation_swarm.get_bot", return_value=mock_bot):
            swarm = InnovationSwarm()
            output = swarm.brainstorm(
                "How to improve code review?",
                techniques=techniques,
            )
        return swarm, output

    def test_brainstorm_returns_innovation_output(self):
        _, output = self._patched_brainstorm(["scamper", "six_hats"])
        self.assertIsInstance(output, InnovationOutput)

    def test_brainstorm_sets_session_id(self):
        swarm, _ = self._patched_brainstorm(["scamper"])
        self.assertIsNotNone(swarm.session_id)
        self.assertIsInstance(swarm.session_id, str)
        self.assertGreater(len(swarm.session_id), 0)

    def test_brainstorm_output_has_correct_problem_statement(self):
        from agents.innovation_swarm import InnovationSwarm

        mock_bot = _make_mock_bot("mock_tech", _make_ideas(3))
        problem = "Reduce build times by 50%"

        with patch("agents.innovation_swarm.log_json"), \
             patch("agents.innovation_swarm.BRAINSTORMING_BOTS", {"t": MagicMock()}), \
             patch("agents.innovation_swarm.get_bot", return_value=mock_bot):
            swarm = InnovationSwarm()
            output = swarm.brainstorm(problem, techniques=["t"])

        self.assertEqual(output.problem_statement, problem)

    def test_brainstorm_output_phase_is_convergence(self):
        _, output = self._patched_brainstorm(["scamper"])
        self.assertEqual(output.phase, InnovationPhase.CONVERGENCE)

    def test_brainstorm_default_techniques_uses_all_bots(self):
        """When techniques=None, all keys of BRAINSTORMING_BOTS are used."""
        from agents.innovation_swarm import InnovationSwarm

        bots_dict = {"t1": MagicMock(), "t2": MagicMock(), "t3": MagicMock()}
        mock_bot = _make_mock_bot("mock_tech", _make_ideas(3))

        with patch("agents.innovation_swarm.log_json"), \
             patch("agents.innovation_swarm.BRAINSTORMING_BOTS", bots_dict), \
             patch("agents.innovation_swarm.get_bot", return_value=mock_bot) as mock_get_bot:
            swarm = InnovationSwarm()
            output = swarm.brainstorm("problem with no techniques")

        self.assertEqual(set(output.techniques_used), set(bots_dict.keys()))

    def test_brainstorm_stores_in_brain_if_provided(self):
        from agents.innovation_swarm import InnovationSwarm

        brain = MagicMock()
        mock_bot = _make_mock_bot("mock_tech", _make_ideas(3))

        with patch("agents.innovation_swarm.log_json"), \
             patch("agents.innovation_swarm.BRAINSTORMING_BOTS", {"t": MagicMock()}), \
             patch("agents.innovation_swarm.get_bot", return_value=mock_bot):
            swarm = InnovationSwarm(brain=brain)
            swarm.brainstorm("test", techniques=["t"])

        brain.remember.assert_called_once()

    def test_brainstorm_no_brain_does_not_raise(self):
        """Calling brainstorm without a brain should not raise."""
        swarm, output = self._patched_brainstorm(["scamper"])
        self.assertIsInstance(output, InnovationOutput)


# ---------------------------------------------------------------------------
# TestRun
# ---------------------------------------------------------------------------

class TestRun(unittest.TestCase):

    def _run_with_input(self, input_data):
        from agents.innovation_swarm import InnovationSwarm

        mock_bot = _make_mock_bot("mock_tech", _make_ideas(5))
        bots_dict = {"t1": MagicMock()}

        with patch("agents.innovation_swarm.log_json"), \
             patch("agents.innovation_swarm.BRAINSTORMING_BOTS", bots_dict), \
             patch("agents.innovation_swarm.get_bot", return_value=mock_bot):
            swarm = InnovationSwarm()
            result = swarm.run(input_data)
        return result

    def test_run_extracts_task_key(self):
        result = self._run_with_input({"task": "reduce latency", "techniques": ["t1"]})
        self.assertIn("problem_statement", result)
        self.assertEqual(result["problem_statement"], "reduce latency")

    def test_run_extracts_problem_key(self):
        result = self._run_with_input({"problem": "improve UX", "techniques": ["t1"]})
        self.assertIn("problem_statement", result)
        self.assertEqual(result["problem_statement"], "improve UX")

    def test_run_task_takes_precedence_over_problem(self):
        """When both 'task' and 'problem' are present, 'task' wins."""
        result = self._run_with_input(
            {"task": "task value", "problem": "problem value", "techniques": ["t1"]}
        )
        self.assertEqual(result["problem_statement"], "task value")

    def test_run_returns_dict(self):
        result = self._run_with_input({"task": "any goal", "techniques": ["t1"]})
        self.assertIsInstance(result, dict)

    def test_run_result_contains_expected_keys(self):
        result = self._run_with_input({"task": "goal", "techniques": ["t1"]})
        for key in ("session_id", "all_ideas", "selected_ideas", "diversity_score"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_run_passes_techniques_to_brainstorm(self):
        from agents.innovation_swarm import InnovationSwarm

        mock_bot = _make_mock_bot("mock_tech", _make_ideas(3))
        bots_dict = {"alpha": MagicMock(), "beta": MagicMock()}

        with patch("agents.innovation_swarm.log_json"), \
             patch("agents.innovation_swarm.BRAINSTORMING_BOTS", bots_dict), \
             patch("agents.innovation_swarm.get_bot", return_value=mock_bot):
            swarm = InnovationSwarm()
            result = swarm.run({"task": "goal", "techniques": ["alpha", "beta"]})

        self.assertEqual(set(result["techniques_used"]), {"alpha", "beta"})

    def test_run_empty_problem_does_not_raise(self):
        """run() with an empty problem string should not raise."""
        result = self._run_with_input({"techniques": ["t1"]})
        self.assertIsInstance(result, dict)


# ---------------------------------------------------------------------------
# TestGetStats
# ---------------------------------------------------------------------------

class TestGetStats(unittest.TestCase):

    def test_get_stats_before_brainstorm(self):
        from agents.innovation_swarm import InnovationSwarm
        swarm = InnovationSwarm()
        stats = swarm.get_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("session_id", stats)
        self.assertIn("capabilities", stats)
        self.assertIsNone(stats["session_id"])

    @patch("agents.innovation_swarm.log_json")
    @patch("agents.innovation_swarm.BRAINSTORMING_BOTS", {"t": MagicMock()})
    @patch("agents.innovation_swarm.get_bot")
    def test_get_stats_after_brainstorm(self, mock_get_bot, mock_log):
        from agents.innovation_swarm import InnovationSwarm

        mock_get_bot.return_value = _make_mock_bot("t", _make_ideas(3))
        swarm = InnovationSwarm()
        swarm.brainstorm("test problem", techniques=["t"])
        stats = swarm.get_stats()

        self.assertIsNotNone(stats["session_id"])
        self.assertIsInstance(stats["capabilities"], list)
        self.assertGreater(len(stats["capabilities"]), 0)

    def test_get_stats_capabilities_match_class_attribute(self):
        from agents.innovation_swarm import InnovationSwarm
        swarm = InnovationSwarm()
        stats = swarm.get_stats()
        self.assertEqual(stats["capabilities"], InnovationSwarm.capabilities)


if __name__ == "__main__":
    unittest.main()
