"""Tests for core/evolution_loop.py — EvolutionLoop."""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.evolution_loop import EvolutionLoop, InnovationProposal


def _make_loop(**overrides):
    """Create an EvolutionLoop with all agents mocked."""
    defaults = dict(
        planner=MagicMock(),
        coder=MagicMock(),
        critic=MagicMock(),
        brain=MagicMock(),
        vector_store=MagicMock(),
        git_tools=MagicMock(),
        mutator=MagicMock(),
        improvement_service=None,
        goal_queue=None,
        orchestrator=None,
        project_root="/tmp/test_project",
        skills={},
        auto_execute_queued=False,
        innovation_goal_limit=2,
    )
    defaults.update(overrides)
    with patch("core.evolution_loop.ExperimentTracker"), patch("core.evolution_loop.MetricsCollector"):
        return EvolutionLoop(**defaults)


class TestInnovationProposal(unittest.TestCase):
    def test_as_dict(self):
        p = InnovationProposal(
            proposal_id="test:1",
            title="Test",
            category="skill",
            goal="test goal",
            rationale="because",
            evidence=["ev1"],
            smallest_surface="foo.py",
            expected_value="high",
            risk_level="low",
            verification_cost="unit tests",
            recommended_action="queue",
        )
        d = p.as_dict()
        self.assertEqual(d["proposal_id"], "test:1")
        self.assertEqual(d["evidence"], ["ev1"])

    def test_frozen_dataclass(self):
        p = InnovationProposal(
            proposal_id="id",
            title="t",
            category="c",
            goal="g",
            rationale="r",
            evidence=[],
            smallest_surface="s",
            expected_value="v",
            risk_level="l",
            verification_cost="v",
            recommended_action="a",
        )
        with self.assertRaises(AttributeError):
            p.title = "new"


class TestHypothesize(unittest.TestCase):
    def test_returns_string_from_list(self):
        loop = _make_loop()
        loop.planner.plan.return_value = ["step1", "step2"]
        result = loop._hypothesize("goal", "mem", "past", "weak")
        self.assertEqual(result, "step1\nstep2")

    def test_returns_string_from_string(self):
        loop = _make_loop()
        loop.planner.plan.return_value = "hypothesis text"
        result = loop._hypothesize("goal", "mem", "past", "weak")
        self.assertEqual(result, "hypothesis text")


class TestDecomposeTasks(unittest.TestCase):
    def test_returns_task_list(self):
        loop = _make_loop()
        loop.planner.plan.return_value = ["task1", "task2", "task3"]
        result = loop._decompose_tasks("hypothesis", "mem", "past", "weak")
        self.assertEqual(result, ["task1", "task2", "task3"])


class TestImplementAndCritique(unittest.TestCase):
    def test_returns_implementation_and_evaluation(self):
        loop = _make_loop()
        loop.coder.implement.return_value = "code here"
        loop.critic.critique_code.return_value = "looks good"
        loop.brain.analyze_critique_for_weaknesses = MagicMock()

        impl, ev = loop._implement_and_critique("goal", ["task1", "task2"])
        self.assertEqual(impl, "code here")
        self.assertEqual(ev, "looks good")
        loop.coder.implement.assert_called_once_with("task1\ntask2")
        loop.brain.analyze_critique_for_weaknesses.assert_called_once_with("looks good")


class TestParseValidationResult(unittest.TestCase):
    def test_approved_json(self):
        loop = _make_loop()
        raw = json.dumps({"decision": "APPROVED", "confidence_score": 0.85, "impact_assessment": "positive", "reasoning": "good"})
        decision, score = loop._parse_validation_result(raw)
        self.assertEqual(decision, "APPROVED")
        self.assertAlmostEqual(score, 0.85)

    def test_rejected_json(self):
        loop = _make_loop()
        raw = json.dumps({"decision": "REJECTED", "confidence_score": 0.3})
        decision, score = loop._parse_validation_result(raw)
        self.assertEqual(decision, "REJECTED")
        self.assertAlmostEqual(score, 0.3)

    def test_invalid_json(self):
        loop = _make_loop()
        decision, score = loop._parse_validation_result("not json at all")
        self.assertEqual(decision, "REJECTED")
        self.assertEqual(score, 0.0)

    def test_non_dict_json(self):
        loop = _make_loop()
        decision, score = loop._parse_validation_result('"just a string"')
        self.assertEqual(decision, "REJECTED")
        self.assertEqual(score, 0.0)

    def test_missing_fields_uses_defaults(self):
        loop = _make_loop()
        raw = json.dumps({})
        decision, score = loop._parse_validation_result(raw)
        self.assertEqual(decision, "REJECTED")
        self.assertEqual(score, 0.0)


class TestMutationPlanToDSL(unittest.TestCase):
    def test_replace_in_file(self):
        loop = _make_loop()
        plan = {"mutations": [{"file_path": "a.py", "old_content": "old", "new_content": "new"}]}
        dsl = loop._mutation_plan_to_dsl(plan)
        self.assertIn("REPLACE_IN_FILE a.py", dsl)
        self.assertIn("old", dsl)
        self.assertIn("new", dsl)

    def test_add_file(self):
        loop = _make_loop()
        plan = {"mutations": [{"file_path": "b.py", "new_content": "content"}]}
        dsl = loop._mutation_plan_to_dsl(plan)
        self.assertIn("ADD_FILE b.py", dsl)

    def test_empty_mutations(self):
        loop = _make_loop()
        plan = {"mutations": []}
        dsl = loop._mutation_plan_to_dsl(plan)
        self.assertEqual(dsl, "")

    def test_skips_invalid_entries(self):
        loop = _make_loop()
        plan = {"mutations": ["not a dict", {"file_path": "", "new_content": "x"}, {"new_content": "y"}]}
        dsl = loop._mutation_plan_to_dsl(plan)
        self.assertEqual(dsl.strip(), "")


class TestSelectProposals(unittest.TestCase):
    def _make_proposal(self, category, risk="medium", title="Test"):
        return InnovationProposal(
            proposal_id=f"{category}:test",
            title=title,
            category=category,
            goal="g",
            rationale="r",
            evidence=[],
            smallest_surface="s",
            expected_value="h",
            risk_level=risk,
            verification_cost="v",
            recommended_action="queue",
        )

    def test_capability_focus(self):
        loop = _make_loop()
        proposals = [
            self._make_proposal("verification"),
            self._make_proposal("skill"),
            self._make_proposal("mcp"),
        ]
        selected = loop._select_proposals(proposals, focus="capability", proposal_limit=2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0].category, "skill")

    def test_quality_focus(self):
        loop = _make_loop()
        proposals = [
            self._make_proposal("skill"),
            self._make_proposal("verification"),
        ]
        selected = loop._select_proposals(proposals, focus="quality", proposal_limit=1)
        self.assertEqual(selected[0].category, "verification")

    def test_limit_respected(self):
        loop = _make_loop()
        proposals = [self._make_proposal("skill", title=f"P{i}") for i in range(5)]
        selected = loop._select_proposals(proposals, focus="capability", proposal_limit=3)
        self.assertEqual(len(selected), 3)


class TestQueueSelectedGoals(unittest.TestCase):
    def test_dry_run(self):
        loop = _make_loop()
        proposals = [
            InnovationProposal(
                proposal_id="t:1",
                title="T",
                category="skill",
                goal="g",
                rationale="r",
                evidence=[],
                smallest_surface="s",
                expected_value="h",
                risk_level="l",
                verification_cost="v",
                recommended_action="queue",
            )
        ]
        result = loop._queue_selected_goals(proposals, dry_run=True)
        self.assertFalse(result["attempted"])
        self.assertEqual(len(result["skipped"]), 1)
        self.assertEqual(result["skipped"][0]["reason"], "dry_run")

    def test_no_goal_queue(self):
        loop = _make_loop(goal_queue=None)
        proposals = [
            InnovationProposal(
                proposal_id="t:1",
                title="T",
                category="skill",
                goal="g",
                rationale="r",
                evidence=[],
                smallest_surface="s",
                expected_value="h",
                risk_level="l",
                verification_cost="v",
                recommended_action="queue",
            )
        ]
        result = loop._queue_selected_goals(proposals, dry_run=False)
        self.assertFalse(result["attempted"])

    def test_empty_proposals(self):
        loop = _make_loop()
        result = loop._queue_selected_goals([], dry_run=False)
        self.assertFalse(result["attempted"])

    def test_prepend_batch(self):
        mock_queue = MagicMock()
        mock_queue.queue = []
        mock_queue.prepend_batch = MagicMock()
        loop = _make_loop(goal_queue=mock_queue)
        proposals = [
            InnovationProposal(
                proposal_id="t:1",
                title="T",
                category="skill",
                goal="new goal",
                rationale="r",
                evidence=[],
                smallest_surface="s",
                expected_value="h",
                risk_level="l",
                verification_cost="v",
                recommended_action="queue",
            )
        ]
        result = loop._queue_selected_goals(proposals, dry_run=False)
        self.assertTrue(result["attempted"])
        mock_queue.prepend_batch.assert_called_once()


class TestExecuteSelectedGoals(unittest.TestCase):
    def test_dry_run(self):
        loop = _make_loop()
        result = loop._execute_selected_goals(["goal1"], dry_run=True, execution_limit=1)
        self.assertFalse(result["attempted"])

    def test_no_orchestrator(self):
        loop = _make_loop(orchestrator=None)
        result = loop._execute_selected_goals(["goal1"], dry_run=False, execution_limit=1)
        self.assertFalse(result["attempted"])

    def test_empty_goals(self):
        loop = _make_loop()
        result = loop._execute_selected_goals([], dry_run=False, execution_limit=1)
        self.assertFalse(result["attempted"])

    def test_executes_with_orchestrator(self):
        mock_orch = MagicMock()
        mock_orch.run_loop.return_value = {"stop_reason": "done", "history": []}
        loop = _make_loop(orchestrator=mock_orch)
        result = loop._execute_selected_goals(["goal1"], dry_run=False, execution_limit=1)
        self.assertTrue(result["attempted"])
        mock_orch.run_loop.assert_called_once()


class TestOnCycleComplete(unittest.TestCase):
    def test_triggers_after_n_cycles(self):
        loop = _make_loop()
        loop.TRIGGER_EVERY_N = 2
        loop.run = MagicMock()
        loop.on_cycle_complete({"goal": "g"})
        loop.run.assert_not_called()
        loop.on_cycle_complete({"goal": "g"})
        loop.run.assert_called_once()

    def test_triggers_on_hotspot_signal(self):
        loop = _make_loop()
        loop.TRIGGER_EVERY_N = 999
        loop.run = MagicMock()
        entry = {"goal": "refactor hotspot in orchestrator"}
        loop.on_cycle_complete(entry)
        loop.run.assert_called_once()


class TestPersistMemories(unittest.TestCase):
    def test_stores_in_brain_and_vector(self):
        loop = _make_loop()
        loop._persist_memories("goal", "hyp", "eval", "mutation")
        self.assertEqual(loop.brain.remember.call_count, 4)
        self.assertEqual(loop.vector.add.call_count, 2)

    def test_no_vector_store(self):
        loop = _make_loop(vector_store=None)
        loop._persist_memories("goal", "hyp", "eval", "mutation")
        self.assertEqual(loop.brain.remember.call_count, 4)


if __name__ == "__main__":
    unittest.main()
