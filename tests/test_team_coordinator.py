"""Tests for core.team_coordinator — multi-agent team coordination."""
import json
import unittest
from unittest.mock import MagicMock, patch

from core.team_coordinator import (
    SubGoal,
    TeamCoordinator,
    TeamResult,
    WorkerStatus,
)


class TestDecomposeGoal(unittest.TestCase):
    """Goal decomposition with a mock model."""

    def test_decompose_with_model(self):
        model = MagicMock(spec=["respond"])
        model.respond.return_value = json.dumps([
            {"description": "Write unit tests", "priority": 2, "dependencies": []},
            {"description": "Refactor module", "priority": 1, "dependencies": []},
        ])
        coord = TeamCoordinator(model=model)
        sub_goals = coord.decompose_goal("Improve code quality")

        self.assertEqual(len(sub_goals), 2)
        self.assertEqual(sub_goals[0].description, "Write unit tests")
        self.assertEqual(sub_goals[0].priority, 2)
        self.assertEqual(sub_goals[1].description, "Refactor module")
        model.respond.assert_called_once()

    def test_decompose_uses_respond_for_role_when_available(self):
        model = MagicMock()
        model.respond_for_role.return_value = json.dumps([
            {"description": "Task A", "priority": 1, "dependencies": []},
        ])
        coord = TeamCoordinator(model=model)
        sub_goals = coord.decompose_goal("Do something")

        self.assertEqual(len(sub_goals), 1)
        model.respond_for_role.assert_called_once()
        # respond should NOT have been called since respond_for_role exists
        model.respond.assert_not_called()

    def test_decompose_with_dependencies(self):
        model = MagicMock(spec=["respond"])
        model.respond.return_value = json.dumps([
            {"description": "Setup DB", "priority": 2, "dependencies": []},
            {"description": "Run migrations", "priority": 1, "dependencies": [0]},
        ])
        coord = TeamCoordinator(model=model)
        sub_goals = coord.decompose_goal("Database setup")

        self.assertEqual(len(sub_goals), 2)
        self.assertEqual(sub_goals[0].dependencies, [])
        # Second sub-goal depends on the first
        self.assertEqual(sub_goals[1].dependencies, [sub_goals[0].id])


class TestDecomposeGoalFallback(unittest.TestCase):
    """Decomposition fallback when no model is available."""

    def test_no_model_returns_single_goal(self):
        coord = TeamCoordinator(model=None)
        sub_goals = coord.decompose_goal("Build feature X")

        self.assertEqual(len(sub_goals), 1)
        self.assertEqual(sub_goals[0].description, "Build feature X")
        self.assertEqual(sub_goals[0].parent_goal, "Build feature X")

    def test_model_raises_returns_fallback(self):
        model = MagicMock(spec=["respond"])
        model.respond.side_effect = RuntimeError("API down")
        coord = TeamCoordinator(model=model)
        sub_goals = coord.decompose_goal("Build feature X")

        self.assertEqual(len(sub_goals), 1)
        self.assertEqual(sub_goals[0].description, "Build feature X")


class TestExecuteTeamParallel(unittest.TestCase):
    """Execute team with parallel sub-goals."""

    def test_parallel_execution(self):
        factory = MagicMock()
        mock_orch = MagicMock()
        mock_orch.run_cycle.return_value = {"status": "ok"}
        factory.return_value = mock_orch

        coord = TeamCoordinator(orchestrator_factory=factory, max_workers=3)
        sub_goals = [
            SubGoal(description="Task A", parent_goal="Goal"),
            SubGoal(description="Task B", parent_goal="Goal"),
            SubGoal(description="Task C", parent_goal="Goal"),
        ]
        result = coord.execute_team("Goal", sub_goals)

        self.assertEqual(result.success_count, 3)
        self.assertEqual(result.failure_count, 0)
        self.assertGreater(result.total_duration, 0)
        for sg in sub_goals:
            self.assertEqual(sg.status, WorkerStatus.COMPLETED)

    def test_parallel_with_failure(self):
        factory = MagicMock()
        mock_orch = MagicMock()
        mock_orch.run_cycle.side_effect = [{"status": "ok"}, RuntimeError("boom"), {"status": "ok"}]
        factory.return_value = mock_orch

        coord = TeamCoordinator(orchestrator_factory=factory, max_workers=3)
        sub_goals = [
            SubGoal(description="Task A", parent_goal="Goal"),
            SubGoal(description="Task B", parent_goal="Goal"),
            SubGoal(description="Task C", parent_goal="Goal"),
        ]
        result = coord.execute_team("Goal", sub_goals)

        self.assertEqual(result.success_count, 2)
        self.assertEqual(result.failure_count, 1)


class TestExecuteTeamDependencies(unittest.TestCase):
    """Execute team with dependencies between sub-goals."""

    def test_dependent_runs_after_independent(self):
        factory = MagicMock()
        mock_orch = MagicMock()
        mock_orch.run_cycle.return_value = {"status": "ok"}
        factory.return_value = mock_orch

        sg_a = SubGoal(id="aaa", description="Independent", parent_goal="Goal")
        sg_b = SubGoal(id="bbb", description="Dependent", parent_goal="Goal",
                       dependencies=["aaa"])

        coord = TeamCoordinator(orchestrator_factory=factory)
        result = coord.execute_team("Goal", [sg_a, sg_b])

        self.assertEqual(result.success_count, 2)
        self.assertEqual(result.failure_count, 0)
        self.assertEqual(sg_a.status, WorkerStatus.COMPLETED)
        self.assertEqual(sg_b.status, WorkerStatus.COMPLETED)

    def test_dependent_fails_when_dependency_not_met(self):
        factory = MagicMock()
        mock_orch = MagicMock()
        mock_orch.run_cycle.side_effect = RuntimeError("fail")
        factory.return_value = mock_orch

        sg_a = SubGoal(id="aaa", description="Independent", parent_goal="Goal")
        sg_b = SubGoal(id="bbb", description="Dependent", parent_goal="Goal",
                       dependencies=["aaa"])

        coord = TeamCoordinator(orchestrator_factory=factory)
        result = coord.execute_team("Goal", [sg_a, sg_b])

        self.assertEqual(sg_a.status, WorkerStatus.FAILED)
        self.assertEqual(sg_b.status, WorkerStatus.FAILED)
        self.assertIn("dependencies not met", sg_b.result.get("error", ""))


class TestWorkerStatusTracking(unittest.TestCase):
    """Worker status tracking throughout lifecycle."""

    def test_initial_status_is_pending(self):
        sg = SubGoal(description="test")
        self.assertEqual(sg.status, WorkerStatus.PENDING)

    def test_status_transitions(self):
        sg = SubGoal(description="test")
        self.assertEqual(sg.status, WorkerStatus.PENDING)

        sg.status = WorkerStatus.RUNNING
        self.assertEqual(sg.status, WorkerStatus.RUNNING)

        sg.status = WorkerStatus.COMPLETED
        self.assertEqual(sg.status, WorkerStatus.COMPLETED)

    def test_status_values(self):
        self.assertEqual(WorkerStatus.PENDING.value, "pending")
        self.assertEqual(WorkerStatus.RUNNING.value, "running")
        self.assertEqual(WorkerStatus.COMPLETED.value, "completed")
        self.assertEqual(WorkerStatus.FAILED.value, "failed")
        self.assertEqual(WorkerStatus.CANCELED.value, "canceled")

    def test_timestamps_set_during_execution(self):
        factory = MagicMock()
        mock_orch = MagicMock()
        mock_orch.run_cycle.return_value = {"status": "ok"}
        factory.return_value = mock_orch

        coord = TeamCoordinator(orchestrator_factory=factory)
        sg = SubGoal(description="Task", parent_goal="Goal")
        coord.execute_team("Goal", [sg])

        self.assertGreater(sg.started_at, 0)
        self.assertGreater(sg.completed_at, 0)
        self.assertGreaterEqual(sg.completed_at, sg.started_at)


class TestTeamResultAggregation(unittest.TestCase):
    """Team result aggregation."""

    def test_empty_team(self):
        coord = TeamCoordinator()
        result = coord.execute_team("Goal", [])

        self.assertEqual(result.goal, "Goal")
        self.assertEqual(result.success_count, 0)
        self.assertEqual(result.failure_count, 0)
        self.assertEqual(result.sub_goals, [])

    def test_mixed_results(self):
        factory = MagicMock()
        mock_orch = MagicMock()
        call_count = 0

        def side_effect(desc, dry_run=False):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("fail")
            return {"status": "ok"}

        mock_orch.run_cycle.side_effect = side_effect
        factory.return_value = mock_orch

        coord = TeamCoordinator(orchestrator_factory=factory, max_workers=1)
        sub_goals = [
            SubGoal(description="A", parent_goal="Goal"),
            SubGoal(description="B", parent_goal="Goal"),
            SubGoal(description="C", parent_goal="Goal"),
        ]
        result = coord.execute_team("Goal", sub_goals)

        self.assertEqual(result.success_count, 2)
        self.assertEqual(result.failure_count, 1)
        self.assertGreater(result.total_duration, 0)

    def test_get_team_status(self):
        coord = TeamCoordinator()
        sg = SubGoal(id="abc", description="Do something long", parent_goal="Goal",
                     status=WorkerStatus.COMPLETED, priority=3)
        result = TeamResult(goal="Goal", sub_goals=[sg], success_count=1,
                            failure_count=0, total_duration=1.5)
        coord.active_teams["t1"] = result

        status = coord.get_team_status("t1")
        self.assertIsNotNone(status)
        self.assertEqual(status["goal"], "Goal")
        self.assertEqual(len(status["sub_goals"]), 1)
        self.assertEqual(status["sub_goals"][0]["id"], "abc")
        self.assertEqual(status["sub_goals"][0]["status"], "completed")
        self.assertEqual(status["success"], 1)

    def test_get_team_status_not_found(self):
        coord = TeamCoordinator()
        self.assertIsNone(coord.get_team_status("nonexistent"))


class TestParseSubGoals(unittest.TestCase):
    """Parse sub-goals from JSON response."""

    def test_valid_json_array(self):
        coord = TeamCoordinator()
        response = json.dumps([
            {"description": "Task 1", "priority": 2, "dependencies": []},
            {"description": "Task 2", "priority": 1, "dependencies": [0]},
        ])
        result = coord._parse_sub_goals(response, "Parent")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].description, "Task 1")
        self.assertEqual(result[0].priority, 2)
        self.assertEqual(result[0].parent_goal, "Parent")
        self.assertEqual(result[1].dependencies, [result[0].id])

    def test_json_embedded_in_text(self):
        coord = TeamCoordinator()
        response = 'Here are the sub-goals:\n[{"description": "Only task", "priority": 1}]\nDone.'
        result = coord._parse_sub_goals(response, "Parent")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].description, "Only task")

    def test_max_five_sub_goals(self):
        coord = TeamCoordinator()
        items = [{"description": f"Task {i}", "priority": 1} for i in range(10)]
        response = json.dumps(items)
        result = coord._parse_sub_goals(response, "Parent")

        self.assertLessEqual(len(result), 5)


class TestParseSubGoalsFallback(unittest.TestCase):
    """Parse sub-goals fallback for invalid responses."""

    def test_invalid_json(self):
        coord = TeamCoordinator()
        result = coord._parse_sub_goals("not json at all", "Fallback goal")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].description, "Fallback goal")

    def test_empty_array(self):
        coord = TeamCoordinator()
        result = coord._parse_sub_goals("[]", "Fallback goal")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].description, "Fallback goal")

    def test_non_dict_items(self):
        coord = TeamCoordinator()
        result = coord._parse_sub_goals('["just", "strings"]', "Fallback goal")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].description, "Fallback goal")

    def test_fallback_no_model(self):
        """No model means fallback to single sub-goal wrapping the original."""
        coord = TeamCoordinator()
        result = coord._parse_sub_goals("", "My goal")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].description, "My goal")


if __name__ == "__main__":
    unittest.main()
