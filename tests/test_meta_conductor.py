"""Unit tests for agents/meta_conductor.py — MetaConductor 5-phase orchestrator."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.schemas import (
    Idea,
    InnovationOutput,
    InnovationPhase,
    InnovationSessionState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_idea(description: str = "Test idea", novelty: float = 0.8) -> Idea:
    return Idea(
        description=description,
        technique="SCAMPER",
        novelty=novelty,
        feasibility=0.7,
        impact=0.6,
    )


def _make_innovation_output(
    session_id: str = "test1234",
    problem: str = "Test problem",
    num_ideas: int = 5,
    num_selected: int = 2,
) -> InnovationOutput:
    ideas = [_make_idea(f"Idea {i}") for i in range(num_ideas)]
    selected = ideas[:num_selected]
    return InnovationOutput(
        session_id=session_id,
        problem_statement=problem,
        phase=InnovationPhase.CONVERGENCE,
        techniques_used=["SCAMPER"],
        all_ideas=ideas,
        selected_ideas=selected,
        technique_results={},
        diversity_score=0.8,
        novelty_score=0.75,
        feasibility_score=0.65,
        total_ideas_generated=num_ideas,
        total_ideas_selected=num_selected,
    )


# ---------------------------------------------------------------------------
# TestMetaConductorInit
# ---------------------------------------------------------------------------


class TestMetaConductorInit(unittest.TestCase):
    """Tests for MetaConductor.__init__."""

    @patch("agents.meta_conductor.log_json")
    @patch("agents.meta_conductor.InnovationSwarm")
    def test_defaults(self, mock_swarm_cls, mock_log):
        from agents.meta_conductor import MetaConductor

        conductor = MetaConductor()

        self.assertIsNone(conductor.brain)
        self.assertIsNone(conductor.model)
        self.assertIsInstance(conductor.active_sessions, dict)
        self.assertEqual(len(conductor.active_sessions), 0)
        mock_swarm_cls.assert_called_once_with(brain=None, model=None, use_llm=True)

    @patch("agents.meta_conductor.log_json")
    @patch("agents.meta_conductor.InnovationSwarm")
    def test_with_args(self, mock_swarm_cls, mock_log):
        from agents.meta_conductor import MetaConductor

        brain = MagicMock()
        model = MagicMock()
        conductor = MetaConductor(brain=brain, model=model)

        self.assertIs(conductor.brain, brain)
        self.assertIs(conductor.model, model)
        mock_swarm_cls.assert_called_once_with(brain=brain, model=model, use_llm=True)

    @patch("agents.meta_conductor.log_json")
    @patch("agents.meta_conductor.InnovationSwarm")
    def test_phase_order(self, mock_swarm_cls, mock_log):
        from agents.meta_conductor import MetaConductor

        expected = [
            InnovationPhase.IMMERSION,
            InnovationPhase.DIVERGENCE,
            InnovationPhase.CONVERGENCE,
            InnovationPhase.INCUBATION,
            InnovationPhase.TRANSFORMATION,
        ]
        self.assertEqual(MetaConductor.PHASE_ORDER, expected)
        self.assertEqual(len(MetaConductor.PHASE_ORDER), 5)


# ---------------------------------------------------------------------------
# TestStartSession
# ---------------------------------------------------------------------------


class TestStartSession(unittest.TestCase):
    """Tests for MetaConductor.start_session."""

    def setUp(self):
        patcher_swarm = patch("agents.meta_conductor.InnovationSwarm")
        patcher_log = patch("agents.meta_conductor.log_json")
        self.mock_swarm_cls = patcher_swarm.start()
        self.mock_log = patcher_log.start()
        self.addCleanup(patcher_swarm.stop)
        self.addCleanup(patcher_log.stop)

        from agents.meta_conductor import MetaConductor

        self.conductor = MetaConductor()

    def test_creates_session(self):
        session = self.conductor.start_session("Improve code review")

        self.assertIsInstance(session, InnovationSessionState)
        self.assertIn(session.session_id, self.conductor.active_sessions)

    def test_session_has_id(self):
        session = self.conductor.start_session("Some problem")
        self.assertEqual(len(session.session_id), 8)

    def test_session_problem_stored(self):
        problem = "How might we reduce cycle time?"
        session = self.conductor.start_session(problem)
        self.assertEqual(session.problem_statement, problem)

    def test_default_constraints(self):
        session = self.conductor.start_session("Test problem", constraints=None)

        self.assertIn("selection_ratio", session.constraints)
        self.assertIn("min_novelty", session.constraints)
        self.assertIn("min_feasibility", session.constraints)
        self.assertIn("max_ideas", session.constraints)
        self.assertIn("diversity_threshold", session.constraints)

    def test_custom_constraints(self):
        custom = {"selection_ratio": 0.5, "max_ideas": 10}
        session = self.conductor.start_session("Test problem", constraints=custom)
        self.assertEqual(session.constraints, custom)

    def test_session_status_active(self):
        session = self.conductor.start_session("Active session problem")
        self.assertEqual(session.status, "active")

    def test_unique_session_ids(self):
        session_a = self.conductor.start_session("Problem A")
        session_b = self.conductor.start_session("Problem B")
        self.assertNotEqual(session_a.session_id, session_b.session_id)

    def test_session_stored_in_active_sessions(self):
        session = self.conductor.start_session("Persisted problem")
        retrieved = self.conductor.active_sessions[session.session_id]
        self.assertIs(retrieved, session)

    def test_custom_techniques_stored(self):
        techniques = ["SCAMPER", "Mind Mapping"]
        session = self.conductor.start_session("Problem", techniques=techniques)
        self.assertEqual(session.techniques, techniques)

    def test_brain_save_called_when_brain_present(self):
        brain = MagicMock()
        from agents.meta_conductor import MetaConductor

        conductor = MetaConductor(brain=brain)
        conductor.start_session("Problem with brain")
        brain.save_innovation_session.assert_called_once()


# ---------------------------------------------------------------------------
# TestResumeSession
# ---------------------------------------------------------------------------


class TestResumeSession(unittest.TestCase):
    """Tests for MetaConductor.resume_session."""

    def setUp(self):
        patcher_swarm = patch("agents.meta_conductor.InnovationSwarm")
        patcher_log = patch("agents.meta_conductor.log_json")
        self.mock_swarm_cls = patcher_swarm.start()
        self.mock_log = patcher_log.start()
        self.addCleanup(patcher_swarm.stop)
        self.addCleanup(patcher_log.stop)

        from agents.meta_conductor import MetaConductor

        self.conductor = MetaConductor()

    def test_resume_existing(self):
        session = self.conductor.start_session("Resumable problem")
        resumed = self.conductor.resume_session(session.session_id)

        self.assertIs(resumed, session)
        self.assertEqual(resumed.status, "active")

    def test_resume_nonexistent_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.conductor.resume_session("nonexistent")
        self.assertIn("nonexistent", str(ctx.exception))

    def test_resume_logs_current_phase(self):
        session = self.conductor.start_session("Logged problem")
        self.conductor.resume_session(session.session_id)
        # log_json should have been called for both start and resume
        self.assertGreaterEqual(self.mock_log.call_count, 2)


# ---------------------------------------------------------------------------
# TestExecutePhase
# ---------------------------------------------------------------------------


class TestExecutePhase(unittest.TestCase):
    """Tests for MetaConductor.execute_phase."""

    def setUp(self):
        patcher_swarm = patch("agents.meta_conductor.InnovationSwarm")
        patcher_log = patch("agents.meta_conductor.log_json")
        self.mock_swarm_cls = patcher_swarm.start()
        self.mock_log = patcher_log.start()
        self.addCleanup(patcher_swarm.stop)
        self.addCleanup(patcher_log.stop)

        from agents.meta_conductor import MetaConductor

        self.conductor = MetaConductor()

    def test_execute_immersion(self):
        session = self.conductor.start_session("Immersion test problem")
        result = self.conductor.execute_phase(session.session_id, InnovationPhase.IMMERSION)

        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("phase"), InnovationPhase.IMMERSION.value)
        self.assertIn("problem_analysis", result)
        self.assertIn("ready_for_divergence", result)

    def test_execute_divergence(self):
        session = self.conductor.start_session("Divergence test problem")
        mock_output = _make_innovation_output(session_id=session.session_id)
        self.conductor.innovation_swarm.brainstorm = MagicMock(return_value=mock_output)

        result = self.conductor.execute_phase(session.session_id, InnovationPhase.DIVERGENCE)

        self.assertEqual(result.get("phase"), InnovationPhase.DIVERGENCE.value)
        self.assertIs(session.output, mock_output)
        self.assertEqual(session.ideas_generated, mock_output.total_ideas_generated)
        self.conductor.innovation_swarm.brainstorm.assert_called_once()

    def test_execute_divergence_passes_problem_to_swarm(self):
        problem = "Specific problem statement for swarm"
        session = self.conductor.start_session(problem)
        mock_output = _make_innovation_output()
        self.conductor.innovation_swarm.brainstorm = MagicMock(return_value=mock_output)

        self.conductor.execute_phase(session.session_id, InnovationPhase.DIVERGENCE)

        call_kwargs = self.conductor.innovation_swarm.brainstorm.call_args
        self.assertEqual(call_kwargs.kwargs.get("problem_statement"), problem)

    def test_execute_unknown_session_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.conductor.execute_phase("does-not-exist", InnovationPhase.IMMERSION)
        self.assertIn("does-not-exist", str(ctx.exception))

    def test_phase_advances_after_execute(self):
        session = self.conductor.start_session("Advance phase problem")
        initial_completed = len(session.phases_completed)

        self.conductor.execute_phase(session.session_id, InnovationPhase.IMMERSION)

        self.assertEqual(len(session.phases_completed), initial_completed + 1)
        self.assertIn(InnovationPhase.IMMERSION, session.phases_completed)

    def test_current_phase_advances_to_next(self):
        session = self.conductor.start_session("Next phase problem")
        # After executing IMMERSION the current phase should become DIVERGENCE
        self.conductor.execute_phase(session.session_id, InnovationPhase.IMMERSION)
        self.assertEqual(session.current_phase, InnovationPhase.DIVERGENCE)

    def test_execute_convergence_sets_ideas_selected(self):
        session = self.conductor.start_session("Convergence problem")
        mock_output = _make_innovation_output(num_selected=3)
        session.output = mock_output

        result = self.conductor.execute_phase(session.session_id, InnovationPhase.CONVERGENCE)

        self.assertEqual(session.ideas_selected, 3)
        self.assertEqual(result.get("phase"), InnovationPhase.CONVERGENCE.value)

    def test_execute_convergence_without_output_raises(self):
        session = self.conductor.start_session("No-output convergence problem")
        session.output = None

        with self.assertRaises(ValueError):
            self.conductor.execute_phase(session.session_id, InnovationPhase.CONVERGENCE)

    def test_execute_incubation(self):
        session = self.conductor.start_session("Incubation problem")
        mock_output = _make_innovation_output()
        session.output = mock_output

        result = self.conductor.execute_phase(session.session_id, InnovationPhase.INCUBATION)

        self.assertEqual(result.get("phase"), InnovationPhase.INCUBATION.value)
        self.assertIn("ready_for_transformation", result)

    def test_execute_transformation_sets_completed(self):
        session = self.conductor.start_session("Transformation problem")
        mock_output = _make_innovation_output(num_ideas=3, num_selected=2)
        session.output = mock_output

        result = self.conductor.execute_phase(session.session_id, InnovationPhase.TRANSFORMATION)

        self.assertEqual(session.status, "completed")
        self.assertEqual(result.get("phase"), InnovationPhase.TRANSFORMATION.value)
        self.assertIn("actionable_tasks", result)

    def test_execute_transformation_tasks_have_session_id_prefix(self):
        session = self.conductor.start_session("Task prefix problem")
        mock_output = _make_innovation_output(num_selected=2)
        session.output = mock_output

        result = self.conductor.execute_phase(session.session_id, InnovationPhase.TRANSFORMATION)

        tasks = result.get("actionable_tasks", [])
        for task in tasks:
            self.assertTrue(
                task["task_id"].startswith(session.session_id),
                f"task_id '{task['task_id']}' should start with session_id '{session.session_id}'",
            )

    def test_last_phase_sets_status_completed(self):
        session = self.conductor.start_session("Final phase problem")
        mock_output = _make_innovation_output()
        session.output = mock_output
        # Execute the last phase in PHASE_ORDER (TRANSFORMATION)
        self.conductor.execute_phase(session.session_id, InnovationPhase.TRANSFORMATION)
        self.assertEqual(session.status, "completed")

    def test_uses_current_phase_when_phase_is_none(self):
        session = self.conductor.start_session("Default phase problem")
        # Current phase starts as IMMERSION
        self.assertEqual(session.current_phase, InnovationPhase.IMMERSION)
        result = self.conductor.execute_phase(session.session_id)  # phase=None
        self.assertEqual(result.get("phase"), InnovationPhase.IMMERSION.value)


# ---------------------------------------------------------------------------
# TestGetSession
# ---------------------------------------------------------------------------


class TestGetSession(unittest.TestCase):
    """Tests for MetaConductor.get_session."""

    def setUp(self):
        patcher_swarm = patch("agents.meta_conductor.InnovationSwarm")
        patcher_log = patch("agents.meta_conductor.log_json")
        self.mock_swarm_cls = patcher_swarm.start()
        self.mock_log = patcher_log.start()
        self.addCleanup(patcher_swarm.stop)
        self.addCleanup(patcher_log.stop)

        from agents.meta_conductor import MetaConductor

        self.conductor = MetaConductor()

    def test_get_existing(self):
        session = self.conductor.start_session("Existing session")
        retrieved = self.conductor.get_session(session.session_id)
        self.assertIs(retrieved, session)

    def test_get_nonexistent(self):
        result = self.conductor.get_session("unknown-id")
        self.assertIsNone(result)

    def test_get_session_uses_brain_fallback(self):
        brain = MagicMock()
        brain.get_innovation_session.return_value = None
        from agents.meta_conductor import MetaConductor

        conductor = MetaConductor(brain=brain)

        result = conductor.get_session("missing-id")

        brain.get_innovation_session.assert_called_once_with("missing-id")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# TestListSessions
# ---------------------------------------------------------------------------


class TestListSessions(unittest.TestCase):
    """Tests for MetaConductor.list_sessions."""

    def setUp(self):
        patcher_swarm = patch("agents.meta_conductor.InnovationSwarm")
        patcher_log = patch("agents.meta_conductor.log_json")
        self.mock_swarm_cls = patcher_swarm.start()
        self.mock_log = patcher_log.start()
        self.addCleanup(patcher_swarm.stop)
        self.addCleanup(patcher_log.stop)

        from agents.meta_conductor import MetaConductor

        self.conductor = MetaConductor()

    def test_list_all(self):
        self.conductor.start_session("Problem one")
        self.conductor.start_session("Problem two")

        sessions = self.conductor.list_sessions()
        self.assertEqual(len(sessions), 2)

    def test_list_by_status_active(self):
        self.conductor.start_session("Active problem A")
        self.conductor.start_session("Active problem B")

        # Manually mark one as completed
        sessions_list = list(self.conductor.active_sessions.values())
        sessions_list[0].status = "completed"

        active = self.conductor.list_sessions(status="active")
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].status, "active")

    def test_list_by_status_completed(self):
        self.conductor.start_session("Will be completed")
        self.conductor.start_session("Still active")

        first = list(self.conductor.active_sessions.values())[0]
        first.status = "completed"

        completed = self.conductor.list_sessions(status="completed")
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].status, "completed")

    def test_list_empty_when_no_sessions(self):
        sessions = self.conductor.list_sessions()
        self.assertEqual(sessions, [])

    def test_list_with_no_filter_returns_all_statuses(self):
        self.conductor.start_session("Active")
        self.conductor.start_session("Will be completed")
        list(self.conductor.active_sessions.values())[1].status = "completed"

        all_sessions = self.conductor.list_sessions()
        statuses = {s.status for s in all_sessions}
        self.assertIn("active", statuses)
        self.assertIn("completed", statuses)

    def test_list_from_brain_hydrates_output_scores(self):
        brain = MagicMock()
        brain.list_innovation_sessions.return_value = [
            {
                "session_id": "persist01",
                "problem_statement": "Persisted problem",
                "status": "completed",
                "current_phase": InnovationPhase.TRANSFORMATION.value,
                "phases_completed": [InnovationPhase.DIVERGENCE.value, InnovationPhase.CONVERGENCE.value],
                "techniques": ["SCAMPER"],
                "constraints": {"selection_ratio": 0.2},
                "ideas_generated": 5,
                "ideas_selected": 2,
                "output_data": _make_innovation_output(session_id="persist01").model_dump(),
            }
        ]
        from agents.meta_conductor import MetaConductor

        conductor = MetaConductor(brain=brain)

        sessions = conductor.list_sessions()

        self.assertEqual(len(sessions), 1)
        self.assertIsNotNone(sessions[0].output)
        self.assertEqual(sessions[0].output.novelty_score, 0.75)
        self.assertEqual(sessions[0].constraints["selection_ratio"], 0.2)


# ---------------------------------------------------------------------------
# TestRun
# ---------------------------------------------------------------------------


class TestRun(unittest.TestCase):
    """Tests for MetaConductor.run (standard agent interface)."""

    def setUp(self):
        patcher_swarm = patch("agents.meta_conductor.InnovationSwarm")
        patcher_log = patch("agents.meta_conductor.log_json")
        self.mock_swarm_cls = patcher_swarm.start()
        self.mock_log = patcher_log.start()
        self.addCleanup(patcher_swarm.stop)
        self.addCleanup(patcher_log.stop)

        from agents.meta_conductor import MetaConductor

        self.conductor = MetaConductor()

    def test_run_starts_new_session(self):
        input_data = {"problem": "Improve deployment pipeline"}
        self.conductor.run(input_data)
        self.assertEqual(len(self.conductor.active_sessions), 1)

    def test_run_returns_dict(self):
        result = self.conductor.run({"problem": "Some interesting problem"})
        self.assertIsInstance(result, dict)

    def test_run_uses_goal_key_as_fallback(self):
        result = self.conductor.run({"goal": "Use goal key instead of problem"})
        self.assertIsInstance(result, dict)
        self.assertEqual(len(self.conductor.active_sessions), 1)

    def test_run_resumes_session(self):
        # Start a session first
        session = self.conductor.start_session("Existing problem")
        initial_count = len(self.conductor.active_sessions)

        # Run with the existing session_id
        result = self.conductor.run(
            {
                "problem": "Existing problem",
                "session_id": session.session_id,
            }
        )

        self.assertIsInstance(result, dict)
        # No new session should have been created
        self.assertEqual(len(self.conductor.active_sessions), initial_count)

    def test_run_with_unknown_session_id_creates_new(self):
        result = self.conductor.run(
            {
                "problem": "New problem",
                "session_id": "unknown-session-id",
            }
        )
        self.assertIsInstance(result, dict)
        # A new session should be created since the session_id isn't found
        self.assertEqual(len(self.conductor.active_sessions), 1)

    def test_run_result_has_session_id(self):
        result = self.conductor.run({"problem": "Check session id in output"})
        self.assertIn("session_id", result)

    def test_run_result_has_problem_statement(self):
        problem = "Check problem statement in output"
        result = self.conductor.run({"problem": problem})
        self.assertEqual(result.get("problem_statement"), problem)


# ---------------------------------------------------------------------------
# TestCapabilitiesAndDescription
# ---------------------------------------------------------------------------


class TestCapabilitiesAndDescription(unittest.TestCase):
    """Tests for MetaConductor class-level attributes."""

    def test_capabilities_is_list(self):
        from agents.meta_conductor import MetaConductor

        self.assertIsInstance(MetaConductor.capabilities, list)
        self.assertGreater(len(MetaConductor.capabilities), 0)

    def test_description_is_non_empty_string(self):
        from agents.meta_conductor import MetaConductor

        self.assertIsInstance(MetaConductor.description, str)
        self.assertTrue(len(MetaConductor.description) > 0)


if __name__ == "__main__":
    unittest.main()
