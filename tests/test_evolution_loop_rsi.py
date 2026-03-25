import contextlib
import importlib
import io
import json
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.evolution_loop import EvolutionLoop


class TestEvolutionLoopRSIContracts(unittest.TestCase):
    def setUp(self):
        self.planner = MagicMock()
        self.coder = MagicMock()
        self.critic = MagicMock()
        self.brain = MagicMock()
        self.brain.recall_with_budget.return_value = ["recent memory"]
        self.brain.recall_weaknesses.return_value = ["weakness"]
        self.vector = MagicMock()
        self.vector.search.return_value = ["similar issue"]
        self.git_tools = MagicMock()
        self.mutator = MagicMock()
        self.improvement_service = MagicMock()
        self.loop = EvolutionLoop(
            self.planner,
            self.coder,
            self.critic,
            self.brain,
            self.vector,
            self.git_tools,
            self.mutator,
            improvement_service=self.improvement_service,
        )

    def test_on_cycle_complete_passes_real_cycle_entry_to_improvement_service(self):
        entry = {
            "cycle_id": "cycle-1",
            "goal": "fix parser",
            "phase_outputs": {
                "verification": {"status": "fail"},
                "retry_count": 3,
                "skill_context": {},
            },
        }
        self.improvement_service.observe_cycle.return_value = [entry]
        self.improvement_service.evaluate_candidates.return_value = [{"proposal_id": "p1"}]

        self.loop.on_cycle_complete(entry)

        self.improvement_service.observe_cycle.assert_called_once_with(entry)
        self.improvement_service.evaluate_candidates.assert_called_once_with([entry])
        self.improvement_service.log_proposal.assert_called_once()

    def test_run_translates_structured_mutation_plan_before_applying(self):
        self.planner.plan.side_effect = [
            ["Inspect architecture"],
            ["Implement changes"],
        ]
        self.planner._respond.return_value = json.dumps(
            {
                "mutations": [
                    {
                        "type": "file_change",
                        "file_path": "core/demo.py",
                        "reason": "test",
                        "new_content": "print('ok')\n",
                    }
                ],
                "model_routing_updates": {},
                "capability_updates": [],
            }
        )
        self.coder.implement.return_value = "print('done')"
        self.critic.critique_code.return_value = "looks fine"
        self.critic.validate_mutation.return_value = json.dumps(
            {
                "decision": "APPROVED",
                "confidence_score": 0.95,
                "impact_assessment": "good",
                "reasoning": "safe",
            }
        )

        result = self.loop.run("improve aura")

        self.critic.validate_mutation.assert_called_once()
        validation_payload = self.critic.validate_mutation.call_args.args[0]
        assert '"mutations"' in validation_payload
        self.mutator.apply_mutation.assert_called_once()
        applied_payload = self.mutator.apply_mutation.call_args.args[0]
        assert applied_payload.startswith("ADD_FILE core/demo.py\n")
        assert result["mutation"]["mutations"][0]["file_path"] == "core/demo.py"

    def test_rsi_integration_verification_has_no_import_side_effect(self):
        module_name = "core.rsi_integration_verification"
        importlib.invalidate_caches()
        with patch("sys.modules", dict(importlib.sys.modules)):
            importlib.sys.modules.pop(module_name, None)
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                importlib.import_module(module_name)
        self.assertEqual(captured.getvalue(), "")

    def test_run_uses_innovation_workflow_when_queue_and_orchestrator_are_available(self):
        goal_queue = MagicMock()
        goal_queue.queue = deque()
        goal_queue.prepend_batch = MagicMock(side_effect=lambda goals: goal_queue.queue.extendleft(reversed(goals)))
        orchestrator = MagicMock()
        orchestrator.run_loop.return_value = {"stop_reason": "PASS", "history": [{"cycle": 1}]}
        skills = {
            "structural_analyzer": MagicMock(
                run=MagicMock(
                    return_value={
                        "hotspots": [{"file": "core/orchestrator.py", "risk_level": "HIGH"}],
                        "circular_dependencies": [],
                        "summary": "1 hotspot",
                    }
                )
            ),
            "tech_debt_quantifier": MagicMock(
                run=MagicMock(
                    return_value={
                        "debt_score": 72.0,
                        "summary": "Debt summary",
                    }
                )
            ),
            "code_clone_detector": MagicMock(
                run=MagicMock(
                    return_value={
                        "clone_count": 1,
                        "consolidation_suggestions": ["Extract shared helper"],
                    }
                )
            ),
        }
        loop = EvolutionLoop(
            self.planner,
            self.coder,
            self.critic,
            self.brain,
            self.vector,
            self.git_tools,
            self.mutator,
            goal_queue=goal_queue,
            orchestrator=orchestrator,
            project_root=Path("."),
            skills=skills,
            auto_execute_queued=True,
            innovation_goal_limit=2,
        )

        result = loop.run("research and expand capabilities")

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["innovation_report"]["selected_proposals"])
        self.assertTrue(result["queued_goals"])
        orchestrator.run_loop.assert_called()
        self.mutator.apply_mutation.assert_not_called()

    def test_run_can_queue_without_execution_for_improvement_loop_mode(self):
        goal_queue = MagicMock()
        goal_queue.queue = deque()
        goal_queue.prepend_batch = MagicMock()
        orchestrator = MagicMock()
        skills = {
            "structural_analyzer": MagicMock(
                run=MagicMock(
                    return_value={
                        "hotspots": [],
                        "circular_dependencies": [],
                        "summary": "clean",
                    }
                )
            ),
            "tech_debt_quantifier": MagicMock(
                run=MagicMock(
                    return_value={
                        "debt_score": 90.0,
                        "summary": "low debt",
                    }
                )
            ),
            "code_clone_detector": MagicMock(
                run=MagicMock(
                    return_value={
                        "clone_count": 0,
                        "consolidation_suggestions": [],
                    }
                )
            ),
        }
        loop = EvolutionLoop(
            self.planner,
            self.coder,
            self.critic,
            self.brain,
            self.vector,
            self.git_tools,
            self.mutator,
            goal_queue=goal_queue,
            orchestrator=orchestrator,
            project_root=Path("."),
            skills=skills,
            auto_execute_queued=False,
            innovation_goal_limit=1,
        )

        result = loop.run("improve research and capability expansion", execute_queued=False, dry_run=True)

        self.assertEqual(result["meta"]["mode"], "queue_only")
        self.assertFalse(result["implementation_results"])
        orchestrator.run_loop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
