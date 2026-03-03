import contextlib
import importlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
        self.planner._respond.return_value = json.dumps({
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
        })
        self.coder.implement.return_value = "print('done')"
        self.critic.critique_code.return_value = "looks fine"
        self.critic.validate_mutation.return_value = json.dumps({
            "decision": "APPROVED",
            "confidence_score": 0.95,
            "impact_assessment": "good",
            "reasoning": "safe",
        })

        result = self.loop.run("improve aura")

        self.critic.validate_mutation.assert_called_once()
        validation_payload = self.critic.validate_mutation.call_args.args[0]
        assert "\"mutations\"" in validation_payload
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


if __name__ == "__main__":
    unittest.main()
