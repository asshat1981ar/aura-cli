import contextlib
import importlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.evolution_loop import EvolutionLoop, validate_mutation_plan


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

    def test_on_cycle_complete_skips_dry_run_entries(self):
        entry = {
            "cycle_id": "cycle-dry",
            "goal": "fix parser",
            "dry_run": True,
            "phase_outputs": {
                "verification": {"status": "pass"},
                "retry_count": 0,
                "skill_context": {"structural_hotspot": {"files": ["core/demo.py"]}},
            },
        }
        self.loop.run = MagicMock()

        self.loop.on_cycle_complete(entry)

        self.improvement_service.observe_cycle.assert_not_called()
        self.improvement_service.log_proposal.assert_not_called()
        self.loop.run.assert_not_called()
        self.assertEqual(self.loop._cycle_count, 0)

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


class TestValidateMutationPlan(unittest.TestCase):
    def test_valid_plan_passes(self):
        plan = {"mutations": [{"file_path": "core/foo.py", "new_content": "x = 1"}]}
        errors = validate_mutation_plan(plan, Path("."))
        assert errors == []

    def test_path_traversal_rejected(self):
        plan = {"mutations": [{"file_path": "../../etc/passwd", "new_content": "bad"}]}
        errors = validate_mutation_plan(plan, Path("."))
        assert any("path traversal" in e for e in errors)

    def test_too_many_mutations_rejected(self):
        mutations = [{"file_path": f"f{i}.py", "new_content": "x"} for i in range(25)]
        plan = {"mutations": mutations}
        errors = validate_mutation_plan(plan, Path("."))
        assert any("Too many" in e for e in errors)

    def test_oversized_content_rejected(self):
        plan = {"mutations": [{"file_path": "big.py", "new_content": "x" * 200_000}]}
        errors = validate_mutation_plan(plan, Path("."))
        assert any("too large" in e for e in errors)

    def test_missing_mutations_key(self):
        errors = validate_mutation_plan({"mutations": "not a list"}, Path("."))
        assert any("must be a list" in e for e in errors)


if __name__ == "__main__":
    unittest.main()
