import unittest
from unittest.mock import MagicMock

from core.evolution_loop import EvolutionLoop
from core.rsi_integration_verification import (
    audit_architectural_delta,
    summarize_rsi_audit,
    verify_rsi_integration,
)


def _make_cycle_entry(cycle_id: str, *, hotspot: bool = False, status: str = "pass", retries: int = 0, metrics=None):
    skill_context = metrics or {}
    if hotspot:
        skill_context = dict(skill_context)
        skill_context["structural_hotspot"] = {"files": ["core/demo.py"]}
    return {
        "cycle_id": cycle_id,
        "goal": "refactor hotspot in planner" if hotspot else "improve planner stability",
        "phase_outputs": {
            "verification": {"status": status},
            "retry_count": retries,
            "skill_context": skill_context,
        },
    }


def _make_evolution_loop(improvement_service=None, *, self_dev_mode="propose"):
    planner = MagicMock()
    coder = MagicMock()
    critic = MagicMock()
    brain = MagicMock()
    brain.recall_with_budget.return_value = ["recent memory"]
    brain.recall_weaknesses.return_value = ["weakness"]
    vector = MagicMock()
    git_tools = MagicMock()
    mutator = MagicMock()
    return EvolutionLoop(
        planner,
        coder,
        critic,
        brain,
        vector,
        git_tools,
        mutator,
        improvement_service=improvement_service,
        self_dev_mode=self_dev_mode,
    )


class TestRSIIntegrationVerification(unittest.TestCase):
    def test_verify_rsi_integration_tracks_hotspot_and_scheduled_runs(self):
        loop = _make_evolution_loop(improvement_service=None, self_dev_mode="full_mutation")
        loop.run = MagicMock(return_value={"status": "ok"})

        entries = []
        for idx in range(1, 51):
            entries.append(
                _make_cycle_entry(
                    f"cycle-{idx}",
                    hotspot=(idx == 7),
                    status="fail" if idx % 10 == 0 else "pass",
                    retries=2 if idx % 10 == 0 else 0,
                )
            )

        report = verify_rsi_integration(loop, entries, target_cycles=50)

        self.assertEqual(report.processed_cycles, 50)
        self.assertEqual(report.evolution_runs, 3)
        self.assertEqual(report.hotspot_triggers, 1)
        self.assertEqual(report.scheduled_triggers, 2)
        self.assertEqual(report.failure_count, 5)
        self.assertEqual(report.average_retry_count, 0.2)
        self.assertIn("No architectural metrics were present", report.notes[0])

    def test_audit_architectural_delta_uses_first_and_last_metric_values(self):
        entries = [
            _make_cycle_entry(
                "cycle-1",
                metrics={
                    "architecture_validator": {"coupling_score": 0.9},
                    "complexity_scorer": {"high_risk_count": 22},
                    "test_coverage_analyzer": {"coverage_pct": 41.0},
                },
            ),
            _make_cycle_entry("cycle-2"),
            _make_cycle_entry(
                "cycle-3",
                metrics={
                    "architecture_validator": {"coupling_score": 0.5},
                    "complexity_scorer": {"high_risk_count": 7},
                    "test_coverage_analyzer": {"coverage_pct": 66.0},
                },
            ),
        ]

        delta = audit_architectural_delta(entries)

        self.assertEqual(delta["architecture_validator.coupling_score"]["baseline"], 0.9)
        self.assertEqual(delta["architecture_validator.coupling_score"]["current"], 0.5)
        self.assertEqual(delta["architecture_validator.coupling_score"]["delta"], -0.4)
        self.assertEqual(delta["complexity_scorer.high_risk_count"]["delta"], -15.0)
        self.assertEqual(delta["test_coverage_analyzer.coverage_pct"]["delta"], 25.0)

    def test_verify_rsi_integration_counts_logged_proposals(self):
        improvement_service = MagicMock()
        improvement_service.observe_cycle.side_effect = lambda entry: [entry]
        improvement_service.evaluate_candidates.return_value = [
            {"proposal_id": "ri-1"},
            {"proposal_id": "ri-2"},
        ]

        loop = _make_evolution_loop(improvement_service=improvement_service)
        loop.run = MagicMock(return_value={"status": "ok"})

        report = verify_rsi_integration(
            loop,
            [_make_cycle_entry("cycle-1", retries=1)],
            target_cycles=1,
        )

        self.assertEqual(report.proposal_count, 2)
        self.assertEqual(improvement_service.log_proposal.call_count, 2)
        self.assertEqual(report.evolution_runs, 0)
        self.assertTrue(any("No evolution runs were triggered" in note for note in report.notes))

    def test_summarize_rsi_audit_uses_existing_live_counts(self):
        report = summarize_rsi_audit(
            [
                _make_cycle_entry(
                    "cycle-1",
                    metrics={"architecture_validator": {"coupling_score": 0.8}},
                ),
                _make_cycle_entry(
                    "cycle-2",
                    metrics={"architecture_validator": {"coupling_score": 0.6}},
                    status="fail",
                    retries=2,
                ),
            ],
            target_cycles=2,
            evolution_runs=1,
            scheduled_triggers=1,
            hotspot_triggers=0,
            proposal_count=3,
        )

        self.assertEqual(report.processed_cycles, 2)
        self.assertEqual(report.evolution_runs, 1)
        self.assertEqual(report.scheduled_triggers, 1)
        self.assertEqual(report.proposal_count, 3)
        self.assertEqual(report.failure_count, 1)
        self.assertEqual(report.average_retry_count, 1.0)
        self.assertAlmostEqual(
            report.architectural_delta["architecture_validator.coupling_score"]["delta"],
            -0.2,
        )


if __name__ == "__main__":
    unittest.main()
