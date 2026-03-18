import unittest

from core.recursive_improvement import RecursiveImprovementService

class TestRecursiveImprovementService(unittest.TestCase):
    def setUp(self):
        self.service = RecursiveImprovementService()

    def test_evaluate_candidates_empty(self):
        proposals = self.service.evaluate_candidates([])
        self.assertEqual(proposals, [])

    def test_evaluate_candidates_high_failure(self):
        cycle_history = [
            {"cycle_id": "c1", "verification_status": "fail", "retries": 3},
            {"cycle_id": "c2", "verification_status": "fail", "retries": 2},
            {"cycle_id": "c3", "verification_status": "fail", "retries": 5},
        ]
        proposals = self.service.evaluate_candidates(cycle_history)
        self.assertGreaterEqual(len(proposals), 1)
        self.assertIn("High failure rate", proposals[0]["summary"])
        self.assertEqual(len(proposals[0]["source_cycles"]), 3)

    def test_create_proposal_contract(self):
        metrics = {"success_rate": 0.5, "complexity_delta": 0.1}
        proposal = self.service.create_proposal(
            proposal_id="test_ri",
            summary="test summary",
            source_cycles=["c1"],
            metrics=metrics,
            hypotheses=["h1"],
            actions=["a1"]
        )
        self.assertEqual(proposal["proposal_id"], "test_ri")
        self.assertTrue(proposal["requires_operator_review"])
        self.assertIn("score", proposal["fitness_snapshot"])

    def test_normalize_cycle_entry_uses_real_cycle_fields(self):
        normalized = self.service.normalize_cycle_entry({
            "cycle_id": "c9",
            "goal": "stabilize parser",
            "phase_outputs": {
                "verification": {"status": "fail"},
                "retry_count": 2,
            },
        })
        self.assertEqual(normalized["cycle_id"], "c9")
        self.assertEqual(normalized["verification_status"], "fail")
        self.assertEqual(normalized["retries"], 2)

    def test_observe_cycle_builds_recent_history(self):
        history = self.service.observe_cycle({
            "cycle_id": "c1",
            "phase_outputs": {"verification": {"status": "pass"}, "retry_count": 1},
        })
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["verification_status"], "pass")
        self.assertEqual(history[0]["retries"], 1)

    def test_queue_proposals_only_enqueues_bounded_goals(self):
        class _Queue:
            def __init__(self):
                self.queue = []

            def add(self, goal):
                self.queue.append(goal)

        queue = _Queue()
        service = RecursiveImprovementService(goal_queue=queue, mode="auto_queue", max_auto_queue=1)
        safe = service.create_proposal(
            proposal_id="safe",
            summary="safe",
            source_cycles=["c1"],
            metrics={"success_rate": 0.5, "complexity_delta": 0.1},
            hypotheses=["h1"],
            actions=["a1"],
            recommended_goal="Reduce retry churn in the act/verify handoff with a targeted regression fix.",
            queueable=True,
        )
        broad = service.create_proposal(
            proposal_id="broad",
            summary="broad",
            source_cycles=["c1"],
            metrics={"success_rate": 0.5, "complexity_delta": 0.1},
            hypotheses=["h1"],
            actions=["a1"],
            recommended_goal="Rewrite the system to fix everything.",
            queueable=True,
        )

        queued = service.queue_proposals([safe, broad], mode="auto_queue")

        self.assertEqual(queued, [safe["recommended_goal"]])
        self.assertEqual(queue.queue, [safe["recommended_goal"]])
        self.assertFalse(broad["queueable"])
        self.assertEqual(broad["queue_block_reason"], "goal_too_broad")

    def test_run_manual_returns_structured_payload(self):
        service = RecursiveImprovementService()

        payload = service.run_manual(
            goal="Stabilize parser retry churn",
            cycle_history=[],
            mode="propose",
            allow_queue=False,
        )

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["goal"], "Stabilize parser retry churn")
        self.assertEqual(payload["self_dev_mode"], "propose")
        self.assertGreaterEqual(payload["proposal_count"], 1)
        self.assertTrue(payload["follow_up_goals"])

if __name__ == "__main__":
    unittest.main()
