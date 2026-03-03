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
        self.assertEqual(len(proposals), 1)
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

if __name__ == "__main__":
    unittest.main()
