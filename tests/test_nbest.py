"""Tests for N-Best code generation with critic tournament."""
import unittest
from unittest.mock import MagicMock

from core.nbest import NBestEngine, CodeCandidate, SCORING_AXES


class TestCodeCandidate(unittest.TestCase):
    def test_defaults(self):
        c = CodeCandidate(variant_id=0)
        self.assertEqual(c.variant_id, 0)
        self.assertEqual(c.changes, [])
        self.assertEqual(c.total_score, 0.0)
        self.assertFalse(c.sandbox_passed)

    def test_with_scores(self):
        c = CodeCandidate(variant_id=1, total_score=0.85, sandbox_passed=True)
        self.assertEqual(c.total_score, 0.85)
        self.assertTrue(c.sandbox_passed)


class TestNBestEngine(unittest.TestCase):
    def setUp(self):
        self.engine = NBestEngine(n_candidates=3)

    def test_init_defaults(self):
        self.assertEqual(self.engine.n_candidates, 3)
        self.assertEqual(self.engine.temperature_spread, (0.2, 0.5, 0.8))

    def test_init_custom(self):
        engine = NBestEngine(n_candidates=2, temperature_spread=(0.1, 0.9))
        self.assertEqual(engine.n_candidates, 2)

    def test_init_clamps_minimum(self):
        engine = NBestEngine(n_candidates=0)
        self.assertEqual(engine.n_candidates, 1)

    def test_generate_candidates_count(self):
        model = MagicMock()
        model.respond.return_value = '{"aura_target": "test.py", "code": "print(1)"}'
        candidates = self.engine.generate_candidates(model, "Write hello world")
        self.assertEqual(len(candidates), 3)

    def test_generate_candidates_temperatures(self):
        model = MagicMock()
        model.respond.return_value = '{"aura_target": "test.py", "code": "x=1"}'
        candidates = self.engine.generate_candidates(model, "test")
        temps = [c.temperature for c in candidates]
        self.assertEqual(temps, [0.2, 0.5, 0.8])

    def test_generate_candidates_with_role_model(self):
        model = MagicMock()
        model.respond_for_role.return_value = '{"aura_target": "t.py", "code": "y=1"}'
        candidates = self.engine.generate_candidates(model, "test")
        self.assertEqual(len(candidates), 3)
        model.respond_for_role.assert_called()

    def test_generate_candidates_handles_error(self):
        model = MagicMock()
        model.respond.side_effect = Exception("API error")
        delattr(model, "respond_for_role")
        candidates = self.engine.generate_candidates(model, "test")
        self.assertEqual(len(candidates), 3)
        self.assertTrue(all("ERROR" in c.raw_response for c in candidates))

    def test_sandbox_all_marks_results(self):
        sandbox = MagicMock()
        sandbox.run.return_value = {"success": True, "output": "ok"}
        candidates = [
            CodeCandidate(variant_id=0, changes=[{"file_path": "a.py"}]),
            CodeCandidate(variant_id=1, changes=[]),  # No changes, skipped
        ]
        result = self.engine.sandbox_all(sandbox, candidates)
        self.assertTrue(result[0].sandbox_passed)
        self.assertFalse(result[1].sandbox_passed)

    def test_critic_tournament_single_candidate(self):
        model = MagicMock()
        candidates = [CodeCandidate(variant_id=0, changes=[{"file_path": "x.py"}])]
        winner = self.engine.critic_tournament(model, candidates, "test goal")
        self.assertEqual(winner.variant_id, 0)
        self.assertEqual(winner.total_score, 1.0)

    def test_critic_tournament_no_candidates(self):
        model = MagicMock()
        candidates = [CodeCandidate(variant_id=0, changes=[])]
        with self.assertRaises(ValueError):
            self.engine.critic_tournament(model, candidates, "test")

    def test_critic_tournament_fallback_scoring(self):
        model = MagicMock()
        model.respond.return_value = "invalid json"
        delattr(model, "respond_for_role")
        candidates = [
            CodeCandidate(variant_id=0, changes=[{"file_path": "a.py"}],
                          temperature=0.2, sandbox_passed=True),
            CodeCandidate(variant_id=1, changes=[{"file_path": "b.py"}],
                          temperature=0.8, sandbox_passed=False),
        ]
        winner = self.engine.critic_tournament(model, candidates, "test")
        self.assertEqual(winner.variant_id, 0)  # sandbox passed + lower temp

    def test_critic_tournament_with_valid_scores(self):
        import json
        model = MagicMock()
        scores = {
            "scores": {
                "0": {a: 0.6 for a in SCORING_AXES},
                "1": {a: 0.9 for a in SCORING_AXES},
            }
        }
        model.respond.return_value = json.dumps(scores)
        delattr(model, "respond_for_role")
        candidates = [
            CodeCandidate(variant_id=0, changes=[{"file_path": "a.py"}]),
            CodeCandidate(variant_id=1, changes=[{"file_path": "b.py"}]),
        ]
        winner = self.engine.critic_tournament(model, candidates, "test")
        self.assertEqual(winner.variant_id, 1)
        self.assertAlmostEqual(winner.total_score, 0.9)

    def test_parse_changes_json_format(self):
        response = '{"aura_target": "hello.py", "code": "print(42)"}'
        changes = self.engine._parse_changes(response)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["file_path"], "hello.py")

    def test_parse_changes_code_block(self):
        response = "Here is the code:\n```python\nprint('hi')\n```"
        changes = self.engine._parse_changes(response)
        self.assertEqual(len(changes), 1)

    def test_parse_changes_empty(self):
        changes = self.engine._parse_changes("no code here")
        self.assertEqual(changes, [])


if __name__ == "__main__":
    unittest.main()
