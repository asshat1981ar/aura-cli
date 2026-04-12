import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import io

from aura_cli.dispatch import _handle_metrics_show_dispatch, DispatchContext
from memory.store import MemoryStore


class TestMetricsReliability(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.memory_store = MemoryStore(self.root)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_metrics_distinguishes_outcomes(self):
        # Mock cycles with different outcomes
        cycles = [
            {"cycle_id": "c1", "goal": "g1", "outcome": "SUCCESS", "duration_s": 10.0, "stop_reason": "PASS"},
            {"cycle_id": "c2", "goal": "g2", "outcome": "FAILED", "duration_s": 20.0, "stop_reason": "MAX_CYCLES"},
            {"cycle_id": "c3", "goal": "g3", "outcome": "SKIPPED", "duration_s": 5.0, "stop_reason": "PASS"},
        ]

        for c in cycles:
            entry = {
                "cycle_id": c["cycle_id"],
                "goal": c["goal"],
                "cycle_summary": c,  # Structured summary
            }
            self.memory_store.append_log(entry)

        ctx = MagicMock(spec=DispatchContext)
        ctx.runtime = {"memory_store": self.memory_store, "brain": MagicMock()}
        ctx.args = MagicMock()
        ctx.args.json = True
        ctx.parsed = MagicMock()

        out = io.StringIO()
        with patch("sys.stdout", out):
            _handle_metrics_show_dispatch(ctx)

        payload = json.loads(out.getvalue())

        recent = payload["recent_cycles"]
        self.assertEqual(len(recent), 3)

        self.assertEqual(recent[0]["status"], "SUCCESS")
        self.assertEqual(recent[0]["stop_reason"], "PASS")
        self.assertEqual(recent[1]["status"], "FAILED")
        self.assertEqual(recent[1]["stop_reason"], "MAX_CYCLES")
        self.assertEqual(recent[2]["status"], "SKIPPED")
        self.assertEqual(recent[2]["stop_reason"], "PASS")

        summary = payload["summary"]
        self.assertEqual(summary["successes"], 1)
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(summary["fails"], 1)
        self.assertEqual(summary["count"], 3)
        self.assertAlmostEqual(summary["win_rate"], 33.33333333)
        self.assertAlmostEqual(summary["avg_duration"], 11.66666666)


if __name__ == "__main__":
    unittest.main()
