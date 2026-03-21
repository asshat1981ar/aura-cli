"""Tests for Karpathy-style experiment tracker."""
import json
import tempfile
import unittest
from pathlib import Path

from core.experiment_tracker import (
    ExperimentResult, MetricsCollector, ExperimentTracker,
)


class TestExperimentResult(unittest.TestCase):
    def test_net_improvement_positive(self):
        r = ExperimentResult(
            experiment_id="e1", hypothesis="test",
            change_description="change",
            improvement={"test_pass_rate": 0.1, "avg_cycle_seconds": -5.0},
        )
        expected = (0.1 + (-5.0)) / 2
        self.assertAlmostEqual(r.net_improvement, expected)

    def test_net_improvement_empty(self):
        r = ExperimentResult(
            experiment_id="e2", hypothesis="test",
            change_description="change",
        )
        self.assertEqual(r.net_improvement, 0.0)


class TestMetricsCollector(unittest.TestCase):
    def test_collect_no_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = MetricsCollector(Path(tmpdir))
            metrics = mc.collect()
            self.assertIn("test_pass_rate", metrics)
            self.assertIn("avg_cycle_seconds", metrics)
            self.assertIn("goal_completion_rate", metrics)

    def test_collect_with_decision_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "decision_log.jsonl"
            entries = [
                {"event": "verify_result", "details": {"success": True}},
                {"event": "verify_result", "details": {"success": True}},
                {"event": "verify_result", "details": {"success": False}},
            ]
            log_path.write_text("\n".join(json.dumps(e) for e in entries))
            mc = MetricsCollector(Path(tmpdir))
            metrics = mc.collect()
            self.assertAlmostEqual(metrics["test_pass_rate"], 2 / 3)

    def test_collect_with_goal_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "goal_archive.json"
            queue = Path(tmpdir) / "goal_queue.json"
            archive.write_text(json.dumps([{"g": 1}, {"g": 2}]))
            queue.write_text(json.dumps([{"g": 3}]))
            mc = MetricsCollector(Path(tmpdir))
            metrics = mc.collect()
            self.assertAlmostEqual(metrics["goal_completion_rate"], 2 / 3)


class TestExperimentTracker(unittest.TestCase):
    def test_start_and_finish(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_path = Path(tmpdir) / "experiments.jsonl"
            mc = MetricsCollector(Path(tmpdir))
            tracker = ExperimentTracker(exp_path, mc)

            baseline = tracker.start_experiment("exp1", "improve tests")
            self.assertIsInstance(baseline, dict)

            result = tracker.finish_experiment(
                "exp1", "improve tests", "added retry logic",
                baseline, cycle_number=1, duration=5.0,
            )
            self.assertIsInstance(result, ExperimentResult)
            self.assertEqual(result.experiment_id, "exp1")

    def test_persistence_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_path = Path(tmpdir) / "experiments.jsonl"
            mc = MetricsCollector(Path(tmpdir))

            tracker1 = ExperimentTracker(exp_path, mc)
            baseline = tracker1.start_experiment("exp1", "h1")
            tracker1.finish_experiment("exp1", "h1", "d1", baseline)

            # Reload
            tracker2 = ExperimentTracker(exp_path, mc)
            self.assertEqual(len(tracker2.experiments), 1)
            self.assertEqual(tracker2.experiments[0].experiment_id, "exp1")

    def test_summary_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_path = Path(tmpdir) / "experiments.jsonl"
            mc = MetricsCollector(Path(tmpdir))
            tracker = ExperimentTracker(exp_path, mc)
            summary = tracker.get_summary()
            self.assertEqual(summary["total"], 0)

    def test_summary_with_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_path = Path(tmpdir) / "experiments.jsonl"
            mc = MetricsCollector(Path(tmpdir))
            tracker = ExperimentTracker(exp_path, mc)

            for i in range(3):
                baseline = tracker.start_experiment(f"e{i}", f"h{i}")
                tracker.finish_experiment(f"e{i}", f"h{i}", f"d{i}", baseline)

            summary = tracker.get_summary()
            self.assertEqual(summary["total"], 3)
            self.assertIn("kept", summary)
            self.assertIn("discarded", summary)


if __name__ == "__main__":
    unittest.main()
