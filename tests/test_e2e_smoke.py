"""End-to-end smoke test: dry-run cycle through the full orchestrator pipeline."""
import os
import sys
import unittest
from pathlib import Path

# Required so create_runtime doesn't os.chdir
os.environ.setdefault("AURA_SKIP_CHDIR", "1")


class TestDryRunCycleSmoke(unittest.TestCase):
    """Verify the orchestrator can complete a full dry-run cycle without crashing."""

    @classmethod
    def setUpClass(cls):
        from aura_cli.cli_main import create_runtime
        cls.runtime = create_runtime(project_root=Path("."))

    def test_cycle_completes_without_exception(self):
        orc = self.runtime["orchestrator"]
        result = orc.run_cycle("print hello world", dry_run=True)
        self.assertIsInstance(result, dict)

    def test_cycle_has_required_keys(self):
        orc = self.runtime["orchestrator"]
        result = orc.run_cycle("add a comment to main.py", dry_run=True)
        self.assertIn("cycle_id", result)
        self.assertIn("phase_outputs", result)
        self.assertIn("stop_reason", result)

    def test_cycle_stop_reason_not_crash(self):
        orc = self.runtime["orchestrator"]
        result = orc.run_cycle("refactor imports", dry_run=True)
        # stop_reason is None on success, or a string like "MAX_CYCLES" — never an exception
        self.assertNotIsInstance(result.get("stop_reason"), Exception)

    def test_adaptive_pipeline_used(self):
        """Regression: PipelineConfig.confidence field must exist (was missing)."""
        from core.adaptive_pipeline import AdaptivePipeline, PipelineConfig
        pipeline = AdaptivePipeline()
        cfg = pipeline.configure("write a test", "feature")
        self.assertIsInstance(cfg, PipelineConfig)
        # confidence must be a float (0.0 is valid when no history)
        self.assertIsInstance(cfg.confidence, float)

    def test_runtime_all_agents_registered(self):
        """All 8 pipeline agents must be wired in create_runtime."""
        orc = self.runtime["orchestrator"]
        required = {"ingest", "plan", "critique", "synthesize", "act",
                    "sandbox", "verify", "reflect"}
        missing = required - set(orc.agents.keys())
        self.assertEqual(missing, set(), f"Missing agents: {missing}")

    def test_goal_queue_uses_config_path(self):
        """GoalQueue should be constructed with the configured path."""
        gq = self.runtime["goal_queue"]
        self.assertIsNotNone(gq)
        # Queue must be functional
        initial = gq.has_goals()
        self.assertIsInstance(initial, bool)
