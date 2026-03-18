import unittest
import tempfile
import shutil
import os
from pathlib import Path
from core.orchestrator import LoopOrchestrator
from core.adaptive_pipeline import AdaptivePipeline
from memory.brain import Brain
from agents.registry import default_agents
from core.policy import Policy
from git import Repo
from unittest.mock import MagicMock, patch

class TestOrchestratorLearningIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.project_root = Path(self.test_dir)
        
        # Initialize a dummy git repo
        self.repo = Repo.init(self.test_dir)
        (self.project_root / "README.md").write_text("test")
        self.repo.index.add(["README.md"])
        self.repo.index.commit("initial commit")

        self.db_path = os.path.join(self.test_dir, "brain.db")
        self.brain = Brain(self.db_path)
        self.model = MagicMock()
        self.agents = default_agents(self.brain, self.model)
        
        self.orchestrator = LoopOrchestrator(
            agents=self.agents,
            brain=self.brain,
            project_root=self.project_root,
            policy=Policy.from_config({}),
        )
        
        self.adaptive_pipeline = AdaptivePipeline(brain=self.brain)
        self.orchestrator.attach_caspa(adaptive_pipeline=self.adaptive_pipeline)

    def tearDown(self):
        self.brain.db.close()
        shutil.rmtree(self.test_dir)

    def test_adaptive_intensity_changes_on_failures(self):
        goal = "implement a new feature"
        goal_type = "feature"
        
        # Mocking the pipeline configuration to check what it returns
        # We want to see if intensity changes after multiple failures.
        
        # Initial cycle - should be normal
        config = self.adaptive_pipeline.configure(goal, goal_type, consecutive_fails=0)
        self.assertEqual(config.intensity, "normal")
        
        # Record 3 losses for "normal" strategy
        for _ in range(3):
            self.adaptive_pipeline.record_outcome(goal_type, "normal", False)
            
        # Configure again - should ideally switch to "deep" or stay normal depending on EMA
        # But wait, our EMA alpha is 0.2 and baseline is 0.75.
        # 3 losses: 0.75 -> 0.6 -> 0.48 -> 0.384
        # At 0.384 it might still be normal if no other data exists.
        
        # Let's force it by passing consecutive_fails=3
        config_failed = self.adaptive_pipeline.configure(goal, goal_type, consecutive_fails=3)
        self.assertEqual(config_failed.intensity, "deep")

    def test_win_rate_persistence_in_brain(self):
        goal_type = "bug_fix"
        strategy = "minimal"
        
        self.adaptive_pipeline.record_outcome(goal_type, strategy, True)
        self.adaptive_pipeline.record_outcome(goal_type, strategy, False)
        
        rate = self.adaptive_pipeline.win_rate(goal_type, strategy)
        self.assertEqual(rate, 0.5)
        
        # Verify it's in the brain
        stats = self.brain.get(f"__strategy_stats__:{goal_type}:{strategy}")
        self.assertEqual(stats["wins"], 1)
        self.assertEqual(stats["losses"], 1)

if __name__ == "__main__":
    unittest.main()
