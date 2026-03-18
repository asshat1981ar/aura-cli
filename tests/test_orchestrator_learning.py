import unittest
import tempfile
import shutil
import os
from unittest.mock import MagicMock, patch
from pathlib import Path
from core.orchestrator import LoopOrchestrator
from core.adaptive_pipeline import AdaptivePipeline
from memory.brain import Brain
from agents.registry import default_agents
from core.policy import Policy
from git import Repo

class TestOrchestratorLearning(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.project_root = Path(self.test_dir)
        
        # Initialize a dummy git repo
        self.repo = Repo.init(self.test_dir)
        (self.project_root / "README.md").write_text("test")
        self.repo.index.add(["README.md"])
        self.repo.index.commit("initial commit")

        self.brain = MagicMock(spec=Brain)
        self.brain.get.return_value = None
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
        shutil.rmtree(self.test_dir)

    def test_record_outcome_called_at_end_of_cycle(self):
        # Mock run_cycle parts to avoid actual LLM calls
        goal = "implement a new feature"
        goal_type = "feature"
        
        with patch.object(self.orchestrator, "_configure_pipeline") as mock_cfg, \
             patch.object(self.orchestrator, "_run_ingest_phase", return_value={}), \
             patch.object(self.orchestrator, "_dispatch_skills", return_value={}), \
             patch.object(self.orchestrator, "_run_plan_loop", return_value=({}, None)), \
             patch.object(self.orchestrator, "_run_reflection_phase", return_value={}), \
             patch("core.quality_snapshot.run_quality_snapshot", return_value={"test_count": 0}), \
             patch.object(self.adaptive_pipeline, "record_outcome") as mock_record:
            
            mock_cfg.return_value = MagicMock(phases=["ingest"])
            
            self.orchestrator.run_cycle(goal)
            
            # Verify record_outcome was called
            self.assertTrue(mock_record.called)
            args = mock_record.call_args[0]
            self.assertEqual(args[0], goal_type)

if __name__ == "__main__":
    unittest.main()
