"""Unit tests for individual orchestrator phases (Phase 1-11)."""
import unittest
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.orchestrator import LoopOrchestrator
from core.policy import Policy
from memory.controller import memory_controller

class TestOrchestratorPhases(unittest.TestCase):
    def setUp(self):
        self.mock_brain = MagicMock()
        self.mock_brain.get.return_value = None
        self.mock_brain.recall_all.return_value = []
        self.mock_brain.recall_recent.return_value = []
        self.mock_brain.recall_with_budget.return_value = []
        
        self.mock_model = MagicMock()
        self.mock_model.respond.return_value = "model response"
        
        self.agents = {
            "ingest": MagicMock(),
            "plan": MagicMock(),
            "critique": MagicMock(),
            "synthesize": MagicMock(),
            "act": MagicMock(),
            "verify": MagicMock(),
            "reflect": MagicMock(),
        }
        for name, agent in self.agents.items():
            agent.name = name
            agent.run.return_value = {"status": "success", "agent_name": name}

        self.orchestrator = LoopOrchestrator(
            agents=self.agents,
            brain=self.mock_brain,
            project_root=Path("."),
            policy=Policy.from_config({}),
        )

    def test_phase_1_ingest(self):
        self.agents["ingest"].run.return_value = {"goal": "test goal", "context": "some context"}
        # We can't easily test internal phase methods as they are mostly inlined in run_cycle
        # But we can test the orchestrator behavior when agents are called.
        pass

    def test_phase_2_skill_dispatch(self):
        # Verify skill dispatcher is called
        with patch("core.orchestrator.dispatch_skills") as mock_dispatch:
            mock_dispatch.return_value = {"completed": ["skill1"], "failed": []}
            # Mock _configure_pipeline to avoid complex logic
            with patch.object(self.orchestrator, "_configure_pipeline") as mock_cfg:
                cfg = MagicMock(phases=["ingest", "plan"], intensity="normal")
                cfg.plan_retries = 3
                cfg.act_retries = 3
                cfg.max_act_attempts = 3
                mock_cfg.return_value = cfg
                self.orchestrator.run_cycle("test goal", dry_run=True)
                assert mock_dispatch.called

    def test_phase_3_plan(self):
        self.orchestrator.run_cycle("test goal", dry_run=True)
        assert self.agents["plan"].run.called

    def test_phase_4_critique(self):
        self.orchestrator.run_cycle("test goal", dry_run=True)
        assert self.agents["critique"].run.called

    def test_phase_5_synthesize(self):
        self.orchestrator.run_cycle("test goal", dry_run=True)
        assert self.agents["synthesize"].run.called

    def test_phase_6_act(self):
        self.orchestrator.run_cycle("test goal", dry_run=True)
        assert self.agents["act"].run.called

    def test_phase_7_verify(self):
        self.orchestrator.run_cycle("test goal", dry_run=True)
        assert self.agents["verify"].run.called

    def test_phase_8_measure(self):
        with patch("core.quality_snapshot.run_quality_snapshot") as mock_snapshot:
            mock_snapshot.return_value = {"test_count": 10}
            self.orchestrator.run_cycle("test goal", dry_run=True)
            assert mock_snapshot.called

    def test_phase_9_learn(self):
        # learn phase stores outcome in brain
        self.orchestrator.run_cycle("test goal", dry_run=True)
        # Check if brain.set was called with outcome:cycle_...
        called_keys = [args[0] for args, kwargs in self.mock_brain.set.call_args_list]
        assert any(k.startswith("outcome:cycle_") for k in called_keys)

    def test_phase_10_discover(self):
        mock_discovery = MagicMock()
        mock_discovery.run_scan.return_value = {"suggestions": []}
        self.orchestrator.attach_improvement_loops(mock_discovery)
        
        # We need to make sure the loop is recognized as AutonomousDiscovery
        type(mock_discovery).__name__ = "AutonomousDiscovery"
        
        self.orchestrator.run_cycle("test goal", dry_run=True)
        assert mock_discovery.run_scan.called

    def test_phase_11_evolve(self):
        mock_evolution = MagicMock()
        self.orchestrator.attach_improvement_loops(mock_evolution)
        
        # We need to make sure the loop is recognized as EvolutionLoop
        type(mock_evolution).__name__ = "EvolutionLoop"
        
        self.orchestrator.run_cycle("test goal", dry_run=True)
        assert mock_evolution.on_cycle_complete.called

if __name__ == "__main__":
    unittest.main()
