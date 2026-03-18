"""Unit tests for individual orchestrator phases (Phase 1-11)."""
import unittest
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from core.phases import Phase
from core.orchestrator import BeadsSyncLoop, LoopOrchestrator
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
        self.orchestrator.run_cycle("test goal", dry_run=True)
        assert self.agents["ingest"].run.called
        call_args = self.agents["ingest"].run.call_args
        assert "goal" in call_args[0][0]
        assert call_args[0][0]["goal"] == "test goal"

    def test_phase_registry_exposes_modular_phase_wrappers(self):
        expected = {"ingest", "plan", "critique", "act", "sandbox", "apply", "verify", "reflect"}
        self.assertTrue(expected.issubset(self.orchestrator.phase_registry.keys()))
        self.assertTrue(all(isinstance(phase, Phase) for phase in self.orchestrator.phase_registry.values()))

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
        assert mock_discovery.on_cycle_complete.called

    def test_phase_11_evolve(self):
        mock_evolution = MagicMock()
        self.orchestrator.attach_improvement_loops(mock_evolution)
        
        # We need to make sure the loop is recognized as EvolutionLoop
        type(mock_evolution).__name__ = "EvolutionLoop"
        
        self.orchestrator.run_cycle("test goal", dry_run=True)
        assert mock_evolution.on_cycle_complete.called

    def test_beads_block_stops_before_plan(self):
        beads_bridge = MagicMock()
        beads_bridge.run.return_value = {
            "schema_version": 1,
            "ok": True,
            "status": "ok",
            "decision": {
                "schema_version": 1,
                "decision_id": "beads-block-1",
                "status": "block",
                "summary": "Blocked pending operator review.",
                "rationale": ["The change scope is too broad."],
                "required_constraints": [],
                "required_skills": [],
                "required_tests": [],
                "follow_up_goals": [],
                "stop_reason": None,
            },
            "error": None,
            "stderr": None,
            "duration_ms": 5,
        }
        orchestrator = LoopOrchestrator(
            agents=self.agents,
            brain=self.mock_brain,
            project_root=Path("."),
            policy=Policy.from_config({}),
            beads_bridge=beads_bridge,
            beads_enabled=True,
            beads_required=True,
        )

        result = orchestrator.run_cycle("test goal", dry_run=True)

        self.assertEqual(result["stop_reason"], "BEADS_BLOCKED")
        self.assertEqual(result["beads"]["status"], "block")
        self.assertIn("beads_gate", result["phase_outputs"])
        self.agents["plan"].run.assert_not_called()

    def test_beads_error_required_stops_before_plan(self):
        beads_bridge = MagicMock()
        beads_bridge.run.return_value = {
            "schema_version": 1,
            "ok": False,
            "status": "error",
            "decision": None,
            "error": "timeout",
            "stderr": "timed out",
            "duration_ms": 20,
        }
        orchestrator = LoopOrchestrator(
            agents=self.agents,
            brain=self.mock_brain,
            project_root=Path("."),
            policy=Policy.from_config({}),
            beads_bridge=beads_bridge,
            beads_enabled=True,
            beads_required=True,
        )

        result = orchestrator.run_cycle("test goal", dry_run=True)

        self.assertEqual(result["stop_reason"], "BEADS_UNAVAILABLE")
        self.assertEqual(result["beads"]["error"], "timeout")
        self.assertIn("beads_gate", result["phase_outputs"])
        self.agents["plan"].run.assert_not_called()

    def test_beads_gate_builds_recursive_self_improvement_development_context(self):
        beads_bridge = MagicMock()
        beads_bridge.run.return_value = {
            "schema_version": 1,
            "ok": True,
            "status": "ok",
            "decision": {
                "schema_version": 1,
                "decision_id": "beads-rsi-1",
                "status": "allow",
                "summary": "Proceed.",
                "rationale": [],
                "required_constraints": [],
                "required_skills": [],
                "required_tests": [],
                "follow_up_goals": [],
                "target_subsystem": "recursive_self_improvement",
                "canonical_path": "core/recursive_improvement.py",
                "overlap_classification": "legacy_overlap_present",
                "validation_status": "pending_live_evolution_audit",
                "required_remediation": [],
                "stop_reason": None,
            },
            "error": None,
            "stderr": None,
            "duration_ms": 5,
        }
        orchestrator = LoopOrchestrator(
            agents=self.agents,
            brain=self.mock_brain,
            project_root=Path("."),
            policy=Policy.from_config({}),
            beads_bridge=beads_bridge,
            beads_enabled=True,
            beads_required=True,
        )

        orchestrator.run_cycle("Harden recursive self-improvement routing", dry_run=True)

        payload = beads_bridge.run.call_args[0][0]
        self.assertEqual(payload["development_context"]["target_subsystem"], "recursive_self_improvement")
        self.assertEqual(payload["development_context"]["canonical_path"], "core/recursive_improvement.py")
        self.assertEqual(payload["development_context"]["overlap_classification"], "legacy_overlap_present")

    def test_bead_side_effects_are_disabled_when_beads_disabled(self):
        beads_skill = MagicMock()
        orchestrator = LoopOrchestrator(
            agents=self.agents,
            brain=self.mock_brain,
            project_root=Path("."),
            policy=Policy.from_config({}),
            beads_enabled=False,
        )
        orchestrator.skills = {"beads_skill": beads_skill}

        orchestrator._claim_bead("bd-1")
        orchestrator._close_bead("bd-1", "done")

        self.assertEqual(orchestrator.poll_external_goals(), [])
        beads_skill.run.assert_not_called()

    def test_bead_side_effects_claim_and_close_when_enabled(self):
        beads_skill = MagicMock()
        orchestrator = LoopOrchestrator(
            agents=self.agents,
            brain=self.mock_brain,
            project_root=Path("."),
            policy=Policy.from_config({}),
            beads_enabled=True,
        )
        orchestrator.skills = {"beads_skill": beads_skill}

        orchestrator._claim_bead("bd-1")
        orchestrator._close_bead("bd-1", "done")

        self.assertEqual(
            beads_skill.run.call_args_list,
            [
                call({"cmd": "update", "id": "bd-1", "args": ["--status", "in_progress"]}),
                call({"cmd": "close", "id": "bd-1", "args": ["--reason", "done"]}),
            ],
        )

    def test_beads_sync_loop_skips_external_sync_for_dry_run_cycles(self):
        beads_skill = MagicMock()
        sync_loop = BeadsSyncLoop(beads_skill)
        sync_loop._n = sync_loop.EVERY_N - 1
        self.orchestrator.attach_improvement_loops(sync_loop)

        self.orchestrator.run_cycle("test goal", dry_run=True)

        beads_skill.run.assert_not_called()

    def test_poll_external_goals_reads_ready_dict_shape(self):
        beads_skill = MagicMock()
        beads_skill.run.return_value = {
            "ready": [
                {"id": "bd-1", "title": "Fix tests"},
                {"id": "bd-2", "summary": "Refresh snapshots"},
            ]
        }
        orchestrator = LoopOrchestrator(
            agents=self.agents,
            brain=self.mock_brain,
            project_root=Path("."),
            policy=Policy.from_config({}),
            beads_enabled=True,
        )
        orchestrator.skills = {"beads_skill": beads_skill}

        goals = orchestrator.poll_external_goals()

        self.assertEqual(goals, ["bead:bd-1: Fix tests", "bead:bd-2: Refresh snapshots"])
        beads_skill.run.assert_called_once_with({"cmd": "ready"})

    def test_circuit_breaker_blocks_cycle(self):
        """When the circuit breaker is open, run_cycle returns early."""
        # Force the circuit breaker open
        for _ in range(5):
            self.orchestrator._circuit_breaker.record_failure()
        assert self.orchestrator._circuit_breaker.is_open()

        result = self.orchestrator.run_cycle("test goal", dry_run=False)

        self.assertEqual(result["stop_reason"], "CIRCUIT_BREAKER_OPEN")
        # No agents should have been called
        self.agents["ingest"].run.assert_not_called()
        self.agents["plan"].run.assert_not_called()

    def test_circuit_breaker_resets_on_success(self):
        """A successful cycle resets the circuit breaker."""
        self.orchestrator._circuit_breaker.record_failure()
        self.orchestrator._circuit_breaker.record_failure()
        self.assertEqual(self.orchestrator._circuit_breaker.consecutive_fails, 2)

        self.agents["verify"].run.return_value = {"status": "pass"}
        self.orchestrator.run_cycle("test goal", dry_run=True)

        self.assertEqual(self.orchestrator._circuit_breaker.consecutive_fails, 0)


if __name__ == "__main__":
    unittest.main()
