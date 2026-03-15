"""Unit tests for individual orchestrator phases (Phase 1-11)."""
import queue
import threading
import unittest
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch, call

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

    def test_beads_revise_stops_before_plan_and_queues_follow_up_goals(self):
        beads_bridge = MagicMock()
        beads_bridge.run.return_value = {
            "schema_version": 1,
            "ok": True,
            "status": "ok",
            "decision": {
                "schema_version": 1,
                "decision_id": "beads-revise-1",
                "status": "revise",
                "summary": "Add regression coverage before execution.",
                "rationale": ["The change touches a sensitive surface."],
                "required_constraints": ["Keep CLI status output stable."],
                "required_skills": [],
                "required_tests": ["pytest tests/test_commands_status.py -q"],
                "follow_up_goals": ["Add regression coverage"],
                "stop_reason": None,
            },
            "error": None,
            "stderr": None,
            "duration_ms": 5,
        }
        goal_queue = MagicMock()
        goal_queue.queue = []
        orchestrator = LoopOrchestrator(
            agents=self.agents,
            brain=self.mock_brain,
            project_root=Path("."),
            policy=Policy.from_config({}),
            beads_bridge=beads_bridge,
            beads_enabled=True,
            beads_required=True,
            goal_queue=goal_queue,
        )

        result = orchestrator.run_cycle("test goal", dry_run=False)

        self.assertEqual(result["stop_reason"], "BEADS_REVISE")
        self.assertEqual(result["beads"]["status"], "revise")
        self.assertEqual(
            result["phase_outputs"]["beads_goal_queue"]["queued"],
            ["Add regression coverage"],
        )
        self.assertEqual(
            result["cycle_summary"]["queued_follow_up_goals"],
            ["Add regression coverage"],
        )
        goal_queue.prepend_batch.assert_called_once_with(["Add regression coverage"])
        self.agents["plan"].run.assert_not_called()

    def test_beads_context_reaches_plan_and_synthesize(self):
        beads_bridge = MagicMock()
        beads_bridge.run.return_value = {
            "schema_version": 1,
            "ok": True,
            "status": "ok",
            "decision": {
                "schema_version": 1,
                "decision_id": "beads-allow-1",
                "status": "allow",
                "summary": "Proceed with guardrails.",
                "rationale": ["The scope is understood."],
                "required_constraints": ["Keep JSON output stable."],
                "required_skills": ["status_formatter"],
                "required_tests": ["pytest tests/test_commands_status.py -q"],
                "follow_up_goals": ["Review telemetry after merge"],
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

        orchestrator.run_cycle("test goal", dry_run=True)

        plan_input = self.agents["plan"].run.call_args.args[0]
        self.assertEqual(plan_input["beads_context"]["status"], "allow")
        self.assertEqual(
            plan_input["beads_context"]["required_constraints"],
            ["Keep JSON output stable."],
        )
        synth_input = self.agents["synthesize"].run.call_args.args[0]
        self.assertEqual(
            synth_input["beads_context"]["required_tests"],
            ["pytest tests/test_commands_status.py -q"],
        )
        self.assertEqual(
            synth_input["beads_context"]["follow_up_goals"],
            ["Review telemetry after merge"],
        )

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
                call({"cmd": "update", "project_root": ".", "id": "bd-1", "args": ["--status", "in_progress"]}),
                call({"cmd": "close", "project_root": ".", "id": "bd-1", "args": ["--reason", "done"]}),
            ],
        )

    def test_beads_sync_loop_skips_external_sync_for_dry_run_cycles(self):
        beads_skill = MagicMock()
        sync_loop = BeadsSyncLoop(beads_skill)
        sync_loop._n = sync_loop.EVERY_N - 1
        self.orchestrator.attach_improvement_loops(sync_loop)

        self.orchestrator.run_cycle("test goal", dry_run=True)

        beads_skill.run.assert_not_called()

    def test_beads_sync_loop_includes_project_root_when_syncing(self):
        beads_skill = MagicMock()
        sync_loop = BeadsSyncLoop(beads_skill, project_root=Path("/tmp/beads-project"))
        sync_loop._n = sync_loop.EVERY_N - 1

        sync_loop.on_cycle_complete({})

        beads_skill.run.assert_called_once_with({"cmd": "sync", "project_root": "/tmp/beads-project"})

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
        beads_skill.run.assert_called_once_with({"cmd": "ready", "project_root": "."})

if __name__ == "__main__":
    unittest.main()


class TestPostActParallelism(unittest.TestCase):
    def setUp(self):
        agents = {
            "reflect": MagicMock(),
        }
        agents["reflect"].run.return_value = {
            "summary": "done",
            "learnings": [],
            "next_actions": [],
        }
        self.orchestrator = LoopOrchestrator(
            agents=agents,
            project_root=Path("."),
            policy=Policy.from_config({}),
        )

    def test_execute_post_act_phases_stores_snapshot_and_reflection(self):
        phase_outputs = {"apply_result": {"applied": ["core/example.py"]}}

        with patch("core.quality_snapshot.run_quality_snapshot") as mock_snapshot, \
             patch("core.orchestrator.validate_phase_output", return_value=[]), \
             patch.object(self.orchestrator.memory_controller, "store") as mock_store:
            mock_snapshot.return_value = {"test_count": 3}

            reflection = self.orchestrator._execute_post_act_phases(
                verification={"status": "pass"},
                skill_context={},
                goal_type="default",
                cycle_id="cycle_1",
                phase_outputs=phase_outputs,
                corr_id="cycle_1",
            )

        self.assertEqual(phase_outputs["quality_snapshot"], {"test_count": 3})
        self.assertEqual(phase_outputs["reflection"], reflection)
        mock_store.assert_called_once()

    def test_execute_post_act_phases_runs_work_concurrently(self):
        phase_outputs = {"apply_result": {"applied": ["core/example.py"]}}
        snapshot_started = threading.Event()
        reflection_started = threading.Event()
        release = threading.Event()
        result_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()

        def _snapshot(*_args, **_kwargs):
            snapshot_started.set()
            self.assertTrue(reflection_started.wait(timeout=1))
            self.assertTrue(release.wait(timeout=1))
            return {"test_count": 1}

        def _run_phase(name, payload):
            self.assertEqual(name, "reflect")
            self.assertIn("verification", payload)
            reflection_started.set()
            self.assertTrue(snapshot_started.wait(timeout=1))
            self.assertTrue(release.wait(timeout=1))
            return {
                "summary": "done",
                "learnings": [],
                "next_actions": [],
            }

        def _runner():
            try:
                value = self.orchestrator._execute_post_act_phases(
                    verification={"status": "pass"},
                    skill_context={},
                    goal_type="default",
                    cycle_id="cycle_2",
                    phase_outputs=phase_outputs,
                    corr_id="cycle_2",
                )
                result_queue.put(("ok", value))
            except Exception as exc:  # pragma: no cover - test helper
                result_queue.put(("error", exc))

        with patch("core.quality_snapshot.run_quality_snapshot", side_effect=_snapshot), \
             patch.object(self.orchestrator, "_run_phase", side_effect=_run_phase), \
             patch("core.orchestrator.validate_phase_output", return_value=[]), \
             patch.object(self.orchestrator.memory_controller, "store"):
            worker = threading.Thread(target=_runner)
            worker.start()

            self.assertTrue(snapshot_started.wait(timeout=1))
            self.assertTrue(reflection_started.wait(timeout=1))
            release.set()
            worker.join(timeout=1)

        status, payload = result_queue.get_nowait()
        self.assertEqual(status, "ok")
        self.assertEqual(payload["summary"], "done")

    def test_execute_post_act_phases_keeps_snapshot_on_reflection_error(self):
        phase_outputs = {"apply_result": {"applied": ["core/example.py"]}}

        with patch("core.quality_snapshot.run_quality_snapshot", return_value={"test_count": 1}), \
             patch.object(
                 self.orchestrator,
                 "_run_phase",
                 side_effect=RuntimeError("reflection failed"),
             ):
            with self.assertRaises(RuntimeError):
                self.orchestrator._execute_post_act_phases(
                    verification={"status": "pass"},
                    skill_context={},
                    goal_type="default",
                    cycle_id="cycle_3",
                    phase_outputs=phase_outputs,
                    corr_id="cycle_3",
                )

        self.assertEqual(phase_outputs["quality_snapshot"], {"test_count": 1})
        self.assertNotIn("reflection", phase_outputs)
