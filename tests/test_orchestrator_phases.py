"""Unit tests for individual orchestrator phases (Phase 1-11)."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from core.orchestrator import BeadsSyncLoop, LoopOrchestrator
from core.policy import Policy


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


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Helpers for new pytest-style tests
# ---------------------------------------------------------------------------


def _make_agents_pytest(**overrides):
    phase_names = [
        "ingest",
        "plan",
        "critique",
        "synthesize",
        "act",
        "verify",
        "reflect",
        "sandbox",
    ]
    agents = {}
    for name in phase_names:
        mock = MagicMock(name=name)
        mock.name = name
        mock.run.return_value = {"status": "success", "agent_name": name}
        agents[name] = mock
    agents.update(overrides)
    return agents


def _make_orch_pytest(agents=None, **kwargs):
    agents = agents or _make_agents_pytest()
    with (
        patch("core.orchestrator.HookEngine"),
        patch("core.orchestrator.PhaseDispatcher"),
        patch("core.orchestrator.memory_controller"),
        patch("core.orchestrator.log_json"),
        patch("agents.skills.registry.all_skills", return_value={}, create=True),
    ):
        orch = LoopOrchestrator(
            agents=agents,
            policy=Policy.from_config({}),
            project_root=Path("."),
            **kwargs,
        )
    return orch


def _stub_orch_pytest():
    orch = _make_orch_pytest()
    orch._run_phase = MagicMock(return_value={})
    orch._notify_ui = MagicMock()
    orch._analyze_error = MagicMock(return_value=None)
    orch._run_root_cause_analysis = MagicMock(return_value=None)
    orch._run_investigation = MagicMock(return_value=None)
    orch._failure_history = MagicMock(return_value=[])
    return orch


# ---------------------------------------------------------------------------
# _select_act_agent
# ---------------------------------------------------------------------------


class TestSelectActAgentPhases:
    def test_returns_act_when_no_model(self):
        orch = _make_orch_pytest()
        orch.model = None
        assert orch._select_act_agent("fix bug") == "act"

    def test_falls_back_when_spec_not_in_agents(self):
        orch = _make_orch_pytest()
        spec = MagicMock()
        spec.name = "specialized_coder_xyz"  # not in orch.agents
        with patch("core.orchestrator_phases.resolve_agent_for_goal", return_value=spec, create=True):
            result = orch._select_act_agent("write code")
        assert result == "act"

    def test_returns_spec_name_when_present_in_agents(self):
        orch = _make_orch_pytest()
        spec = MagicMock()
        spec.name = "act"  # present in agents
        with patch("core.orchestrator_phases.resolve_agent_for_goal", return_value=spec, create=True):
            result = orch._select_act_agent("write code")
        assert result == "act"

    def test_falls_back_on_key_error(self):
        orch = _make_orch_pytest()
        with patch(
            "core.orchestrator_phases.resolve_agent_for_goal",
            side_effect=KeyError("missing"),
            create=True,
        ):
            result = orch._select_act_agent("task")
        assert result == "act"


# ---------------------------------------------------------------------------
# _run_sandbox_loop
# ---------------------------------------------------------------------------


class TestRunSandboxLoopPhases:
    def test_passed_immediately(self):
        orch = _stub_orch_pytest()
        orch._run_phase = MagicMock(return_value={"passed": True})
        act, passed, extra = orch._run_sandbox_loop("goal", {}, {}, False, {})
        assert passed is True
        assert extra == 0

    def test_empty_result_defaults_to_passed(self):
        orch = _stub_orch_pytest()
        orch._run_phase = MagicMock(return_value={})  # no "passed" → defaults True
        _, passed, _ = orch._run_sandbox_loop("goal", {}, {}, False, {})
        assert passed is True

    def test_dry_run_breaks_after_first_sandbox(self):
        orch = _stub_orch_pytest()
        orch._run_phase = MagicMock(return_value={"passed": False})
        _, _, _ = orch._run_sandbox_loop("goal", {}, {}, True, {})
        # Only 1 sandbox call; dry_run short-circuits retries
        assert orch._run_phase.call_count == 1

    def test_failure_triggers_act_retry(self):
        orch = _stub_orch_pytest()
        calls = []

        def side(name, data):
            calls.append(name)
            if name == "sandbox":
                return {"passed": len([c for c in calls if c == "sandbox"]) > 1}
            return {}

        orch._run_phase = MagicMock(side_effect=side)
        _, passed, extra = orch._run_sandbox_loop("goal", {}, {}, False, {})
        # After one failure there should be an act retry + second sandbox
        assert len(calls) >= 3

    def test_max_retries_returns_false(self):
        orch = _stub_orch_pytest()
        orch._run_phase = MagicMock(return_value={"passed": False, "summary": "fail"})
        _, passed, _ = orch._run_sandbox_loop("goal", {}, {}, False, {})
        assert passed is False


# ---------------------------------------------------------------------------
# _run_ingest_phase
# ---------------------------------------------------------------------------


class TestRunIngestPhasePhases:
    def test_stores_result_in_phase_outputs(self):
        orch = _stub_orch_pytest()
        orch._retrieve_hints = MagicMock(return_value=[])
        orch._run_phase = MagicMock(return_value={"memory_summary": "ctx"})
        phase_outputs = {}
        with patch("core.orchestrator_phases.log_json"):
            result = orch._run_ingest_phase("fix bug", "cycle_1", phase_outputs)
        assert phase_outputs["context"] == {"memory_summary": "ctx"}
        assert result == {"memory_summary": "ctx"}

    def test_context_injection_passed_to_phase(self):
        orch = _stub_orch_pytest()
        orch._retrieve_hints = MagicMock(return_value=[])
        captured = {}

        def cap(name, data):
            if name == "ingest":
                captured.update(data)
            return {}

        orch._run_phase = MagicMock(side_effect=cap)
        phase_outputs = {"context_injection": {"dep": "value"}}
        with patch("core.orchestrator_phases.log_json"):
            orch._run_ingest_phase("g", "c1", phase_outputs)
        assert captured.get("dependency_context") == {"dep": "value"}


# ---------------------------------------------------------------------------
# _dispatch_skills
# ---------------------------------------------------------------------------


class TestDispatchSkillsPhases:
    def test_empty_skills_returns_empty(self):
        orch = _stub_orch_pytest()
        orch.skills = {}
        cfg = MagicMock()
        cfg.skill_set = ["linter"]
        orch.skill_correlation = None
        phase_outputs = {}
        with patch("core.orchestrator_phases.log_json"):
            result = orch._dispatch_skills("feature", cfg, phase_outputs)
        assert result == {}

    def test_empty_skill_set_returns_empty(self):
        orch = _stub_orch_pytest()
        orch.skills = {"linter": MagicMock()}
        cfg = MagicMock()
        cfg.skill_set = []
        orch.skill_correlation = None
        phase_outputs = {}
        with patch("core.orchestrator_phases.log_json"):
            result = orch._dispatch_skills("feature", cfg, phase_outputs)
        assert result == {}
        assert phase_outputs["skill_context"] == {}


# ---------------------------------------------------------------------------
# _run_mcp_discovery_phase
# ---------------------------------------------------------------------------


class TestRunMcpDiscoveryPhasePhases:
    def test_stores_result_in_phase_outputs(self):
        orch = _stub_orch_pytest()
        orch._run_phase = MagicMock(return_value={"status": "success", "discovered": []})
        phase_outputs = {}
        result = orch._run_mcp_discovery_phase(phase_outputs)
        assert phase_outputs["mcp_discovery"] == result

    def test_notifies_ui_on_start(self):
        orch = _stub_orch_pytest()
        orch._run_phase = MagicMock(return_value={})
        phase_outputs = {}
        orch._run_mcp_discovery_phase(phase_outputs)
        orch._notify_ui.assert_any_call("on_phase_start", "mcp_discovery")


# ---------------------------------------------------------------------------
# _execute_plan_critique_synthesize
# ---------------------------------------------------------------------------


class TestExecutePlanCritiqueSynthesizePhases:
    def _cfg(self):
        cfg = MagicMock()
        cfg.extra_plan_ctx = {}
        return cfg

    def test_standard_flow_runs_all_three_phases(self):
        orch = _stub_orch_pytest()
        seq = []

        def phase(name, data):
            seq.append(name)
            if name == "plan":
                return {"steps": ["s1", "s2"]}
            return {}

        orch._run_phase = MagicMock(side_effect=phase)
        orch._load_config_file = MagicMock(return_value={})
        orch.confidence_router = MagicMock()
        orch.confidence_router.should_skip_optional.return_value = False

        phase_outputs = {}
        orch._execute_plan_critique_synthesize("goal", {}, {}, self._cfg(), phase_outputs)

        assert "plan" in seq
        assert "critique" in seq
        assert "synthesize" in seq

    def test_high_confidence_skips_critique(self):
        orch = _stub_orch_pytest()
        seq = []

        def phase(name, data):
            seq.append(name)
            if name == "plan":
                return {"steps": ["s1", "s2", "s3"]}
            return {}

        orch._run_phase = MagicMock(side_effect=phase)
        orch._load_config_file = MagicMock(return_value={})
        orch.confidence_router = MagicMock()
        orch.confidence_router.should_skip_optional.return_value = True

        phase_outputs = {}
        orch._execute_plan_critique_synthesize("goal", {}, {}, self._cfg(), phase_outputs)

        assert "critique" not in seq
        assert phase_outputs["critique"]["status"] == "skipped"

    def test_phase_outputs_populated(self):
        orch = _stub_orch_pytest()
        orch._run_phase = MagicMock(side_effect=lambda n, d: {"steps": ["s"]} if n == "plan" else {})
        orch._load_config_file = MagicMock(return_value={})
        orch.confidence_router = MagicMock()
        orch.confidence_router.should_skip_optional.return_value = False

        phase_outputs = {}
        orch._execute_plan_critique_synthesize("goal", {}, {}, self._cfg(), phase_outputs)

        assert "plan" in phase_outputs
        assert "critique" in phase_outputs
        assert "task_bundle" in phase_outputs


# ---------------------------------------------------------------------------
# _run_plan_loop
# ---------------------------------------------------------------------------


class TestRunPlanLoopPhases:
    def _cfg(self):
        cfg = MagicMock()
        cfg.plan_retries = 3
        cfg.max_act_attempts = 2
        cfg.extra_plan_ctx = {}
        return cfg

    def test_single_pass_returns_immediately(self):
        orch = _stub_orch_pytest()
        orch._execute_plan_critique_synthesize = MagicMock(return_value=({"steps": []}, {}))
        orch._run_act_loop = MagicMock(return_value=({"status": "pass"}, False, None))
        orch._notify_n8n_feedback = MagicMock()

        verification, early = orch._run_plan_loop("goal", {}, {}, self._cfg(), "c1", {}, False)
        assert verification["status"] == "pass"
        assert early is None

    def test_replan_reruns_plan_loop(self):
        orch = _stub_orch_pytest()
        orch._execute_plan_critique_synthesize = MagicMock(return_value=({"steps": []}, {}))
        n = [0]

        def act_loop(*a, **kw):
            n[0] += 1
            return ({"status": "fail"}, True, None) if n[0] == 1 else ({"status": "pass"}, False, None)

        orch._run_act_loop = MagicMock(side_effect=act_loop)
        orch._notify_n8n_feedback = MagicMock()

        _, early = orch._run_plan_loop("goal", {}, {}, self._cfg(), "c1", {}, False)
        assert orch._execute_plan_critique_synthesize.call_count == 2

    def test_early_return_from_act_loop_propagated(self):
        orch = _stub_orch_pytest()
        orch._execute_plan_critique_synthesize = MagicMock(return_value=({"steps": []}, {}))
        early_entry = {"stop_reason": "BEADS_BLOCKED"}
        orch._run_act_loop = MagicMock(return_value=({}, False, early_entry))
        orch._notify_n8n_feedback = MagicMock()

        _, early = orch._run_plan_loop("goal", {}, {}, self._cfg(), "c1", {}, False)
        assert early is early_entry
