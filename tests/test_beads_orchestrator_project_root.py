from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

from core.orchestrator import BeadsSyncLoop, LoopOrchestrator
from core.policy import Policy


def _make_agents() -> dict[str, MagicMock]:
    agents: dict[str, MagicMock] = {}
    for name in ("ingest", "plan", "critique", "synthesize", "act", "verify", "reflect"):
        agent = MagicMock()
        agent.name = name
        agent.run.return_value = {"status": "success", "agent_name": name}
        agents[name] = agent
    return agents


def test_orchestrator_bead_side_effects_include_project_root(tmp_path: Path):
    beads_skill = MagicMock()
    orchestrator = LoopOrchestrator(
        agents=_make_agents(),
        project_root=tmp_path,
        policy=Policy.from_config({}),
        beads_enabled=True,
    )
    orchestrator.skills = {"beads_skill": beads_skill}

    orchestrator._claim_bead("bd-1")
    orchestrator._close_bead("bd-1", "done")

    assert beads_skill.run.call_args_list == [
        call({"cmd": "update", "project_root": str(tmp_path), "id": "bd-1", "args": ["--status", "in_progress"]}),
        call({"cmd": "close", "project_root": str(tmp_path), "id": "bd-1", "args": ["--reason", "done"]}),
    ]


def test_orchestrator_poll_external_goals_includes_project_root(tmp_path: Path):
    beads_skill = MagicMock()
    beads_skill.run.return_value = {
        "ready": [
            {"id": "bd-1", "title": "Fix tests"},
            {"id": "bd-2", "summary": "Refresh snapshots"},
        ]
    }
    orchestrator = LoopOrchestrator(
        agents=_make_agents(),
        project_root=tmp_path,
        policy=Policy.from_config({}),
        beads_enabled=True,
    )
    orchestrator.skills = {"beads_skill": beads_skill}

    goals = orchestrator.poll_external_goals()

    assert goals == ["bead:bd-1: Fix tests", "bead:bd-2: Refresh snapshots"]
    beads_skill.run.assert_called_once_with({"cmd": "ready", "project_root": str(tmp_path)})


def test_beads_sync_loop_includes_project_root(tmp_path: Path):
    beads_skill = MagicMock()
    sync_loop = BeadsSyncLoop(beads_skill, project_root=tmp_path)
    sync_loop._n = sync_loop.EVERY_N - 1

    sync_loop.on_cycle_complete({})

    assert beads_skill.run.call_args_list == [
        call({"cmd": "dolt", "project_root": str(tmp_path), "args": ["pull"]}),
        call({"cmd": "dolt", "project_root": str(tmp_path), "args": ["push"]}),
    ]
