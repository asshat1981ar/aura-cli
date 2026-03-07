from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from core.beads_bridge import BeadsBridge, build_beads_input, build_beads_runtime_input
from core.beads_contract import BEADS_SCHEMA_VERSION


def _make_bridge(tmp_path: Path) -> BeadsBridge:
    return BeadsBridge.from_defaults(
        tmp_path,
        command=("node", "scripts/beads_bridge.mjs"),
        timeout_seconds=3.0,
    )


def _script_bridge(project_root: Path, *, env: dict[str, str]) -> BeadsBridge:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "beads_bridge.mjs"
    return BeadsBridge.from_defaults(
        project_root,
        command=("node", str(script_path)),
        timeout_seconds=6.0,
        env=env,
    )


def _write_fake_bd(
    tmp_path: Path,
    *,
    info_payload: dict | None = None,
    info_exit: int = 0,
    ready_payload: list[dict] | dict | None = None,
    ready_exit: int = 0,
) -> Path:
    fake_bd = tmp_path / "fake_bd.py"
    info_payload = info_payload or {"issue_count": 0, "mode": "direct"}
    ready_payload = ready_payload if ready_payload is not None else []
    fake_bd.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import sys",
                "args = sys.argv[1:]",
                'if args and args[0] == "--no-daemon":',
                "    args = args[1:]",
                f"INFO_PAYLOAD = {info_payload!r}",
                f"READY_PAYLOAD = {ready_payload!r}",
                f"INFO_EXIT = {info_exit}",
                f"READY_EXIT = {ready_exit}",
                'if args == ["--json", "version"]:',
                '    print(json.dumps({"version": "0.57.0"}))',
                "    raise SystemExit(0)",
                'if args == ["info", "--json"]:',
                "    print(json.dumps(INFO_PAYLOAD))",
                "    raise SystemExit(INFO_EXIT)",
                'if args == ["ready", "--json"]:',
                "    print(json.dumps(READY_PAYLOAD))",
                "    raise SystemExit(READY_EXIT)",
                'print(json.dumps({"error": "unexpected command", "args": args}))',
                "raise SystemExit(1)",
            ]
        ),
        encoding="utf-8",
    )
    fake_bd.chmod(0o755)
    return fake_bd


def test_build_beads_input_sets_canonical_shape(tmp_path: Path):
    payload = build_beads_input(
        goal="Harden orchestrator gating",
        goal_type="feature",
        runtime_mode="full",
        project_root=tmp_path,
        queue_summary={"pending_count": 1},
        active_context={"active_goal": "Harden orchestrator gating"},
        prd_context={"title": "BEADS-Orchestrator"},
        conductor_track={"track_id": "beads_orchestrator_20260302"},
    )

    assert payload["schema_version"] == BEADS_SCHEMA_VERSION
    assert payload["goal"] == "Harden orchestrator gating"
    assert payload["goal_type"] == "feature"
    assert payload["runtime_mode"] == "full"
    assert payload["project_root"] == str(tmp_path)
    assert payload["queue_summary"]["pending_count"] == 1


def test_build_beads_runtime_input_loads_prd_and_track_context(tmp_path: Path):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    (plans_dir / "beads-orchestrator-prd.md").write_text("# BEADS PRD\n", encoding="utf-8")

    track_dir = tmp_path / "conductor" / "tracks" / "beads_orchestrator_20260302"
    track_dir.mkdir(parents=True)
    (track_dir / "metadata.json").write_text(
        json.dumps({"status": "active", "title": "BEADS Track"}),
        encoding="utf-8",
    )

    goal_queue = type("Queue", (), {"queue": ["Gate runtime"]})()
    goal_archive = type("Archive", (), {"completed": [("Done goal", 9.0)]})()
    payload = build_beads_runtime_input(
        goal="Gate runtime",
        goal_type="feature",
        project_root=tmp_path,
        runtime_mode="full",
        goal_queue=goal_queue,
        goal_archive=goal_archive,
        active_goal="Gate runtime",
        active_context={"skill_context": {"router": "used"}},
    )

    assert payload["goal"] == "Gate runtime"
    assert payload["prd_context"]["title"] == "BEADS PRD"
    assert payload["conductor_track"]["track_id"] == "beads_orchestrator_20260302"
    assert payload["queue_summary"]["pending_count"] == 1


def test_beads_bridge_returns_validated_result(tmp_path: Path):
    bridge = _make_bridge(tmp_path)
    payload = build_beads_input(
        goal="Test goal",
        goal_type=None,
        runtime_mode="full",
        project_root=tmp_path,
        queue_summary={},
        active_context={},
    )
    proc = subprocess.CompletedProcess(
        args=["node", "scripts/beads_bridge.mjs"],
        returncode=0,
        stdout=json.dumps(
            {
                "schema_version": 1,
                "ok": True,
                "status": "ok",
                "decision": {
                    "schema_version": 1,
                    "decision_id": "beads-1",
                    "status": "allow",
                    "summary": "Proceed.",
                    "rationale": ["Scoped goal is safe."],
                    "required_constraints": ["Keep status JSON stable."],
                    "required_skills": [],
                    "required_tests": ["tests/test_beads_bridge.py"],
                    "follow_up_goals": [],
                    "stop_reason": None,
                },
                "error": None,
                "stderr": None,
                "duration_ms": 4,
            }
        ),
        stderr="",
    )

    with patch("core.beads_bridge.subprocess.run", return_value=proc):
        result = bridge.run(payload)

    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["decision"]["status"] == "allow"
    assert result["decision"]["required_tests"] == ["tests/test_beads_bridge.py"]


def test_beads_bridge_defaults_pin_local_bd_binary(tmp_path: Path):
    local_bd = tmp_path / "node_modules" / ".bin" / "bd"
    local_bd.parent.mkdir(parents=True)
    local_bd.write_text("#!/bin/sh\n", encoding="utf-8")
    local_bd.chmod(0o755)

    bridge = BeadsBridge.from_defaults(tmp_path)

    assert bridge.config.env["BEADS_CLI"] == str(local_bd)


def test_beads_bridge_defaults_respect_explicit_bd_override(tmp_path: Path):
    local_bd = tmp_path / "node_modules" / ".bin" / "bd"
    local_bd.parent.mkdir(parents=True)
    local_bd.write_text("#!/bin/sh\n", encoding="utf-8")
    local_bd.chmod(0o755)

    bridge = BeadsBridge.from_defaults(tmp_path, env={"BEADS_CLI": "/custom/bd"})

    assert bridge.config.env["BEADS_CLI"] == "/custom/bd"


def test_beads_bridge_handles_missing_command(tmp_path: Path):
    bridge = _make_bridge(tmp_path)
    payload = build_beads_input(
        goal="Test goal",
        goal_type=None,
        runtime_mode="full",
        project_root=tmp_path,
        queue_summary={},
        active_context={},
    )

    with patch("core.beads_bridge.subprocess.run", side_effect=FileNotFoundError("node not found")):
        result = bridge.run(payload)

    assert result["ok"] is False
    assert result["error"] == "bridge_not_found"


def test_beads_bridge_handles_timeout(tmp_path: Path):
    bridge = _make_bridge(tmp_path)
    payload = build_beads_input(
        goal="Test goal",
        goal_type=None,
        runtime_mode="full",
        project_root=tmp_path,
        queue_summary={},
        active_context={},
    )

    with patch("core.beads_bridge.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["node"], timeout=3.0)):
        result = bridge.run(payload)

    assert result["ok"] is False
    assert result["error"] == "timeout"


def test_beads_bridge_handles_invalid_json(tmp_path: Path):
    bridge = _make_bridge(tmp_path)
    payload = build_beads_input(
        goal="Test goal",
        goal_type=None,
        runtime_mode="full",
        project_root=tmp_path,
        queue_summary={},
        active_context={},
    )
    proc = subprocess.CompletedProcess(
        args=["node", "scripts/beads_bridge.mjs"],
        returncode=0,
        stdout="not-json",
        stderr="",
    )

    with patch("core.beads_bridge.subprocess.run", return_value=proc):
        result = bridge.run(payload)

    assert result["ok"] is False
    assert result["error"] == "invalid_json"


def test_beads_bridge_handles_invalid_contract(tmp_path: Path):
    bridge = _make_bridge(tmp_path)
    payload = build_beads_input(
        goal="Test goal",
        goal_type=None,
        runtime_mode="full",
        project_root=tmp_path,
        queue_summary={},
        active_context={},
    )
    proc = subprocess.CompletedProcess(
        args=["node", "scripts/beads_bridge.mjs"],
        returncode=0,
        stdout=json.dumps(
            {
                "schema_version": 1,
                "ok": True,
                "status": "ok",
                "decision": "bad",
                "error": None,
                "stderr": None,
                "duration_ms": 1,
            }
        ),
        stderr="",
    )

    with patch("core.beads_bridge.subprocess.run", return_value=proc):
        result = bridge.run(payload)

    assert result["ok"] is False
    assert result["error"] == "invalid_contract"


def test_beads_bridge_preserves_structured_error_from_nonzero_process(tmp_path: Path):
    bridge = _make_bridge(tmp_path)
    payload = build_beads_input(
        goal="Test goal",
        goal_type=None,
        runtime_mode="full",
        project_root=tmp_path,
        queue_summary={},
        active_context={},
    )
    proc = subprocess.CompletedProcess(
        args=["node", "scripts/beads_bridge.mjs"],
        returncode=1,
        stdout=json.dumps(
            {
                "schema_version": 1,
                "ok": False,
                "status": "error",
                "decision": None,
                "error": "beads_database_missing",
                "stderr": "no beads database found",
                "duration_ms": 1,
            }
        ),
        stderr="no beads database found",
    )

    with patch("core.beads_bridge.subprocess.run", return_value=proc):
        result = bridge.run(payload)

    assert result["ok"] is False
    assert result["error"] == "beads_database_missing"


def test_live_bridge_script_returns_allow_and_imports_ready_followups(tmp_path: Path):
    fake_bd = _write_fake_bd(
        tmp_path,
        info_payload={"issue_count": 2, "mode": "direct"},
        ready_payload=[
            {"id": "bd-1", "title": "Review BEADS telemetry"},
            {"id": "bd-2", "summary": "Refresh operator docs"},
        ],
    )
    bridge = _script_bridge(tmp_path, env={"BEADS_CLI": str(fake_bd)})
    payload = build_beads_input(
        goal="Harden BEADS gate behavior",
        goal_type="feature",
        runtime_mode="full",
        project_root=tmp_path,
        queue_summary={"pending_count": 1},
        active_context={},
        prd_context={"title": "BEADS PRD", "path": "plans/beads-orchestrator-prd.md"},
        conductor_track={"track_id": "beads_orchestrator_20260302", "path": "conductor/tracks/beads_orchestrator_20260302"},
    )

    result = bridge.run(payload)

    assert result["ok"] is True
    assert result["decision"]["status"] == "allow"
    assert "bead:bd-1: Review BEADS telemetry" in result["decision"]["follow_up_goals"]
    assert "bead:bd-2: Refresh operator docs" in result["decision"]["follow_up_goals"]
    assert "lint" in result["decision"]["required_skills"]


def test_live_bridge_script_revises_when_capabilities_are_missing(tmp_path: Path):
    fake_bd = _write_fake_bd(
        tmp_path,
        info_payload={"issue_count": 0, "mode": "direct"},
        info_exit=1,
    )
    bridge = _script_bridge(tmp_path, env={"BEADS_CLI": str(fake_bd)})
    payload = build_beads_input(
        goal="Wire GitHub bridge into BEADS gating",
        goal_type="feature",
        runtime_mode="full",
        project_root=tmp_path,
        queue_summary={"pending_count": 2},
        active_context={
            "capability_plan": {
                "missing_skills": ["github_bridge"],
                "recommended_skills": ["git_history_analyzer"],
                "provisioning_actions": [{"action": "start_github_mcp_bridge"}],
            },
            "capability_goal_queue": {
                "queued": ["Add AURA skill 'github_bridge' so BEADS gate can inspect repository state"],
            },
        },
        prd_context={"title": "BEADS PRD", "path": "plans/beads-orchestrator-prd.md"},
        conductor_track={"track_id": "beads_orchestrator_20260302", "path": "conductor/tracks/beads_orchestrator_20260302"},
    )

    result = bridge.run(payload)

    assert result["ok"] is True
    assert result["decision"]["status"] == "revise"
    assert result["decision"]["stop_reason"] == "capability_prerequisites_missing"
    assert "github_bridge" in result["decision"]["required_skills"]
    assert any(
        "ready-work lookup completed even though runtime info was unavailable" in line
        for line in result["decision"]["rationale"]
    )
    assert any(goal.startswith("Add AURA skill 'github_bridge'") for goal in result["decision"]["follow_up_goals"])


def test_live_bridge_script_prefers_repo_pinned_bd_without_env_override(tmp_path: Path):
    local_bd = tmp_path / "node_modules" / ".bin" / "bd"
    local_bd.parent.mkdir(parents=True)
    fake_bd = _write_fake_bd(
        tmp_path,
        info_payload={"issue_count": 1, "mode": "direct"},
        ready_payload=[{"id": "bd-7", "title": "Use repo-local BD"}],
    )
    local_bd.write_text(fake_bd.read_text(encoding="utf-8"), encoding="utf-8")
    local_bd.chmod(0o755)

    bridge = _script_bridge(tmp_path, env={})
    payload = build_beads_input(
        goal="Verify repo-local bridge resolution",
        goal_type="feature",
        runtime_mode="full",
        project_root=tmp_path,
        queue_summary={"pending_count": 0},
        active_context={},
        prd_context={"title": "BEADS PRD", "path": "plans/beads-orchestrator-prd.md"},
        conductor_track={"track_id": "beads_orchestrator_20260302", "path": "conductor/tracks/beads_orchestrator_20260302"},
    )

    result = bridge.run(payload)

    assert result["ok"] is True
    assert result["decision"]["status"] == "allow"
    assert "bead:bd-7: Use repo-local BD" in result["decision"]["follow_up_goals"]
