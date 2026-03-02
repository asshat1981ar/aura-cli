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
