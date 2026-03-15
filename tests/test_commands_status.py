import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aura_cli.commands import _handle_status


class _Bridge:
    def to_runtime_metadata(self):
        return {
            "enabled": True,
            "required": True,
            "scope": "goal_run",
            "timeout_seconds": 20.0,
        }


def test_handle_status_json_includes_beads_runtime_and_cycle_fields():
    goal_queue = SimpleNamespace(queue=["Gate runtime"])
    goal_archive = SimpleNamespace(completed=[("Done goal", 9.0)])
    orchestrator = SimpleNamespace(
        active_cycle_summary={
            "cycle_id": "cycle-1",
            "goal": "Gate runtime",
            "goal_type": "feature",
            "state": "running",
            "current_phase": "plan",
            "phase_status": {"plan": "running"},
            "verification_status": None,
            "outcome": "RUNNING",
            "stop_reason": None,
            "failures": [],
            "retry_count": 0,
            "applied_files": [],
            "failed_files": [],
            "queued_follow_up_goals": [],
            "beads_status": "allow",
            "beads_decision_id": "beads-1",
            "beads_summary": "Proceed.",
            "beads_required_constraints": ["Keep output stable."],
            "beads_follow_up_goals": [],
            "started_at": 1.0,
            "completed_at": None,
            "duration_s": None,
        },
        last_cycle_summary=None,
        last_capability_status=None,
        beads_bridge=_Bridge(),
        beads_enabled=True,
        beads_required=True,
        beads_scope="goal_run",
        runtime_mode="full",
    )

    out = io.StringIO()
    with patch("aura_cli.commands.build_capability_status_report", return_value={"status": "ok"}), redirect_stdout(out):
        _handle_status(
            goal_queue,
            goal_archive,
            orchestrator,
            as_json=True,
            project_root=Path("."),
        )

    payload = json.loads(out.getvalue())
    assert payload["beads_runtime"]["enabled"] is True
    assert payload["beads_runtime"]["required"] is True
    assert payload["active_cycle"]["beads_status"] == "allow"
    assert payload["active_cycle"]["beads_decision_id"] == "beads-1"


def test_handle_status_text_renders_beads_sections():
    goal_queue = SimpleNamespace(queue=[])
    goal_archive = SimpleNamespace(completed=[])
    orchestrator = SimpleNamespace(
        active_cycle_summary=None,
        last_cycle_summary={
            "cycle_id": "cycle-2",
            "goal": "Summarize decisions",
            "goal_type": "feature",
            "state": "complete",
            "current_phase": "reflect",
            "phase_status": {"reflect": "pass"},
            "verification_status": "pass",
            "outcome": "SUCCESS",
            "stop_reason": "PASS",
            "failures": [],
            "retry_count": 0,
            "applied_files": [],
            "failed_files": [],
            "queued_follow_up_goals": [],
            "beads_status": "allow",
            "beads_decision_id": "beads-2",
            "beads_summary": "Proceed with the scoped patch.",
            "beads_required_constraints": [],
            "beads_follow_up_goals": [],
            "started_at": 1.0,
            "completed_at": 2.0,
            "duration_s": 1.0,
        },
        beads_bridge=_Bridge(),
        beads_enabled=True,
        beads_required=True,
        beads_scope="goal_run",
        runtime_mode="full",
    )

    out = io.StringIO()
    with redirect_stdout(out):
        _handle_status(goal_queue, goal_archive, orchestrator, project_root=Path("."))

    rendered = out.getvalue()
    assert "--- BEADS Gate ---" in rendered
    assert "Mode: required" in rendered
    assert "--- Last Cycle ---" in rendered
    assert "BEADS: allow (beads-2)" in rendered
    assert "BEADS Summary: Proceed with the scoped patch." in rendered


def test_handle_status_text_renders_cycle_diagnostics():
    goal_queue = SimpleNamespace(queue=[])
    goal_archive = SimpleNamespace(completed=[])
    orchestrator = SimpleNamespace(
        active_cycle_summary=None,
        last_cycle_summary={
            "cycle_id": "cycle-3",
            "goal": "Recover failed apply",
            "goal_type": "bug_fix",
            "state": "complete",
            "current_phase": "reflect",
            "phase_status": {"plan": "pass", "act": "fail", "verify": "fail"},
            "verification_status": "fail",
            "outcome": "FAILED",
            "stop_reason": "MAX_CYCLES",
            "failure_routing_decision": "plan",
            "failure_routing_reason": "structural: detected architecture drift",
            "failures": [
                "apply conflict in aura_cli/commands.py",
                "verification assertion failed",
                "sandbox reported formatting drift",
                "follow-up retry still blocked",
            ],
            "retry_count": 2,
            "applied_files": ["aura_cli/commands.py", "tests/test_commands_status.py"],
            "failed_files": ["core/orchestrator.py"],
            "queued_follow_up_goals": ["Review failed apply", "Inspect formatting drift"],
            "beads_status": "allow",
            "beads_decision_id": "beads-3",
            "beads_summary": "Proceed with operator-visible diagnostics.",
            "beads_required_constraints": [],
            "beads_follow_up_goals": [],
            "started_at": 1.0,
            "completed_at": 2.5,
            "duration_s": 1.5,
        },
        beads_bridge=_Bridge(),
        beads_enabled=True,
        beads_required=True,
        beads_scope="goal_run",
        runtime_mode="full",
    )

    capability_status = {
        "last_goal": None,
        "matched_capability_ids": [],
        "pending_self_development_goals": [],
        "pending_bootstrap_actions": [],
        "running_bootstrap_actions": [],
    }

    out = io.StringIO()
    with (
        patch("aura_cli.commands.build_capability_status_report", return_value=capability_status),
        patch("aura_cli.commands.TaskManager", return_value=SimpleNamespace(root_tasks=[])),
        redirect_stdout(out),
    ):
        _handle_status(goal_queue, goal_archive, orchestrator, project_root=Path("."))

    rendered = out.getvalue()
    assert "Verification: FAIL" in rendered
    assert "Retries: 2" in rendered
    assert "Failed Phases: act, verify" in rendered
    assert "Files Applied: 2" in rendered
    assert "Files Failed: 1" in rendered
    assert (
        "Failures: apply conflict in aura_cli/commands.py; verification assertion failed; "
        "sandbox reported formatting drift (+1 more)"
    ) in rendered
    assert "Follow-up Goals Queued: 2" in rendered
    assert "Failure Route: plan" in rendered
    assert "Failure Route Reason: structural: detected architecture drift" in rendered
