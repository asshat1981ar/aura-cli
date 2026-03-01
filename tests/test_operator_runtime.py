from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.operator_runtime import (
    build_cycle_summary,
    build_operator_runtime_snapshot,
    build_queue_summary,
)
from core.orchestrator import LoopOrchestrator
from memory.store import MemoryStore


def _make_agent(result):
    agent = MagicMock()
    agent.run.return_value = result
    return agent


def _make_orchestrator(tmp_path: Path) -> LoopOrchestrator:
    agents = {
        "ingest": _make_agent(
            {
                "context": "mock context",
                "goal": "test goal",
                "snapshot": "mock snapshot",
                "memory_summary": "mock memory summary",
                "hints_summary": "mock hints summary",
                "constraints": [],
            }
        ),
        "plan": _make_agent({"steps": ["step 1"], "risks": []}),
        "critique": _make_agent({"issues": [], "fixes": []}),
        "synthesize": _make_agent({"tasks": [{"id": "t1", "title": "test", "intent": "test", "files": [], "tests": []}]}),
        "act": _make_agent({"changes": []}),
        "sandbox": _make_agent({"status": "skip", "details": {}}),
        "verify": _make_agent({"passed": True, "score": 1.0, "failures": [], "logs": ""}),
        "reflect": _make_agent({"summary": "done", "weaknesses": [], "learnings": [], "next_actions": []}),
    }
    return LoopOrchestrator(
        agents=agents,
        memory_store=MemoryStore(tmp_path / "mem"),
        project_root=tmp_path,
        strict_schema=False,
    )


def test_build_queue_summary_normalizes_goal_queue_and_archive():
    goal_queue = SimpleNamespace(queue=["Fix snapshots", "Refresh docs"])
    goal_archive = SimpleNamespace(completed=[("Stabilize auth", 8.5), {"goal": "Harden logs", "score": 7.0}])

    summary = build_queue_summary(goal_queue, goal_archive, active_goal="Fix snapshots", updated_at=123.0)

    assert summary == {
        "pending_count": 2,
        "pending": [
            {"position": 1, "goal": "Fix snapshots"},
            {"position": 2, "goal": "Refresh docs"},
        ],
        "completed_count": 2,
        "completed": [
            {"goal": "Stabilize auth", "score": 8.5},
            {"goal": "Harden logs", "score": 7.0},
        ],
        "active_goal": "Fix snapshots",
        "updated_at": 123.0,
    }


def test_build_cycle_summary_derives_operator_fields():
    entry = {
        "cycle_id": "cycle_123",
        "goal": "Fix failing tests",
        "goal_type": "bug_fix",
        "stop_reason": "MAX_CYCLES",
        "started_at": 10.0,
        "completed_at": 13.5,
        "phase_outputs": {
            "retry_count": 2,
            "context": {},
            "skill_context": {},
            "plan": {"steps": ["step 1"]},
            "critique": {"issues": []},
            "task_bundle": {"tasks": []},
            "change_set": {"changes": []},
            "sandbox": {"passed": True},
            "apply_result": {"applied": ["tests/test_example.py"], "failed": []},
            "verification": {"status": "fail", "failures": ["assertion error"], "logs": ""},
            "reflection": {"summary": "retry later"},
            "capability_goal_queue": {"queued": ["Follow-up goal"]},
        },
    }

    summary = build_cycle_summary(entry)

    assert summary["cycle_id"] == "cycle_123"
    assert summary["goal"] == "Fix failing tests"
    assert summary["goal_type"] == "bug_fix"
    assert summary["state"] == "complete"
    assert summary["current_phase"] == "reflect"
    assert summary["verification_status"] == "fail"
    assert summary["stop_reason"] == "MAX_CYCLES"
    assert summary["failures"] == ["assertion error"]
    assert summary["retry_count"] == 2
    assert summary["applied_files"] == ["tests/test_example.py"]
    assert summary["failed_files"] == []
    assert summary["queued_follow_up_goals"] == ["Follow-up goal"]
    assert summary["duration_s"] == 3.5
    assert summary["phase_status"]["verify"] == "fail"
    assert summary["phase_status"]["sandbox"] == "pass"
    assert summary["phase_status"]["reflect"] == "pass"


def test_build_operator_runtime_snapshot_composes_queue_and_cycle_summaries():
    goal_queue = SimpleNamespace(queue=["Fix failing tests"])
    goal_archive = SimpleNamespace(completed=[("Done goal", 9.0)])
    cycle_summary = {
        "cycle_id": "cycle_123",
        "goal": "Fix failing tests",
        "goal_type": "bug_fix",
        "state": "running",
        "current_phase": "plan",
        "phase_status": {},
        "verification_status": None,
        "stop_reason": None,
        "failures": [],
        "retry_count": 0,
        "applied_files": [],
        "failed_files": [],
        "queued_follow_up_goals": [],
        "started_at": 10.0,
        "completed_at": None,
        "duration_s": None,
    }

    snapshot = build_operator_runtime_snapshot(
        goal_queue,
        goal_archive,
        active_cycle=cycle_summary,
        last_cycle=None,
        active_goal="Fix failing tests",
        updated_at=55.0,
    )

    assert snapshot["schema_version"] == 1
    assert snapshot["queue"]["pending_count"] == 1
    assert snapshot["queue"]["active_goal"] == "Fix failing tests"
    assert snapshot["active_cycle"]["cycle_id"] == "cycle_123"
    assert snapshot["last_cycle"] is None


def test_orchestrator_run_cycle_and_loop_publish_cycle_summary(tmp_path: Path):
    orchestrator = _make_orchestrator(tmp_path)

    cycle = orchestrator.run_cycle("Test goal", dry_run=True)
    assert cycle["goal"] == "Test goal"
    assert cycle["cycle_summary"]["goal"] == "Test goal"
    assert cycle["cycle_summary"]["verification_status"] == "pass"
    assert cycle["cycle_summary"]["retry_count"] == 0
    assert orchestrator.last_cycle_summary["cycle_id"] == cycle["cycle_id"]
    assert orchestrator.current_goal is None
    assert orchestrator.active_cycle_summary is None

    loop_result = orchestrator.run_loop("Test goal", max_cycles=1, dry_run=True)
    assert loop_result["stop_reason"]
    assert loop_result["history"][-1]["cycle_summary"]["stop_reason"] == loop_result["stop_reason"]
    assert orchestrator.last_cycle_summary["stop_reason"] == loop_result["stop_reason"]
