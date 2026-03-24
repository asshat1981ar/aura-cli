"""Shared operator-facing runtime summary helpers."""

from __future__ import annotations

import time
from typing import Any, Dict

_PHASE_OUTPUT_KEYS = {
    "ingest": "context",
    "skill_dispatch": "skill_context",
    "plan": "plan",
    "critique": "critique",
    "synthesize": "task_bundle",
    "act": "change_set",
    "sandbox": "sandbox",
    "apply": "apply_result",
    "verify": "verification",
    "reflect": "reflection",
}


def _normalize_completed_entry(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return {
            "goal": item.get("goal", ""),
            "score": item.get("score"),
        }
    if isinstance(item, (list, tuple)):
        goal = item[0] if len(item) > 0 else ""
        score = item[1] if len(item) > 1 else None
        return {"goal": goal, "score": score}
    return {"goal": str(item), "score": None}


def build_queue_summary(
    goal_queue: Any = None,
    goal_archive: Any = None,
    *,
    active_goal: str | None = None,
    updated_at: float | None = None,
) -> Dict[str, Any]:
    pending_goals = []
    if goal_queue is not None:
        try:
            pending_goals = list(goal_queue.queue)
        except Exception:
            pending_goals = []

    completed_goals = []
    if goal_archive is not None:
        try:
            completed_goals = [_normalize_completed_entry(item) for item in list(goal_archive.completed)]
        except Exception:
            completed_goals = []

    return {
        "pending_count": len(pending_goals),
        "pending": [
            {"position": idx, "goal": str(goal)}
            for idx, goal in enumerate(pending_goals, start=1)
        ],
        "completed_count": len(completed_goals),
        "completed": completed_goals,
        "active_goal": active_goal,
        "updated_at": updated_at if updated_at is not None else time.time(),
    }


def _normalize_verification_status(verification: Dict[str, Any]) -> str | None:
    if not isinstance(verification, dict):
        return None
    status = verification.get("status")
    if status in ("pass", "fail", "skip"):
        return status
    if "passed" in verification:
        return "pass" if bool(verification.get("passed")) else "fail"
    return None


def _status_for_phase(phase: str, payload: Any) -> str:
    if phase == "verify":
        return _normalize_verification_status(payload) or "pending"

    if phase == "apply" and isinstance(payload, dict):
        if payload.get("failed"):
            return "fail"
        if "applied" in payload or "snapshots" in payload:
            return "pass"

    if phase == "sandbox" and isinstance(payload, dict):
        if payload.get("status") == "skip":
            return "skip"
        if "passed" in payload:
            return "pass" if bool(payload.get("passed")) else "fail"

    if isinstance(payload, dict):
        if payload.get("status") == "skip":
            return "skip"
        if payload.get("status") == "fail":
            return "fail"
        if payload.get("status") == "pass":
            return "pass"

    return "pass"


def _infer_current_phase(phase_outputs: Dict[str, Any], state: str, current_phase: str | None) -> str | None:
    if current_phase:
        return current_phase
    if state != "complete":
        return None

    last_phase = None
    for phase, key in _PHASE_OUTPUT_KEYS.items():
        if key in phase_outputs:
            last_phase = phase
    return last_phase


def _build_phase_status(
    phase_outputs: Dict[str, Any],
    *,
    state: str,
    current_phase: str | None,
) -> Dict[str, str]:
    status_map: Dict[str, str] = {}
    for phase, key in _PHASE_OUTPUT_KEYS.items():
        if state == "running" and current_phase == phase:
            status_map[phase] = "running"
            continue
        if key not in phase_outputs:
            status_map[phase] = "pending"
            continue
        status_map[phase] = _status_for_phase(phase, phase_outputs[key])
    return status_map


def build_cycle_summary(
    entry: Dict[str, Any] | None = None,
    *,
    cycle_id: str | None = None,
    goal: str | None = None,
    goal_type: str | None = None,
    phase_outputs: Dict[str, Any] | None = None,
    stop_reason: str | None = None,
    state: str | None = None,
    current_phase: str | None = None,
    started_at: float | None = None,
    completed_at: float | None = None,
) -> Dict[str, Any]:
    entry = entry or {}
    phase_outputs = phase_outputs or entry.get("phase_outputs", {}) or {}
    cycle_id = cycle_id or entry.get("cycle_id")
    goal = goal if goal is not None else entry.get("goal")
    goal_type = goal_type if goal_type is not None else entry.get("goal_type")
    stop_reason = stop_reason if stop_reason is not None else entry.get("stop_reason")
    started_at = started_at if started_at is not None else entry.get("started_at")
    completed_at = completed_at if completed_at is not None else entry.get("completed_at")

    if state is None:
        state = "complete" if completed_at is not None else "running"

    current_phase = _infer_current_phase(phase_outputs, state, current_phase)
    verification = phase_outputs.get("verification", {}) if isinstance(phase_outputs, dict) else {}
    apply_result = phase_outputs.get("apply_result", {}) if isinstance(phase_outputs, dict) else {}
    capability_goal_queue = phase_outputs.get("capability_goal_queue", {}) if isinstance(phase_outputs, dict) else {}

    verification_status = _normalize_verification_status(verification)
    failures = list(verification.get("failures", [])) if isinstance(verification, dict) else []
    remediation_plan = verification.get("remediation_plan", {}) if isinstance(verification, dict) else {}
    applied_files = list(apply_result.get("applied", [])) if isinstance(apply_result, dict) else []
    failed_files = [
        item.get("file", str(item))
        for item in apply_result.get("failed", [])
    ] if isinstance(apply_result, dict) else []
    queued_follow_up_goals = list(capability_goal_queue.get("queued", [])) if isinstance(capability_goal_queue, dict) else []
    retry_count = int(phase_outputs.get("retry_count", 0)) if isinstance(phase_outputs, dict) else 0
    beads = entry.get("beads") or phase_outputs.get("beads_gate", {}) if isinstance(phase_outputs, dict) else {}
    if not isinstance(beads, dict):
        beads = {}

    duration_s = None
    if started_at is not None and completed_at is not None:
        duration_s = max(completed_at - started_at, 0.0)

    # Human-friendly outcome label
    outcome = "RUNNING"
    if state == "complete":
        if verification_status == "pass":
            outcome = "SUCCESS"
        elif verification_status == "skip":
            outcome = "SKIPPED"
        else:
            outcome = "FAILED"

    return {
        "cycle_id": cycle_id,
        "goal": goal,
        "goal_type": goal_type,
        "state": state,
        "current_phase": current_phase,
        "phase_status": _build_phase_status(phase_outputs, state=state, current_phase=current_phase),
        "verification_status": verification_status,
        "outcome": outcome,
        "stop_reason": stop_reason,
        "failures": failures,
        "remediation_route": remediation_plan.get("route") if isinstance(remediation_plan, dict) else None,
        "repeated_failure_detected": bool(remediation_plan.get("repeated_failure_detected")) if isinstance(remediation_plan, dict) else False,
        "remediation_next_checks": list(remediation_plan.get("next_checks", [])) if isinstance(remediation_plan, dict) else [],
        "retry_count": retry_count,
        "applied_files": applied_files,
        "failed_files": failed_files,
        "queued_follow_up_goals": queued_follow_up_goals,
        "beads_status": beads.get("status"),
        "beads_decision_id": beads.get("decision_id"),
        "beads_summary": beads.get("summary"),
        "beads_required_constraints": list(beads.get("required_constraints", [])),
        "beads_follow_up_goals": list(beads.get("follow_up_goals", [])),
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_s": duration_s,
    }


def build_beads_runtime_metadata(orchestrator: Any = None) -> Dict[str, Any] | None:
    if orchestrator is None:
        return None

    bridge = getattr(orchestrator, "beads_bridge", None)
    metadata: Dict[str, Any] = {}
    if bridge is not None and hasattr(bridge, "to_runtime_metadata"):
        try:
            maybe_metadata = bridge.to_runtime_metadata()
        except Exception:
            maybe_metadata = None
        if isinstance(maybe_metadata, dict):
            metadata.update(maybe_metadata)

    enabled = getattr(orchestrator, "beads_enabled", None)
    if isinstance(enabled, bool):
        metadata.setdefault("enabled", enabled)

    required = getattr(orchestrator, "beads_required", None)
    if isinstance(required, bool):
        metadata.setdefault("required", required)

    scope = getattr(orchestrator, "beads_scope", None)
    if isinstance(scope, str):
        metadata.setdefault("scope", scope)

    runtime_mode = getattr(orchestrator, "runtime_mode", None)
    if isinstance(runtime_mode, str):
        metadata.setdefault("runtime_mode", runtime_mode)
    override = getattr(orchestrator, "beads_runtime_override", None)
    if isinstance(override, dict):
        metadata["override"] = dict(override)
    return metadata or None


def build_operator_runtime_snapshot(
    goal_queue: Any = None,
    goal_archive: Any = None,
    *,
    active_cycle: Dict[str, Any] | None = None,
    last_cycle: Dict[str, Any] | None = None,
    active_goal: str | None = None,
    updated_at: float | None = None,
    beads_runtime: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if active_goal is None and isinstance(active_cycle, dict):
        active_goal = active_cycle.get("goal")
    return {
        "schema_version": 1,
        "queue": build_queue_summary(
            goal_queue,
            goal_archive,
            active_goal=active_goal,
            updated_at=updated_at,
        ),
        "active_cycle": active_cycle,
        "last_cycle": last_cycle,
        "beads_runtime": beads_runtime,
    }
