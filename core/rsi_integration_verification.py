"""Verification helpers for the Recursive Self-Improvement (RSI) lifecycle.

The original RSI track still has one open item: run AURA in an autonomous
evolve mode long enough to verify that evolution triggers fire, proposals are
observed, and architectural signals trend in the expected direction.

This module provides a deterministic harness for that verification step.  It
does not start runtime services or mutate the repository at import time.
Instead, callers provide recorded or synthetic cycle entries plus an
``EvolutionLoop`` instance, and the harness summarizes the resulting RSI
behavior.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Sequence


ARCHITECTURAL_METRICS: tuple[tuple[str, str], ...] = (
    ("architecture_validator", "coupling_score"),
    ("complexity_scorer", "high_risk_count"),
    ("test_coverage_analyzer", "coverage_pct"),
)


@dataclass
class RSIVerificationReport:
    """Summary produced by :func:`verify_rsi_integration`."""

    target_cycles: int
    processed_cycles: int
    evolution_runs: int
    scheduled_triggers: int
    hotspot_triggers: int
    failure_count: int
    average_retry_count: float
    proposal_count: int
    architectural_delta: Dict[str, Dict[str, float]]
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _phase_outputs(entry: Dict[str, Any]) -> Dict[str, Any]:
    phase_outputs = entry.get("phase_outputs", {}) if isinstance(entry, dict) else {}
    return phase_outputs if isinstance(phase_outputs, dict) else {}


def _skill_context(entry: Dict[str, Any]) -> Dict[str, Any]:
    skill_context = _phase_outputs(entry).get("skill_context", {})
    return skill_context if isinstance(skill_context, dict) else {}


def _verification_status(entry: Dict[str, Any]) -> str:
    verification = _phase_outputs(entry).get("verification", {})
    if isinstance(verification, dict):
        status = verification.get("status")
        if status:
            return str(status)
    status = entry.get("verification_status")
    return str(status or "unknown")


def _retry_count(entry: Dict[str, Any]) -> int:
    retries = entry.get("retries")
    if retries is None:
        retries = _phase_outputs(entry).get("retry_count", 0)
    try:
        return int(retries or 0)
    except (TypeError, ValueError):
        return 0


def _has_hotspot_signal(entry: Dict[str, Any]) -> bool:
    goal = str(entry.get("goal", "")).lower()
    context_blob = json.dumps(_skill_context(entry), sort_keys=True).lower()
    return "refactor hotspot" in goal or "structural_hotspot" in context_blob


def _extract_metric(entry: Dict[str, Any], skill_name: str, field_name: str) -> float | None:
    skill_payload = _skill_context(entry).get(skill_name, {})
    if not isinstance(skill_payload, dict):
        return None

    value = skill_payload.get(field_name)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def audit_architectural_delta(cycle_entries: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Summarize first-to-last drift for known RSI architectural metrics."""
    entries = [entry for entry in cycle_entries if isinstance(entry, dict)]
    deltas: Dict[str, Dict[str, float]] = {}

    for skill_name, field_name in ARCHITECTURAL_METRICS:
        series = [
            value
            for value in (
                _extract_metric(entry, skill_name, field_name)
                for entry in entries
            )
            if value is not None
        ]
        if not series:
            continue

        baseline = float(series[0])
        current = float(series[-1])
        deltas[f"{skill_name}.{field_name}"] = {
            "baseline": baseline,
            "current": current,
            "delta": current - baseline,
        }

    return deltas


def _call_count(obj: Any) -> int:
    value = getattr(obj, "call_count", None)
    return int(value) if isinstance(value, int) else 0


def summarize_rsi_audit(
    cycle_entries: Sequence[Dict[str, Any]],
    *,
    target_cycles: int,
    evolution_runs: int,
    scheduled_triggers: int,
    hotspot_triggers: int,
    proposal_count: int = 0,
    notes: Sequence[str] | None = None,
) -> RSIVerificationReport:
    """Build an RSI verification report from an already executed audit run."""
    entries = [entry for entry in cycle_entries if isinstance(entry, dict)][:target_cycles]
    report_notes = list(notes or [])

    if len(entries) < target_cycles:
        report_notes.append(
            f"Only {len(entries)} cycle entries were available for a target of {target_cycles}."
        )

    failure_count = sum(1 for entry in entries if _verification_status(entry) == "fail")
    average_retry_count = (
        round(sum(_retry_count(entry) for entry in entries) / len(entries), 2)
        if entries
        else 0.0
    )
    architectural_delta = audit_architectural_delta(entries)

    if not architectural_delta:
        report_notes.append(
            "No architectural metrics were present in cycle history; delta audit was limited to trigger and failure statistics."
        )
    if evolution_runs == 0:
        report_notes.append("No evolution runs were triggered during the verification window.")

    return RSIVerificationReport(
        target_cycles=target_cycles,
        processed_cycles=len(entries),
        evolution_runs=evolution_runs,
        scheduled_triggers=scheduled_triggers,
        hotspot_triggers=hotspot_triggers,
        failure_count=failure_count,
        average_retry_count=average_retry_count,
        proposal_count=proposal_count,
        architectural_delta=architectural_delta,
        notes=report_notes,
    )


def verify_rsi_integration(
    loop: Any,
    cycle_entries: Sequence[Dict[str, Any]],
    *,
    target_cycles: int = 50,
) -> RSIVerificationReport:
    """Drive ``EvolutionLoop.on_cycle_complete`` across cycle history entries.

    Args:
        loop: Typically an instance of :class:`core.evolution_loop.EvolutionLoop`.
        cycle_entries: Recorded or synthetic cycle entries.
        target_cycles: Maximum number of entries to process for the verification
            audit.
    """
    if target_cycles <= 0:
        raise ValueError("target_cycles must be > 0")
    if not callable(getattr(loop, "on_cycle_complete", None)):
        raise TypeError("loop must provide an on_cycle_complete(entry) method")

    entries = [entry for entry in cycle_entries if isinstance(entry, dict)][:target_cycles]
    trigger_every = int(getattr(loop, "TRIGGER_EVERY_N", 0) or 0)
    notes: List[str] = []

    if len(entries) < target_cycles:
        notes.append(
            f"Only {len(entries)} cycle entries were available for a target of {target_cycles}."
        )

    improvement_service = getattr(loop, "improvement_service", None)
    proposal_logger = getattr(improvement_service, "log_proposal", None)
    proposal_count_before = _call_count(proposal_logger)

    original_run = getattr(loop, "run", None)
    evolution_runs = 0
    scheduled_triggers = 0
    hotspot_triggers = 0
    expected_cause: str | None = None

    def _counting_run(goal: str):
        nonlocal evolution_runs, scheduled_triggers, hotspot_triggers
        evolution_runs += 1
        if expected_cause == "hotspot":
            hotspot_triggers += 1
        elif expected_cause == "scheduled":
            scheduled_triggers += 1
        if callable(original_run):
            return original_run(goal)
        return None

    if hasattr(loop, "run"):
        loop.run = _counting_run

    try:
        for entry in entries:
            next_cycle_index = int(getattr(loop, "_cycle_count", 0) or 0) + 1
            if _has_hotspot_signal(entry):
                expected_cause = "hotspot"
            elif trigger_every > 0 and next_cycle_index % trigger_every == 0:
                expected_cause = "scheduled"
            else:
                expected_cause = None

            runs_before = evolution_runs
            loop.on_cycle_complete(entry)
            if expected_cause and evolution_runs == runs_before:
                notes.append(
                    f"Expected a {expected_cause} evolution trigger at cycle {next_cycle_index}, but no run occurred."
                )
            expected_cause = None
    finally:
        if hasattr(loop, "run"):
            loop.run = original_run

    proposal_count_after = _call_count(proposal_logger)
    proposal_count = max(0, proposal_count_after - proposal_count_before)

    return summarize_rsi_audit(
        entries,
        target_cycles=target_cycles,
        evolution_runs=evolution_runs,
        scheduled_triggers=scheduled_triggers,
        hotspot_triggers=hotspot_triggers,
        proposal_count=proposal_count,
        notes=notes,
    )
