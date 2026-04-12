"""Structured analysis for regressions in observed test counts."""

from __future__ import annotations

from typing import Any, Dict, List


def investigate_test_count_drop(
    previous_test_count: int,
    current_test_count: int,
    *,
    goal: str | None = None,
    verification: Dict[str, Any] | None = None,
    remediation_plan: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Summarize a test-count regression into stable investigation output."""

    previous_test_count = int(previous_test_count or 0)
    current_test_count = int(current_test_count or 0)
    delta = current_test_count - previous_test_count
    dropped_to_zero = previous_test_count > 0 and current_test_count == 0
    verification = verification if isinstance(verification, dict) else {}
    remediation_plan = remediation_plan if isinstance(remediation_plan, dict) else {}

    severity = "low"
    if delta <= -20 or dropped_to_zero:
        severity = "critical"
    elif delta <= -10:
        severity = "high"
    elif delta <= -5:
        severity = "medium"

    likely_causes: List[str] = [
        "Tests were removed, renamed, or stopped being discovered by the test runner.",
        "A test package path or naming convention changed, reducing collected tests.",
    ]
    if verification.get("status") == "fail":
        likely_causes.append("Verification is already failing; broken imports or syntax may prevent test collection.")
    if remediation_plan.get("repeated_failure_detected"):
        likely_causes.append("Repeated failures suggest the regression is systemic rather than a one-off edit.")

    recommended_actions = [
        "Inspect recent changes to test module names and discovery paths.",
        "Run the narrowest test collection command and compare collected tests against the previous baseline.",
    ]
    if dropped_to_zero:
        recommended_actions.append("Treat this as a collection outage and verify the test root and runner configuration immediately.")
    if remediation_plan.get("route") == "replan":
        recommended_actions.append("Re-plan around restoring the test surface before continuing feature work.")

    goal_prefix = f" for '{goal}'" if goal else ""
    suggested_goal = f"Investigate test count drop{goal_prefix}: {previous_test_count} -> {current_test_count}"

    summary = f"Observed a test-count regression from {previous_test_count} to {current_test_count} ({delta}). Severity: {severity}."

    return {
        "summary": summary,
        "severity": severity,
        "delta": delta,
        "dropped_to_zero": dropped_to_zero,
        "likely_causes": likely_causes,
        "recommended_actions": recommended_actions,
        "suggested_goal": suggested_goal,
    }
