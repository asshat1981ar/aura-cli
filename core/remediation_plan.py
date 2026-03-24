"""Structured remediation planning for failed verification and sandbox steps."""

from __future__ import annotations

from typing import Any, Dict, List


def _coerce_messages(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        items: List[str] = []
        for item in value:
            items.extend(_coerce_messages(item))
        return items
    return [str(value)]


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def build_remediation_plan(
    verification: Dict[str, Any] | None,
    *,
    route: str,
    analysis_suggestion: str | None = None,
    root_cause_analysis: Dict[str, Any] | None = None,
    investigation: Dict[str, Any] | None = None,
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a stable remediation contract layered on top of failure evidence."""

    verification = verification if isinstance(verification, dict) else {}
    root_cause_analysis = root_cause_analysis if isinstance(root_cause_analysis, dict) else {}
    investigation = investigation if isinstance(investigation, dict) else {}
    context = context if isinstance(context, dict) else {}

    fix_hints = _coerce_messages(verification.get("failures"))
    if analysis_suggestion:
        fix_hints.append(analysis_suggestion)
    fix_hints.extend(_coerce_messages(root_cause_analysis.get("recommended_actions"))[:3])

    repeated = bool(investigation.get("repeated_failure_detected"))
    signals = _coerce_messages(investigation.get("signals"))
    if repeated:
        fix_hints.append("The same failure pattern is repeating; reduce scope or re-plan before retrying.")

    next_checks: List[str] = []
    if any(signal in signals for signal in ("syntax_error", "name_error", "import_error")):
        next_checks.append("Inspect the generated file and imports before rerunning verification.")
    if "assertion_failure" in signals:
        next_checks.append("Compare the failing assertion against current behavior and update code or tests deliberately.")
    if any(signal in signals for signal in ("permission", "network", "environment")):
        next_checks.append("Validate environment, permissions, and external dependencies before another code retry.")
    if not next_checks:
        next_checks.append("Inspect the full verification log and narrow the failing surface before retrying.")

    route_label = {
        "act": "retry",
        "plan": "replan",
        "skip": "skip",
    }.get(route, route)

    if route_label == "replan":
        summary = "Verification indicates a structural issue; feed the failure context back into planning."
    elif route_label == "skip":
        summary = "Failure looks external or environmental; avoid repeated code retries until the dependency is addressed."
    else:
        summary = "Failure looks recoverable in the act phase; retry with targeted fix hints."

    return {
        "route": route_label,
        "summary": summary,
        "fix_hints": _dedupe(fix_hints)[:6],
        "next_checks": _dedupe(next_checks),
        "repeated_failure_detected": repeated,
        "signals": signals,
        "operator_action": "review_environment" if route_label == "skip" else "retry_with_hints",
        "context": {
            "goal": context.get("goal"),
            "phase": context.get("phase"),
            "route": context.get("route", route),
        },
    }
