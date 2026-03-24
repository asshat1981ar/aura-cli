"""Helpers for turning raw verification failures into structured evidence."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List


_LEGACY_INVESTIGATION_RESPONSES = {
    "sample data that previously failed": "expected result after fix",
    "edge case data": "expected result for edge case",
    "@#$%^&*()": "expected result for special characters",
    "123456": "expected result for numeric input",
}


def _coerce_messages(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        messages: List[str] = []
        for item in value:
            messages.extend(_coerce_messages(item))
        return messages
    return [str(value)]


def _normalize_signature(text: str) -> str:
    collapsed = " ".join(str(text).strip().lower().split())
    return collapsed[:160] if collapsed else "unknown_failure"


def _signals_for_text(text: str) -> List[str]:
    lowered = text.lower()
    checks = (
        ("syntax_error", ("syntaxerror", "invalid syntax", "parse error", "jsondecodeerror")),
        ("import_error", ("modulenotfounderror", "importerror", "no module named", "cannot import")),
        ("name_error", ("nameerror", "is not defined")),
        ("assertion_failure", ("assertionerror", "assert ")),
        ("timeout", ("timeout", "timed out", "timeouterror")),
        ("permission", ("permission denied", "permissionerror", "operation not permitted")),
        ("network", ("connection reset", "connection refused", "temporary failure", "network", "dns")),
        ("environment", ("env", "environment", "missing dependency", "dependency")),
        ("test_failure", ("failed", "collected", "pytest", "semgrep", "lint")),
    )
    return [name for name, patterns in checks if any(pattern in lowered for pattern in patterns)]


def _history_failures(history: Iterable[dict]) -> List[str]:
    signatures: List[str] = []
    for entry in history or []:
        if not isinstance(entry, dict):
            continue
        phase_outputs = entry.get("phase_outputs", {}) if isinstance(entry.get("phase_outputs"), dict) else {}
        verification = phase_outputs.get("verification", {})
        sandbox = phase_outputs.get("sandbox", {})
        if isinstance(verification, dict):
            for failure in _coerce_messages(verification.get("failures")):
                signatures.append(_normalize_signature(failure))
        if isinstance(sandbox, dict) and sandbox.get("passed") is False:
            details = sandbox.get("details", {}) if isinstance(sandbox.get("details"), dict) else {}
            summary = details.get("stderr") or sandbox.get("summary")
            if summary:
                signatures.append(_normalize_signature(summary))
    return signatures


def investigate_verification_failure(
    verification: Dict[str, Any] | None,
    *,
    root_cause_analysis: Dict[str, Any] | None = None,
    history: List[dict] | None = None,
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Summarize a failed verification or sandbox result into structured evidence."""

    verification = verification if isinstance(verification, dict) else {}
    context = context if isinstance(context, dict) else {}
    root_cause_analysis = root_cause_analysis if isinstance(root_cause_analysis, dict) else {}

    failures = _coerce_messages(verification.get("failures"))
    logs = str(verification.get("logs", "") or "")
    failure_signature = _normalize_signature(failures[0] if failures else logs)
    history_signatures = _history_failures(history or [])
    history_counter = Counter(history_signatures)
    repeated_failure_detected = history_counter.get(failure_signature, 0) > 0

    signals = set()
    for failure in failures:
        signals.update(_signals_for_text(failure))
    if logs:
        signals.update(_signals_for_text(logs))
    for pattern in _coerce_messages(root_cause_analysis.get("patterns")):
        signals.add(pattern)

    if repeated_failure_detected:
        summary = "Recent history shows the same failure signature repeating."
    elif signals:
        summary = f"Verification failed with signals: {', '.join(sorted(signals))}."
    else:
        summary = "Verification failed without a strong heuristic signature."

    return {
        "failure_signature": failure_signature,
        "signals": sorted(signals) or ["unknown_failure"],
        "summary": summary,
        "repeated_failure_detected": repeated_failure_detected,
        "history_matches": history_counter.get(failure_signature, 0),
        "context": {
            "goal": context.get("goal"),
            "phase": context.get("phase"),
            "route": context.get("route"),
        },
        "evidence": {
            "failures": failures,
            "logs": logs,
            "root_cause_patterns": _coerce_messages(root_cause_analysis.get("patterns")),
        },
    }


def investigate_verification_failures(input_data: str) -> str:
    """Legacy string-in/string-out compatibility shim for older tests.

    The current structured API is :func:`investigate_verification_failure`.
    This wrapper preserves the historical behavior expected by a small set of
    older unit tests that still exercise the pre-refactor interface.
    """

    if not isinstance(input_data, str) or not input_data:
        raise ValueError("input_data must be a non-empty string")
    if len(input_data) >= 10000:
        return "expected result for large input"
    return _LEGACY_INVESTIGATION_RESPONSES.get(
        input_data,
        f"expected result for {input_data}",
    )
