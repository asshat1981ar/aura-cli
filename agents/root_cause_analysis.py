"""Structured root-cause analysis for failed AURA cycles."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


class RootCauseAnalysisAgent:
    """Analyze failure evidence and produce a deterministic RCA report."""

    name = "root_cause_analysis"

    _PATTERN_RULES = (
        {
            "pattern": "import_error",
            "signals": ("modulenotfounderror", "importerror", "no module named"),
            "summary": "Missing dependency or invalid import path.",
            "root_cause": "The runtime cannot resolve a required module or symbol.",
            "actions": [
                "Verify dependency installation and Python environment consistency.",
                "Check the import path and symbol name against the current codebase.",
            ],
            "confidence": 0.92,
        },
        {
            "pattern": "syntax_error",
            "signals": ("syntaxerror", "syntax error", "indentationerror"),
            "summary": "Generated or edited code is syntactically invalid.",
            "root_cause": "A code generation or edit step produced invalid Python syntax.",
            "actions": [
                "Inspect the failing file and line number from the traceback.",
                "Regenerate or repair the affected block before retrying.",
            ],
            "confidence": 0.95,
        },
        {
            "pattern": "timeout",
            "signals": ("timeout", "timed out"),
            "summary": "The task exceeded the available execution time.",
            "root_cause": "The operation is too large, blocked, or waiting on an unavailable dependency.",
            "actions": [
                "Break the task into smaller steps or tighten the verification scope.",
                "Check for hanging subprocesses, network calls, or expensive loops.",
            ],
            "confidence": 0.88,
        },
        {
            "pattern": "assertion_failure",
            "signals": ("assertionerror", "assert "),
            "summary": "Behavioral verification failed.",
            "root_cause": "Implementation output diverges from the asserted contract or test expectation.",
            "actions": [
                "Compare the failing assertion with the current implementation contract.",
                "Inspect whether the test or the code changed behavior most recently.",
            ],
            "confidence": 0.78,
        },
        {
            "pattern": "permission",
            "signals": ("permission denied", "operation not permitted", "unauthorized", "forbidden"),
            "summary": "The task failed due to missing permissions or sandbox restrictions.",
            "root_cause": "The runtime environment cannot access a required resource.",
            "actions": [
                "Verify filesystem, network, or secret permissions for the failing operation.",
                "Move the task to a permitted environment or request explicit access.",
            ],
            "confidence": 0.9,
        },
        {
            "pattern": "network",
            "signals": ("connection refused", "name resolution", "temporary failure", "network"),
            "summary": "An external dependency or service endpoint is unavailable.",
            "root_cause": "The workflow depends on a network resource that is unreachable or misconfigured.",
            "actions": [
                "Check service availability and endpoint configuration.",
                "Retry only after validating that the dependency is reachable.",
            ],
            "confidence": 0.84,
        },
    )

    def run(self, input_data: dict) -> dict:
        failures = [str(item) for item in input_data.get("failures", []) if item]
        logs = str(input_data.get("logs", "") or "")
        context = dict(input_data.get("context", {}) or {})
        goal = str(input_data.get("goal") or context.get("goal") or "")
        history = list(input_data.get("history", []) or [])

        evidence = self._build_evidence(failures, logs, history)
        matches = self._match_patterns(evidence)
        if not matches:
            matches = [self._fallback_match(evidence)]

        repeated_failures = self._detect_repeated_failures(failures, history)
        recommended_actions = self._dedupe(
            action for match in matches for action in match["recommended_actions"]
        )
        if repeated_failures:
            recommended_actions.append(
                "The same failure is repeating. Revisit the plan or reduce scope before another retry."
            )

        summary_parts = [matches[0]["summary"]]
        if goal:
            summary_parts.append(f"Goal: {goal}.")
        if context.get("phase"):
            summary_parts.append(f"Phase: {context['phase']}.")
        if repeated_failures:
            summary_parts.append("Repeated failure pattern detected.")

        confidence = round(
            min(0.99, sum(match["confidence"] for match in matches) / max(len(matches), 1)),
            2,
        )

        return {
            "status": "analyzed",
            "summary": " ".join(summary_parts).strip(),
            "patterns": [match["pattern"] for match in matches],
            "likely_root_causes": [match["root_cause"] for match in matches],
            "recommended_actions": recommended_actions,
            "confidence": confidence,
            "evidence": evidence,
            "repeated_failure_detected": repeated_failures,
        }

    def _build_evidence(
        self,
        failures: List[str],
        logs: str,
        history: List[dict],
    ) -> Dict[str, Any]:
        condensed_logs = logs.strip()[:2000]
        repeated_signals = Counter(failure.strip().lower() for failure in failures if failure.strip())
        return {
            "failures": failures,
            "logs": condensed_logs,
            "history_count": len(history),
            "repeated_signals": dict(repeated_signals),
        }

    def _match_patterns(self, evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        haystack = "\n".join(evidence["failures"] + [evidence["logs"]]).lower()
        matches: List[Dict[str, Any]] = []
        for rule in self._PATTERN_RULES:
            if any(signal in haystack for signal in rule["signals"]):
                matches.append(
                    {
                        "pattern": rule["pattern"],
                        "summary": rule["summary"],
                        "root_cause": rule["root_cause"],
                        "recommended_actions": list(rule["actions"]),
                        "confidence": rule["confidence"],
                    }
                )
        return matches

    def _fallback_match(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pattern": "unknown_failure",
            "summary": "The failure does not match a known heuristic signature.",
            "root_cause": "Insufficient structured evidence to classify the failure automatically.",
            "recommended_actions": [
                "Inspect the full verification logs and the latest changed files.",
                "Capture a narrower reproduction with explicit failing inputs.",
            ],
            "confidence": 0.45,
        }

    def _detect_repeated_failures(self, failures: List[str], history: List[dict]) -> bool:
        if not failures or not history:
            return False
        failure_text = "\n".join(failures).lower()
        repeats = 0
        for entry in history:
            if failure_text and failure_text in str(entry).lower():
                repeats += 1
        return repeats >= 2

    def _dedupe(self, values) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered
