"""Deterministic investigation agent for repeated workflow failures."""

from __future__ import annotations

from typing import Any, Dict

from core.investigate_test_drop import investigate_test_count_drop
from core.investigate_verification_failures import investigate_verification_failure
from core.remediation_plan import build_remediation_plan


class InvestigationAgent:
    """Aggregate existing failure-analysis helpers into one investigation report."""

    name = "investigation"

    def run(self, input_data: Dict[str, Any] | None) -> Dict[str, Any]:
        input_data = input_data if isinstance(input_data, dict) else {}

        goal = input_data.get("goal")
        verification = input_data.get("verification", {})
        context = input_data.get("context", {})
        history = input_data.get("history", [])
        root_cause_analysis = input_data.get("root_cause_analysis", {})
        route = str(input_data.get("route", "act"))

        verification_investigation = investigate_verification_failure(
            verification,
            root_cause_analysis=root_cause_analysis,
            history=history,
            context=context,
        )
        remediation_plan = build_remediation_plan(
            verification,
            route=route,
            analysis_suggestion=input_data.get("analysis_suggestion"),
            root_cause_analysis=root_cause_analysis,
            investigation=verification_investigation,
            context=context,
        )

        test_drop = None
        previous_test_count = input_data.get("previous_test_count")
        current_test_count = input_data.get("current_test_count")
        if previous_test_count is not None and current_test_count is not None:
            test_drop = investigate_test_count_drop(
                previous_test_count,
                current_test_count,
                goal=goal,
                verification=verification,
                remediation_plan=remediation_plan,
            )

        summary_parts = [verification_investigation["summary"], remediation_plan["summary"]]
        if test_drop:
            summary_parts.append(test_drop["summary"])

        return {
            "status": "investigated",
            "goal": goal,
            "verification_investigation": verification_investigation,
            "remediation_plan": remediation_plan,
            "test_drop_investigation": test_drop,
            "summary": " ".join(part for part in summary_parts if part),
        }
