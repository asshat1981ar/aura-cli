"""Policy evaluation for synthesized GitHub review results."""
from __future__ import annotations

from core.github_automation.models import PolicyDecision, ProviderReview
from core.github_automation.pr_context import PRContext
from core.human_gate import HumanGate


class ReviewPolicy:
    """Applies repo-level merge and escalation rules."""

    def __init__(self, human_gate: HumanGate | None = None):
        self.human_gate = human_gate or HumanGate()

    def evaluate(self, context: PRContext, reviews: list[ProviderReview]) -> PolicyDecision:
        findings = [finding for review in reviews for finding in review.findings]
        counts = {level: 0 for level in ("critical", "high", "medium", "low", "info")}
        for finding in findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1

        labels = ["agent-review"]
        risk_label = "risk:low" if context.docs_only else "risk:medium"
        if context.touches_workflows or context.touches_dependencies or context.touches_protected_paths:
            risk_label = "risk:high"
        labels.append(risk_label)

        blocked_reasons: list[str] = []
        human_review_required = False

        paths_blocked, path_reason = self.human_gate.should_block_paths(context.changed_files)
        if paths_blocked and path_reason:
            blocked_reasons.append(path_reason)
            human_review_required = True

        if counts["critical"] > 0:
            blocked_reasons.append("Critical findings must be resolved before merge.")
            human_review_required = True
            labels.append("blocked")
            labels.append("needs-human-review")
            return PolicyDecision(
                recommended_action="request_changes",
                human_review_required=human_review_required,
                blocked_reasons=blocked_reasons,
                labels=_dedupe(labels),
                summary="Critical findings or protected-path changes require manual follow-up.",
            )

        if counts["high"] > 0:
            blocked_reasons.append("High-severity findings require manual review.")
            human_review_required = True
            labels.append("blocked")
            labels.append("needs-human-review")
            return PolicyDecision(
                recommended_action="request_changes",
                human_review_required=human_review_required,
                blocked_reasons=blocked_reasons,
                labels=_dedupe(labels),
                summary="High-severity findings require manual review before merge.",
            )

        if human_review_required:
            labels.append("needs-human-review")
            return PolicyDecision(
                recommended_action="escalate",
                human_review_required=True,
                blocked_reasons=blocked_reasons,
                labels=_dedupe(labels),
                summary="Protected-path changes require a human reviewer.",
            )

        if context.draft:
            return PolicyDecision(
                recommended_action="comment",
                human_review_required=False,
                blocked_reasons=[],
                labels=_dedupe(labels),
                summary="Draft pull request reviewed; final verdict deferred until ready for review.",
            )

        if findings:
            return PolicyDecision(
                recommended_action="comment",
                human_review_required=False,
                blocked_reasons=[],
                labels=_dedupe(labels),
                summary="Review findings were recorded for follow-up.",
            )

        labels.append("merge-ready")
        return PolicyDecision(
            recommended_action="approve",
            human_review_required=False,
            blocked_reasons=[],
            labels=_dedupe(labels),
            summary="No blocking findings detected by the synthesized review policy.",
        )


def evaluate_policy(context: PRContext, reviews: list[ProviderReview], gate: HumanGate | None = None) -> PolicyDecision:
    """Convenience wrapper for single-call policy evaluation."""
    return ReviewPolicy(gate).evaluate(context, reviews)


def _dedupe(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen
