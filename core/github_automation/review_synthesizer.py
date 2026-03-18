"""Formatting and synthesis helpers for PR review outputs."""
from __future__ import annotations

from collections import Counter

from core.github_automation.models import PolicyDecision, ProviderReview, ReviewFinding, SynthesisResult
from core.github_automation.pr_context import PRContext

COMMENT_MARKER = "<!-- aura-pr-review -->"


class ReviewSynthesizer:
    """Combine provider reviews into one repo-owned summary."""

    def merge_findings(self, reviews: list[ProviderReview]) -> list[ReviewFinding]:
        merged: dict[tuple[str, str, int | None, str, str, str], ReviewFinding] = {}
        for review in reviews:
            for finding in review.findings:
                key = (
                    finding.severity,
                    finding.path,
                    finding.line,
                    finding.title,
                    finding.detail,
                    finding.category,
                )
                existing = merged.get(key)
                if existing is None or finding.confidence > existing.confidence:
                    merged[key] = finding
        return sorted(merged.values(), key=lambda finding: finding.sort_key())

    def count_findings(self, findings: list[ReviewFinding]) -> dict[str, int]:
        counts = Counter(finding.severity for finding in findings)
        return {level: counts.get(level, 0) for level in ("critical", "high", "medium", "low", "info")}

    def synthesize(
        self,
        context: PRContext,
        reviews: list[ProviderReview],
        decision: PolicyDecision,
        *,
        planned_providers: list[str] | None = None,
    ) -> SynthesisResult:
        findings = self.merge_findings(reviews)
        counts = self.count_findings(findings)
        providers = planned_providers or [review.provider for review in reviews]
        comment_markdown = self.render_comment(context, findings, decision, providers)
        return SynthesisResult(
            summary=decision.summary,
            recommended_action=decision.recommended_action,
            providers_consulted=providers,
            finding_counts=counts,
            labels=decision.labels,
            human_review_required=decision.human_review_required,
            blocked_reasons=decision.blocked_reasons,
            comment_markdown=comment_markdown,
            findings=findings,
        )

    def render_comment(
        self,
        context: PRContext,
        findings: list[ReviewFinding],
        decision: PolicyDecision,
        providers: list[str],
    ) -> str:
        counts = self.count_findings(findings)
        lines = [
            COMMENT_MARKER,
            "## AURA PR Review",
            "",
            f"**Verdict:** `{decision.recommended_action}`",
            f"**Providers planned:** `{', '.join(providers)}`",
            f"**Human review required:** `{'yes' if decision.human_review_required else 'no'}`",
            f"**Changed files:** `{context.touched_file_count}`",
            "",
            "### Finding Counts",
            "",
            f"- `critical`: {counts['critical']}",
            f"- `high`: {counts['high']}",
            f"- `medium`: {counts['medium']}",
            f"- `low`: {counts['low']}",
            f"- `info`: {counts['info']}",
            "",
        ]

        if decision.blocked_reasons:
            lines.extend(["### Blockers", ""])
            lines.extend(f"- {reason}" for reason in decision.blocked_reasons)
            lines.append("")

        lines.extend(["### Findings", ""])
        if findings:
            for finding in findings[:10]:
                location = finding.path
                if finding.line:
                    location = f"{location}:{finding.line}"
                lines.append(
                    f"- `{finding.severity}` `{location}` {finding.title} "
                    f"(category: {finding.category}, confidence: {finding.confidence:.2f})"
                )
                lines.append(f"  {finding.detail}")
        else:
            lines.append("- No normalized findings were recorded.")
        lines.append("")
        lines.append(f"_Summary: {decision.summary}_")
        lines.append("")
        return "\n".join(lines)
