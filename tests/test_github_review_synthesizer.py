from __future__ import annotations

from core.github_automation import (
    PRContext,
    PolicyDecision,
    ProviderReview,
    ReviewFinding,
    ReviewSynthesizer,
)


def test_synthesizer_counts_findings_by_severity() -> None:
    context = PRContext.from_changed_files(["core/workflow_engine.py"])
    review = ProviderReview(
        provider="aura",
        summary="review summary",
        findings=[
            ReviewFinding(
                severity="high",
                path="core/workflow_engine.py",
                line=12,
                title="High issue",
                detail="Needs attention.",
                confidence=0.88,
                category="correctness",
            ),
            ReviewFinding(
                severity="low",
                path="tests/test_workflow_engine.py",
                line=8,
                title="Low issue",
                detail="Add more coverage.",
                confidence=0.55,
                category="test",
            ),
        ],
    )
    decision = PolicyDecision(
        recommended_action="comment",
        labels=["agent-review"],
        summary="Review findings recorded.",
    )
    result = ReviewSynthesizer().synthesize(context, [review], decision, planned_providers=["aura", "copilot"])
    assert result.finding_counts == {
        "critical": 0,
        "high": 1,
        "medium": 0,
        "low": 1,
        "info": 0,
    }
    assert result.providers_consulted == ["aura", "copilot"]


def test_rendered_markdown_includes_marker_and_blockers() -> None:
    context = PRContext.from_changed_files([".github/workflows/ci.yml"])
    review = ProviderReview(provider="aura", summary="workflow changed")
    decision = PolicyDecision(
        recommended_action="escalate",
        human_review_required=True,
        blocked_reasons=["Protected-path changes require a human reviewer."],
        labels=["agent-review", "needs-human-review"],
        summary="Protected-path changes require a human reviewer.",
    )
    result = ReviewSynthesizer().synthesize(context, [review], decision, planned_providers=["aura", "claude"])
    assert "<!-- aura-pr-review -->" in result.comment_markdown
    assert "Protected-path changes require a human reviewer." in result.comment_markdown
    assert "`aura, claude`" in result.comment_markdown
