from __future__ import annotations

from core.github_automation import PRContext, ProviderReview, ReviewFinding, evaluate_policy


def test_critical_finding_blocks_merge() -> None:
    context = PRContext.from_changed_files(["core/human_gate.py"])
    review = ProviderReview(
        provider="aura",
        summary="critical issue found",
        findings=[
            ReviewFinding(
                severity="critical",
                path="core/human_gate.py",
                line=10,
                title="Critical failure",
                detail="This must be fixed before merge.",
                confidence=0.99,
                category="correctness",
            )
        ],
    )
    decision = evaluate_policy(context, [review])
    assert decision.recommended_action == "request_changes"
    assert decision.human_review_required is True
    assert "blocked" in decision.labels


def test_protected_path_change_escalates_even_without_findings() -> None:
    context = PRContext.from_changed_files([".github/workflows/ci.yml"])
    review = ProviderReview(provider="aura", summary="workflow touched")
    decision = evaluate_policy(context, [review])
    assert decision.recommended_action == "escalate"
    assert decision.human_review_required is True
    assert "needs-human-review" in decision.labels


def test_clean_docs_change_is_merge_ready() -> None:
    context = PRContext.from_changed_files(["README.md", "plans/github-agent-automation-implementation-plan.md"])
    review = ProviderReview(provider="aura", summary="docs only")
    decision = evaluate_policy(context, [review])
    assert decision.recommended_action == "approve"
    assert "merge-ready" in decision.labels
