from __future__ import annotations

from core.github_automation import PRContext
from core.github_automation.merge_advisor import (
    CheckRunStatus,
    MergeAdvisor,
    ReviewGateStatus,
    ReviewSynthesisSnapshot,
    codeowners_required,
    parse_codeowners,
)


def _required_checks() -> list[CheckRunStatus]:
    return [
        CheckRunStatus(name="CI / lint", status="completed", conclusion="success"),
        CheckRunStatus(name="CI / test (3.10)", status="completed", conclusion="success"),
        CheckRunStatus(name="CI / test (3.12)", status="completed", conclusion="success"),
        CheckRunStatus(name="CI / typecheck", status="completed", conclusion="success"),
        CheckRunStatus(name="CI / cli_docs_and_help_contracts", status="completed", conclusion="success"),
        CheckRunStatus(name="PR Review Orchestrator / synthesize", status="completed", conclusion="success"),
    ]


def test_clean_pr_becomes_merge_ready() -> None:
    context = PRContext.from_changed_files(["README.md"], number=7, title="Docs tidy-up")
    synthesis = ReviewSynthesisSnapshot(recommended_action="approve", human_review_required=False)
    reviews = ReviewGateStatus(approvals=1, review_decision="APPROVED")

    decision = MergeAdvisor().evaluate(context, synthesis, reviews, _required_checks())

    assert decision.status == "merge_ready"
    assert decision.ready_for_merge is True
    assert "merge-ready" in decision.labels_to_add


def test_protected_path_with_escalation_can_become_ready_after_approval() -> None:
    context = PRContext.from_changed_files([".github/workflows/ci.yml"], number=9, title="Adjust CI")
    synthesis = ReviewSynthesisSnapshot(
        recommended_action="escalate",
        human_review_required=True,
        blocked_reasons=["protected path changes require a human review (.github/workflows/ci.yml)"],
    )
    reviews = ReviewGateStatus(
        approvals=1,
        review_decision="APPROVED",
        codeowners_required=True,
        codeowners_pending=False,
    )

    decision = MergeAdvisor().evaluate(context, synthesis, reviews, _required_checks())

    assert decision.status == "merge_ready"
    assert decision.ready_for_queue is True
    assert "merge-ready" in decision.labels_to_add


def test_missing_required_check_waits() -> None:
    context = PRContext.from_changed_files(["core/workflow_engine.py"], number=10, title="Queue fix")
    synthesis = ReviewSynthesisSnapshot(recommended_action="approve", human_review_required=False)
    reviews = ReviewGateStatus(approvals=1, review_decision="APPROVED")
    checks = _required_checks()[:-1]

    decision = MergeAdvisor().evaluate(context, synthesis, reviews, checks)

    assert decision.status == "waiting"
    assert decision.ready_for_merge is False
    assert any("pending" in blocker.lower() for blocker in decision.blockers)


def test_requested_changes_blocks_merge() -> None:
    context = PRContext.from_changed_files(["core/workflow_engine.py"], number=11, title="Queue fix")
    synthesis = ReviewSynthesisSnapshot(recommended_action="approve", human_review_required=False)
    reviews = ReviewGateStatus(approvals=0, changes_requested=1, review_decision="CHANGES_REQUESTED")

    decision = MergeAdvisor().evaluate(context, synthesis, reviews, _required_checks())

    assert decision.status == "blocked"
    assert decision.ready_for_merge is False
    assert "blocked" in decision.labels_to_add


def test_comment_verdict_holds_merge_ready_label() -> None:
    context = PRContext.from_changed_files(["core/workflow_engine.py"], number=12, title="Queue fix")
    synthesis = ReviewSynthesisSnapshot(
        recommended_action="comment",
        human_review_required=False,
        finding_counts={"low": 1},
        summary="Review findings were recorded for follow-up.",
    )
    reviews = ReviewGateStatus(approvals=1, review_decision="APPROVED")

    decision = MergeAdvisor().evaluate(context, synthesis, reviews, _required_checks())

    assert decision.status == "waiting"
    assert "merge-ready" in decision.labels_to_remove


def test_codeowners_rules_apply_to_changed_files() -> None:
    rules = parse_codeowners(
        """
        * @repo-owner
        .github/ @repo-owner
        tools/ @repo-owner
        """.strip()
    )
    assert codeowners_required(["README.md"], rules) is True
    assert codeowners_required(["tools/github_tools.py"], rules) is True
