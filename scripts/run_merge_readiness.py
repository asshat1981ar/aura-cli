"""Evaluate merge readiness for a pull request."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.github_automation import (  # noqa: E402
    CheckRunStatus,
    MergeAdvisor,
    PRContext,
    ReviewGateStatus,
    ReviewSynthesisSnapshot,
    codeowners_required,
    load_codeowners,
)


def _load_payload() -> dict[str, object]:
    input_path = os.environ.get("MERGE_READINESS_INPUT_PATH", "").strip()
    if input_path:
        return json.loads(Path(input_path).read_text(encoding="utf-8"))

    return {
        "pull_request": {
            "number": int(os.environ["PR_NUMBER"]) if os.environ.get("PR_NUMBER") else None,
            "title": os.environ.get("PR_TITLE", ""),
            "base_ref": os.environ.get("BASE_REF", "main"),
            "head_ref": os.environ.get("HEAD_REF", ""),
            "draft": os.environ.get("PR_DRAFT", "false").lower() == "true",
            "labels": [label for label in os.environ.get("PR_LABELS", "").split(",") if label],
            "changed_files": json.loads(os.environ.get("PR_CHANGED_FILES_JSON", "[]")),
        },
        "reviews": {
            "approvals": int(os.environ.get("PR_APPROVALS", "0")),
            "changes_requested": int(os.environ.get("PR_CHANGES_REQUESTED", "0")),
            "review_decision": os.environ.get("PR_REVIEW_DECISION", ""),
            "requested_reviewers": [value for value in os.environ.get("PR_REQUESTED_REVIEWERS", "").split(",") if value],
            "requested_teams": [value for value in os.environ.get("PR_REQUESTED_TEAMS", "").split(",") if value],
        },
        "checks": json.loads(os.environ.get("PR_CHECK_RUNS_JSON", "[]")),
    }


def _load_synthesis(path: str | None) -> ReviewSynthesisSnapshot:
    if path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return ReviewSynthesisSnapshot.from_dict(payload)
    return ReviewSynthesisSnapshot(recommended_action="comment", human_review_required=False)


def _build_context(payload: dict[str, object]) -> PRContext:
    pull_request = payload.get("pull_request", {})
    return PRContext.from_changed_files(
        [str(path) for path in pull_request.get("changed_files", []) if path],
        number=int(pull_request["number"]) if pull_request.get("number") is not None else None,
        title=str(pull_request.get("title", "")),
        base_ref=str(pull_request.get("base_ref", "main")),
        head_ref=str(pull_request.get("head_ref", "")),
        draft=bool(pull_request.get("draft")),
        labels=[str(label) for label in pull_request.get("labels", []) if label],
    )


def _build_reviews(payload: dict[str, object], changed_files: list[str]) -> ReviewGateStatus:
    reviews = payload.get("reviews", {})
    rules = load_codeowners(ROOT / ".github" / "CODEOWNERS")
    requested_reviewers = [str(value) for value in reviews.get("requested_reviewers", []) if value]
    requested_teams = [str(value) for value in reviews.get("requested_teams", []) if value]
    required = codeowners_required(changed_files, rules)
    return ReviewGateStatus(
        approvals=int(reviews.get("approvals", 0)),
        changes_requested=int(reviews.get("changes_requested", 0)),
        review_decision=str(reviews.get("review_decision", "")),
        requested_reviewers=requested_reviewers,
        requested_teams=requested_teams,
        codeowners_required=required,
        codeowners_pending=required and bool(requested_reviewers or requested_teams),
    )


def _build_checks(payload: dict[str, object]) -> list[CheckRunStatus]:
    return [CheckRunStatus.from_dict(check, required=bool(check.get("required", True))) for check in payload.get("checks", [])]


def _write_outputs(summary_path: Path, comment_path: Path, decision) -> None:
    summary_path.write_text(json.dumps(decision.to_dict(), indent=2) + "\n", encoding="utf-8")
    comment_path.write_text(decision.comment_markdown + "\n", encoding="utf-8")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return

    with open(github_output, "a", encoding="utf-8") as handle:
        handle.write(f"status={decision.status}\n")
        handle.write(f"ready_for_merge={'true' if decision.ready_for_merge else 'false'}\n")
        handle.write(f"ready_for_queue={'true' if decision.ready_for_queue else 'false'}\n")
        handle.write(f"labels_to_add={','.join(decision.labels_to_add)}\n")
        handle.write(f"labels_to_remove={','.join(decision.labels_to_remove)}\n")


def main() -> None:
    payload = _load_payload()
    context = _build_context(payload)
    synthesis = _load_synthesis(os.environ.get("PR_REVIEW_SUMMARY_PATH"))
    reviews = _build_reviews(payload, context.changed_files)
    checks = _build_checks(payload)
    decision = MergeAdvisor().evaluate(context, synthesis, reviews, checks)
    _write_outputs(Path("merge-readiness-summary.json"), Path("merge-readiness-comment.md"), decision)


if __name__ == "__main__":
    main()
