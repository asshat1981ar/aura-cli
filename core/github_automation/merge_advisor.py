"""Merge-readiness evaluation for pull requests."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from core.github_automation.pr_context import PRContext

MergeStatus = Literal["merge_ready", "waiting", "blocked"]

DEFAULT_REQUIRED_CHECKS = (
    "CI / lint",
    "CI / test (3.10)",
    "CI / test (3.12)",
    "CI / typecheck",
    "CI / cli_docs_and_help_contracts",
    "PR Review Orchestrator / synthesize",
)

_SUCCESS_CONCLUSIONS = {"success", "neutral", "skipped"}
_FAILURE_CONCLUSIONS = {"action_required", "cancelled", "failure", "stale", "startup_failure", "timed_out"}
_PENDING_STATUSES = {"expected", "in_progress", "pending", "queued", "requested", "waiting"}


@dataclass(slots=True)
class CheckRunStatus:
    """Normalized status for a required or optional check run."""

    name: str
    status: str
    conclusion: str | None = None
    required: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, required: bool = True) -> "CheckRunStatus":
        return cls(
            name=str(payload.get("name") or "").strip(),
            status=str(payload.get("status") or "missing").strip().lower(),
            conclusion=_normalize_optional(payload.get("conclusion")),
            required=required,
        )

    @property
    def is_success(self) -> bool:
        return self.conclusion in _SUCCESS_CONCLUSIONS

    @property
    def is_failure(self) -> bool:
        return self.conclusion in _FAILURE_CONCLUSIONS

    @property
    def is_pending(self) -> bool:
        return self.status == "missing" or self.status in _PENDING_STATUSES or (
            self.status == "completed" and self.conclusion not in _SUCCESS_CONCLUSIONS | _FAILURE_CONCLUSIONS
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReviewSynthesisSnapshot:
    """Subset of synthesized PR review output needed by merge gating."""

    recommended_action: str
    human_review_required: bool
    blocked_reasons: list[str] = field(default_factory=list)
    finding_counts: dict[str, int] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)
    summary: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReviewSynthesisSnapshot":
        return cls(
            recommended_action=str(payload.get("recommended_action") or "comment"),
            human_review_required=bool(payload.get("human_review_required")),
            blocked_reasons=[str(value) for value in payload.get("blocked_reasons", []) if value],
            finding_counts={
                level: int(payload.get("finding_counts", {}).get(level, 0))
                for level in ("critical", "high", "medium", "low", "info")
            },
            labels=[str(value) for value in payload.get("labels", []) if value],
            summary=str(payload.get("summary") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReviewGateStatus:
    """Human-review and CODEOWNERS status for the pull request."""

    approvals: int = 0
    changes_requested: int = 0
    review_decision: str = ""
    requested_reviewers: list[str] = field(default_factory=list)
    requested_teams: list[str] = field(default_factory=list)
    codeowners_required: bool = False
    codeowners_pending: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MergeReadinessDecision:
    """Single merge-readiness verdict used by GitHub workflows."""

    status: MergeStatus
    ready_for_merge: bool
    ready_for_queue: bool
    auto_merge_recommended: bool
    labels_to_add: list[str]
    labels_to_remove: list[str]
    blockers: list[str]
    warnings: list[str]
    summary: str
    comment_markdown: str
    required_checks: list[CheckRunStatus] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_checks"] = [check.to_dict() for check in self.required_checks]
        return payload


@dataclass(slots=True)
class CodeownersRule:
    """Single CODEOWNERS rule."""

    pattern: str
    owners: list[str]

    def matches(self, path: str) -> bool:
        normalized = _normalize_path(path)
        if self.pattern == "*":
            return True
        if self.pattern.endswith("/"):
            return normalized.startswith(self.pattern.lstrip("/"))
        normalized_pattern = self.pattern.lstrip("/")
        if "/" not in normalized_pattern:
            return fnmatch(PurePosixPath(normalized).name, normalized_pattern)
        return fnmatch(normalized, normalized_pattern)


def parse_codeowners(text: str) -> list[CodeownersRule]:
    """Parse a minimal subset of CODEOWNERS rules."""
    rules: list[CodeownersRule] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        rules.append(CodeownersRule(pattern=parts[0], owners=parts[1:]))
    return rules


def load_codeowners(path: str | Path) -> list[CodeownersRule]:
    """Load CODEOWNERS rules from disk if the file exists."""
    codeowners_path = Path(path)
    if not codeowners_path.exists():
        return []
    return parse_codeowners(codeowners_path.read_text(encoding="utf-8"))


def codeowners_required(changed_files: list[str], rules: list[CodeownersRule]) -> bool:
    """Return True when any changed file matches a CODEOWNERS rule."""
    if not changed_files or not rules:
        return False
    return any(rule.matches(path) for path in changed_files for rule in rules)


class MergeAdvisor:
    """Evaluate whether a PR is ready for auto-merge or merge queue."""

    def __init__(
        self,
        *,
        required_checks: tuple[str, ...] = DEFAULT_REQUIRED_CHECKS,
        minimum_approvals: int = 1,
        merge_queue_enabled: bool = True,
    ):
        self.required_checks = required_checks
        self.minimum_approvals = minimum_approvals
        self.merge_queue_enabled = merge_queue_enabled

    def evaluate(
        self,
        context: PRContext,
        synthesis: ReviewSynthesisSnapshot,
        reviews: ReviewGateStatus,
        checks: list[CheckRunStatus],
    ) -> MergeReadinessDecision:
        required_checks = self._required_check_statuses(checks)
        blockers: list[str] = []
        warnings: list[str] = []
        status: MergeStatus = "merge_ready"

        if context.draft:
            status = _worse_status(status, "waiting")
            blockers.append("Pull request is still a draft.")

        if reviews.changes_requested > 0 or reviews.review_decision.upper() == "CHANGES_REQUESTED":
            status = _worse_status(status, "blocked")
            blockers.append("A reviewer has requested changes.")

        failed_checks = [check.name for check in required_checks if check.is_failure]
        pending_checks = [check.name for check in required_checks if check.is_pending and not check.is_failure]
        if failed_checks:
            status = _worse_status(status, "blocked")
            blockers.append(f"Required checks failed: {', '.join(failed_checks)}.")
        elif pending_checks:
            status = _worse_status(status, "waiting")
            blockers.append(f"Required checks are still pending: {', '.join(pending_checks)}.")

        if synthesis.recommended_action == "request_changes":
            status = _worse_status(status, "blocked")
            blockers.extend(synthesis.blocked_reasons or ["Synthesized review requested changes before merge."])
        elif synthesis.recommended_action == "comment":
            status = _worse_status(status, "waiting")
            blockers.append("Synthesized review still has follow-up findings to resolve.")
        elif synthesis.recommended_action == "escalate":
            warnings.append("Synthesized review requires a human approval before merge.")

        human_review_required = (
            synthesis.human_review_required
            or reviews.codeowners_required
            or context.touches_protected_paths
            or context.touches_dependencies
            or context.touches_workflows
        )
        if reviews.approvals < self.minimum_approvals:
            status = _worse_status(status, "waiting")
            blockers.append(f"At least {self.minimum_approvals} approval(s) are required before merge.")
        if reviews.codeowners_required and reviews.codeowners_pending:
            status = _worse_status(status, "waiting")
            blockers.append("CODEOWNERS review is still pending.")
        elif human_review_required and reviews.approvals >= self.minimum_approvals:
            warnings.append("Human approval requirement is satisfied.")

        ready_for_merge = status == "merge_ready"
        ready_for_queue = ready_for_merge and self.merge_queue_enabled
        labels_to_add = ["merge-ready"] if ready_for_merge else (["blocked"] if status == "blocked" else [])
        labels_to_remove = ["merge-ready"] if not ready_for_merge else ["blocked", "needs-human-review"]

        summary = self._summary(status, blockers, warnings)
        comment_markdown = self.render_comment(context, synthesis, reviews, required_checks, status, blockers, warnings, summary)
        return MergeReadinessDecision(
            status=status,
            ready_for_merge=ready_for_merge,
            ready_for_queue=ready_for_queue,
            auto_merge_recommended=ready_for_merge,
            labels_to_add=_dedupe(labels_to_add),
            labels_to_remove=_dedupe(labels_to_remove),
            blockers=_dedupe(blockers),
            warnings=_dedupe(warnings),
            summary=summary,
            comment_markdown=comment_markdown,
            required_checks=required_checks,
        )

    def _required_check_statuses(self, checks: list[CheckRunStatus]) -> list[CheckRunStatus]:
        by_name = {check.name: check for check in checks if check.name}
        resolved: list[CheckRunStatus] = []
        for name in self.required_checks:
            resolved.append(by_name.get(name, CheckRunStatus(name=name, status="missing", conclusion=None, required=True)))
        return resolved

    def _summary(self, status: MergeStatus, blockers: list[str], warnings: list[str]) -> str:
        if status == "merge_ready":
            return "Required checks, approvals, and review policy gates are satisfied for merge queue or auto-merge."
        if blockers:
            return blockers[0]
        if warnings:
            return warnings[0]
        return "Merge readiness could not be determined."

    def render_comment(
        self,
        context: PRContext,
        synthesis: ReviewSynthesisSnapshot,
        reviews: ReviewGateStatus,
        checks: list[CheckRunStatus],
        status: MergeStatus,
        blockers: list[str],
        warnings: list[str],
        summary: str,
        *,
        marker: str = "<!-- aura-merge-readiness -->",
    ) -> str:
        lines = [
            marker,
            "## AURA Merge Readiness",
            "",
            f"**Status:** `{status}`",
            f"**Auto-merge recommended:** `{'yes' if status == 'merge_ready' else 'no'}`",
            f"**Merge queue ready:** `{'yes' if status == 'merge_ready' and self.merge_queue_enabled else 'no'}`",
            f"**Approvals:** `{reviews.approvals}`",
            f"**Changes requested:** `{reviews.changes_requested}`",
            f"**Human review required:** `{'yes' if synthesis.human_review_required or reviews.codeowners_required else 'no'}`",
            f"**Changed files:** `{context.touched_file_count}`",
            "",
            "### Required Checks",
            "",
        ]
        lines.extend(
            f"- `{check.name}` status=`{check.status}` conclusion=`{check.conclusion or 'n/a'}`"
            for check in checks
        )
        if blockers:
            lines.extend(["", "### Blockers", ""])
            lines.extend(f"- {blocker}" for blocker in _dedupe(blockers))
        if warnings:
            lines.extend(["", "### Notes", ""])
            lines.extend(f"- {warning}" for warning in _dedupe(warnings))
        lines.extend(["", f"_Summary: {summary}_", ""])
        return "\n".join(lines)


def _normalize_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _normalize_path(path: str) -> str:
    normalized = path.strip()
    if normalized.startswith("./"):
        return normalized[2:]
    return normalized.lstrip("/")


def _dedupe(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return seen


def _worse_status(current: MergeStatus, candidate: MergeStatus) -> MergeStatus:
    rank = {"merge_ready": 0, "waiting": 1, "blocked": 2}
    return candidate if rank[candidate] > rank[current] else current
