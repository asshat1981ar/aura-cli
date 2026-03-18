"""Nightly repository health summaries for GitHub automation."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
import re
from typing import Any


def _parse_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _slugify(value: str) -> str:
    collapsed = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return collapsed[:64] or "repo-health"


def _hotspot_key(path: str) -> str:
    normalized = path.strip().lstrip("./")
    parts = PurePosixPath(normalized).parts
    if not parts:
        return "."
    if len(parts) == 1:
        return parts[0]
    if parts[0].startswith("."):
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


@dataclass(slots=True)
class PullRequestSnapshot:
    """Minimal PR data used by nightly health analysis."""

    number: int
    title: str
    updated_at: str
    draft: bool = False
    labels: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    author: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PullRequestSnapshot":
        return cls(
            number=int(payload.get("number", 0)),
            title=str(payload.get("title", "")),
            updated_at=str(payload.get("updated_at", "")),
            draft=bool(payload.get("draft")),
            labels=[str(value) for value in payload.get("labels", []) if value],
            changed_files=[str(value) for value in payload.get("changed_files", []) if value],
            author=str(payload.get("author", "")),
        )

    def age_days(self, now: datetime) -> float:
        updated = _parse_datetime(self.updated_at)
        return max((now - updated).total_seconds() / 86400.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WorkflowRunSnapshot:
    """Minimal workflow-run data used by nightly health analysis."""

    id: int
    name: str
    conclusion: str
    created_at: str
    head_branch: str = ""
    display_title: str = ""
    failed_jobs: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkflowRunSnapshot":
        return cls(
            id=int(payload.get("id", 0)),
            name=str(payload.get("name", "")),
            conclusion=str(payload.get("conclusion", "")).lower(),
            created_at=str(payload.get("created_at", "")),
            head_branch=str(payload.get("head_branch", "")),
            display_title=str(payload.get("display_title", "")),
            failed_jobs=[str(value) for value in payload.get("failed_jobs", []) if value],
        )

    def created_dt(self) -> datetime:
        return _parse_datetime(self.created_at)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StalePullRequest:
    """Open pull request that has been idle beyond the threshold."""

    number: int
    title: str
    days_stale: int
    draft: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FlakyWorkflow:
    """Workflow with both recent passes and failures."""

    name: str
    total_runs: int
    failure_count: int
    success_count: int
    failure_rate: float
    last_failure_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FailureCluster:
    """Recurring failure signature across recent workflow runs."""

    signature: str
    count: int
    last_seen_at: str
    run_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PathHotspot:
    """Area touched by multiple active PRs."""

    path: str
    pull_request_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FollowUpIssue:
    """Issue suggested by the nightly health analyzer."""

    title: str
    body: str
    labels: list[str]
    marker: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RepoHealthReport:
    """Full nightly repository health output."""

    generated_at: str
    summary: str
    stale_pull_requests: list[StalePullRequest] = field(default_factory=list)
    flaky_workflows: list[FlakyWorkflow] = field(default_factory=list)
    recurring_failures: list[FailureCluster] = field(default_factory=list)
    hotspots: list[PathHotspot] = field(default_factory=list)
    follow_up_issues: list[FollowUpIssue] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "summary": self.summary,
            "stale_pull_requests": [entry.to_dict() for entry in self.stale_pull_requests],
            "flaky_workflows": [entry.to_dict() for entry in self.flaky_workflows],
            "recurring_failures": [entry.to_dict() for entry in self.recurring_failures],
            "hotspots": [entry.to_dict() for entry in self.hotspots],
            "follow_up_issues": [entry.to_dict() for entry in self.follow_up_issues],
            "notes": list(self.notes),
        }

    def render_markdown(self, *, marker: str = "<!-- aura-repo-health -->") -> str:
        lines = [
            marker,
            "## AURA Nightly Repo Health",
            "",
            f"**Generated at:** `{self.generated_at}`",
            "",
            self.summary,
            "",
            "### Stale Pull Requests",
            "",
        ]
        if self.stale_pull_requests:
            lines.extend(
                f"- `#{entry.number}` {entry.title} (`{entry.days_stale}` day(s) stale, draft=`{'yes' if entry.draft else 'no'}`)"
                for entry in self.stale_pull_requests
            )
        else:
            lines.append("- No stale pull requests crossed the alert threshold.")

        lines.extend(["", "### Flaky Workflows", ""])
        if self.flaky_workflows:
            lines.extend(
                f"- `{entry.name}` failures=`{entry.failure_count}` successes=`{entry.success_count}` failure_rate=`{entry.failure_rate:.0%}` last_failure=`{entry.last_failure_at}`"
                for entry in self.flaky_workflows
            )
        else:
            lines.append("- No flaky workflows were detected in the recent run window.")

        lines.extend(["", "### Recurring Failure Clusters", ""])
        if self.recurring_failures:
            lines.extend(
                f"- `{entry.signature}` count=`{entry.count}` last_seen=`{entry.last_seen_at}` runs=`{', '.join(str(run_id) for run_id in entry.run_ids[:5])}`"
                for entry in self.recurring_failures
            )
        else:
            lines.append("- No recurring failure clusters crossed the alert threshold.")

        lines.extend(["", "### Active Hotspots", ""])
        if self.hotspots:
            lines.extend(
                f"- `{entry.path}` touched by `{entry.pull_request_count}` open PR(s)"
                for entry in self.hotspots
            )
        else:
            lines.append("- No path hotspots were detected from the active PR set.")

        if self.notes:
            lines.extend(["", "### Notes", ""])
            lines.extend(f"- {note}" for note in self.notes)

        lines.append("")
        return "\n".join(lines)


class RepoHealthAnalyzer:
    """Build a deterministic nightly health report from GitHub snapshots."""

    def __init__(
        self,
        *,
        now: datetime | None = None,
        stale_days: int = 7,
        recurring_failure_threshold: int = 2,
        follow_up_issue_threshold: int = 3,
        hotspot_limit: int = 5,
    ):
        self.now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        self.stale_days = stale_days
        self.recurring_failure_threshold = recurring_failure_threshold
        self.follow_up_issue_threshold = follow_up_issue_threshold
        self.hotspot_limit = hotspot_limit

    def build_report(
        self,
        pull_requests: list[PullRequestSnapshot],
        workflow_runs: list[WorkflowRunSnapshot],
    ) -> RepoHealthReport:
        stale_pull_requests = self._stale_pull_requests(pull_requests)
        flaky_workflows = self._flaky_workflows(workflow_runs)
        recurring_failures = self._recurring_failures(workflow_runs)
        hotspots = self._hotspots(pull_requests)
        follow_ups = self._follow_up_issues(recurring_failures)
        notes = [
            "Nightly health is advisory only and should not be used as a required PR gate.",
            "Automation-generated code changes must land through a pull request, never a direct push to main.",
        ]
        summary = self._summary(stale_pull_requests, flaky_workflows, recurring_failures, hotspots, follow_ups)
        return RepoHealthReport(
            generated_at=self.now.isoformat(),
            summary=summary,
            stale_pull_requests=stale_pull_requests,
            flaky_workflows=flaky_workflows,
            recurring_failures=recurring_failures,
            hotspots=hotspots,
            follow_up_issues=follow_ups,
            notes=notes,
        )

    def _stale_pull_requests(self, pull_requests: list[PullRequestSnapshot]) -> list[StalePullRequest]:
        stale: list[StalePullRequest] = []
        for pr in pull_requests:
            age_days = int(pr.age_days(self.now))
            if age_days >= self.stale_days:
                stale.append(
                    StalePullRequest(
                        number=pr.number,
                        title=pr.title,
                        days_stale=age_days,
                        draft=pr.draft,
                    )
                )
        return sorted(stale, key=lambda entry: (-entry.days_stale, entry.number))

    def _flaky_workflows(self, workflow_runs: list[WorkflowRunSnapshot]) -> list[FlakyWorkflow]:
        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        last_failure: dict[str, str] = {}
        for run in workflow_runs:
            if run.conclusion not in {"success", "failure"}:
                continue
            grouped[run.name][run.conclusion] += 1
            if run.conclusion == "failure":
                seen = last_failure.get(run.name)
                if seen is None or run.created_at > seen:
                    last_failure[run.name] = run.created_at

        flaky: list[FlakyWorkflow] = []
        for name, counts in grouped.items():
            failures = counts.get("failure", 0)
            successes = counts.get("success", 0)
            total = failures + successes
            if failures >= 2 and successes >= 1:
                flaky.append(
                    FlakyWorkflow(
                        name=name,
                        total_runs=total,
                        failure_count=failures,
                        success_count=successes,
                        failure_rate=failures / total if total else 0.0,
                        last_failure_at=last_failure.get(name, ""),
                    )
                )
        return sorted(flaky, key=lambda entry: (-entry.failure_rate, -entry.failure_count, entry.name))

    def _recurring_failures(self, workflow_runs: list[WorkflowRunSnapshot]) -> list[FailureCluster]:
        grouped: dict[str, list[WorkflowRunSnapshot]] = defaultdict(list)
        for run in workflow_runs:
            if run.conclusion not in {"failure", "timed_out", "cancelled", "startup_failure", "action_required"}:
                continue
            signatures = run.failed_jobs or [run.name]
            for job_name in signatures:
                signature = f"{run.name} :: {job_name}" if job_name != run.name else run.name
                grouped[signature].append(run)

        clusters: list[FailureCluster] = []
        for signature, runs in grouped.items():
            if len(runs) < self.recurring_failure_threshold:
                continue
            ordered = sorted(runs, key=lambda run: run.created_dt(), reverse=True)
            clusters.append(
                FailureCluster(
                    signature=signature,
                    count=len(runs),
                    last_seen_at=ordered[0].created_at,
                    run_ids=[run.id for run in ordered],
                )
            )
        return sorted(clusters, key=lambda entry: (-entry.count, entry.signature))

    def _hotspots(self, pull_requests: list[PullRequestSnapshot]) -> list[PathHotspot]:
        counts: Counter[str] = Counter()
        for pr in pull_requests:
            touched = {_hotspot_key(path) for path in pr.changed_files if path}
            counts.update(touched)

        hotspots = [
            PathHotspot(path=path, pull_request_count=count)
            for path, count in counts.most_common(self.hotspot_limit)
            if count >= 2
        ]
        return hotspots

    def _follow_up_issues(self, recurring_failures: list[FailureCluster]) -> list[FollowUpIssue]:
        issues: list[FollowUpIssue] = []
        for cluster in recurring_failures:
            if cluster.count < self.follow_up_issue_threshold:
                continue
            marker = f"<!-- aura-repo-health-followup:{_slugify(cluster.signature)} -->"
            issues.append(
                FollowUpIssue(
                    title=f"[repo-health] Investigate recurring failures in {cluster.signature}",
                    body="\n".join(
                        [
                            marker,
                            "## Recurring Failure Follow-up",
                            "",
                            f"Failure cluster: `{cluster.signature}`",
                            f"Occurrences in recent workflow window: `{cluster.count}`",
                            f"Last seen at: `{cluster.last_seen_at}`",
                            "",
                            "Recent run IDs:",
                            *(f"- `{run_id}`" for run_id in cluster.run_ids[:5]),
                            "",
                            "_Auto-created by nightly repo health automation._",
                        ]
                    ),
                    labels=["repo-health"],
                    marker=marker,
                )
            )
        return issues

    def _summary(
        self,
        stale_pull_requests: list[StalePullRequest],
        flaky_workflows: list[FlakyWorkflow],
        recurring_failures: list[FailureCluster],
        hotspots: list[PathHotspot],
        follow_ups: list[FollowUpIssue],
    ) -> str:
        return (
            f"Detected `{len(stale_pull_requests)}` stale PR(s), "
            f"`{len(flaky_workflows)}` flaky workflow(s), "
            f"`{len(recurring_failures)}` recurring failure cluster(s), and "
            f"`{len(hotspots)}` active hotspot(s). "
            f"`{len(follow_ups)}` follow-up issue(s) are recommended."
        )
