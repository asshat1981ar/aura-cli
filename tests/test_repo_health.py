from __future__ import annotations

from datetime import datetime, timezone

from core.github_automation.repo_health import PullRequestSnapshot, RepoHealthAnalyzer, WorkflowRunSnapshot


def test_repo_health_detects_stale_prs_flakes_failures_and_hotspots() -> None:
    analyzer = RepoHealthAnalyzer(now=datetime(2026, 3, 18, 12, 0, tzinfo=timezone.utc))
    pull_requests = [
        PullRequestSnapshot(
            number=11,
            title="Stale queue fix",
            updated_at="2026-03-07T10:00:00Z",
            changed_files=["core/workflow_engine.py", "tests/test_workflow_engine.py"],
        ),
        PullRequestSnapshot(
            number=12,
            title="Fresh docs update",
            updated_at="2026-03-17T10:00:00Z",
            changed_files=["README.md"],
        ),
        PullRequestSnapshot(
            number=13,
            title="Another core routing change",
            updated_at="2026-03-10T09:00:00Z",
            changed_files=["core/github_automation/policy.py", ".github/workflows/ci.yml"],
        ),
    ]
    workflow_runs = [
        WorkflowRunSnapshot(id=1001, name="CI", conclusion="failure", created_at="2026-03-18T03:00:00Z", failed_jobs=["test (3.12)"]),
        WorkflowRunSnapshot(id=1002, name="CI", conclusion="success", created_at="2026-03-17T03:00:00Z"),
        WorkflowRunSnapshot(id=1003, name="CI", conclusion="failure", created_at="2026-03-16T03:00:00Z", failed_jobs=["test (3.12)"]),
        WorkflowRunSnapshot(id=1004, name="Merge Readiness", conclusion="failure", created_at="2026-03-15T05:00:00Z", failed_jobs=["evaluate"]),
        WorkflowRunSnapshot(id=1005, name="CI", conclusion="failure", created_at="2026-03-14T03:00:00Z", failed_jobs=["test (3.12)"]),
    ]

    report = analyzer.build_report(pull_requests, workflow_runs)

    assert [entry.number for entry in report.stale_pull_requests] == [11, 13]
    assert report.flaky_workflows[0].name == "CI"
    assert report.flaky_workflows[0].failure_count == 3
    assert report.recurring_failures[0].signature == "CI :: test (3.12)"
    assert report.hotspots[0].path == "core"
    assert report.follow_up_issues[0].title.startswith("[repo-health] Investigate recurring failures")


def test_repo_health_markdown_contains_sections() -> None:
    analyzer = RepoHealthAnalyzer(now=datetime(2026, 3, 18, 12, 0, tzinfo=timezone.utc))
    report = analyzer.build_report(
        [
            PullRequestSnapshot(
                number=9,
                title="Stale workflow cleanup",
                updated_at="2026-03-01T10:00:00Z",
                changed_files=[".github/workflows/ci.yml"],
            )
        ],
        [
            WorkflowRunSnapshot(id=2001, name="CI", conclusion="failure", created_at="2026-03-18T03:00:00Z", failed_jobs=["lint"]),
            WorkflowRunSnapshot(id=2002, name="CI", conclusion="failure", created_at="2026-03-17T03:00:00Z", failed_jobs=["lint"]),
            WorkflowRunSnapshot(id=2003, name="CI", conclusion="success", created_at="2026-03-16T03:00:00Z"),
        ],
    )

    markdown = report.render_markdown()

    assert "<!-- aura-repo-health -->" in markdown
    assert "AURA Nightly Repo Health" in markdown
    assert "Stale Pull Requests" in markdown
    assert "Recurring Failure Clusters" in markdown


def test_repo_health_does_not_create_followup_below_threshold() -> None:
    analyzer = RepoHealthAnalyzer(
        now=datetime(2026, 3, 18, 12, 0, tzinfo=timezone.utc),
        follow_up_issue_threshold=4,
    )
    report = analyzer.build_report(
        [],
        [
            WorkflowRunSnapshot(id=3001, name="CI", conclusion="failure", created_at="2026-03-18T03:00:00Z", failed_jobs=["lint"]),
            WorkflowRunSnapshot(id=3002, name="CI", conclusion="failure", created_at="2026-03-17T03:00:00Z", failed_jobs=["lint"]),
            WorkflowRunSnapshot(id=3003, name="CI", conclusion="success", created_at="2026-03-16T03:00:00Z"),
        ],
    )

    assert report.recurring_failures
    assert report.follow_up_issues == []
