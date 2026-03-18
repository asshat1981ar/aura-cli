from __future__ import annotations

from core.github_automation.issue_planner import IssuePlanner
from core.github_automation.issue_triage import IssueContext, triage_issue


def test_issue_planner_recommends_merge_governance_files(project_root) -> None:
    context = IssueContext(
        number=11,
        title="[feature] Add merge queue readiness checks",
        body="### Problem statement\nThe repo needs merge queue support and branch policy alignment.",
        labels=["enhancement"],
    )
    triage = triage_issue(context)
    plan = IssuePlanner(project_root=project_root).plan(context, triage)
    assert plan.automation_lane == "merge governance"
    assert any(path.startswith(".github/") for path in plan.candidate_files)
    assert any("pytest" in command for command in plan.suggested_tests)


def test_issue_plan_markdown_contains_marker(project_root) -> None:
    context = IssueContext(
        number=5,
        title="[agent-task] Route /plan comments",
        body="### Goal\nAdd slash command routing for issue comments.",
        labels=["agent-task"],
    )
    triage = triage_issue(context)
    plan = IssuePlanner(project_root=project_root).plan(context, triage)
    markdown = plan.render_markdown()
    assert "<!-- aura-issue-plan -->" in markdown
    assert "AURA Issue Intake Plan" in markdown
