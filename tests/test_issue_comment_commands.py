from __future__ import annotations

from core.github_automation.command_router import CommandRouter, parse_slash_command
from core.github_automation.issue_triage import IssueContext


def test_parse_slash_command_reads_first_command_line() -> None:
    request = parse_slash_command("/review gemini\nplease check this")
    assert request is not None
    assert request.command == "review"
    assert request.args == ["gemini"]


def test_plan_command_returns_plan_comment(project_root) -> None:
    context = IssueContext(
        number=22,
        title="[feature] Add structured issue planning",
        body="### Problem statement\nIssue intake should produce structured plans.",
        labels=["enhancement"],
    )
    result = CommandRouter(project_root=str(project_root)).handle(
        context,
        "/plan",
        response_marker="<!-- aura-command-response:1 -->",
    )
    assert result.recognized is True
    assert result.action == "plan"
    assert "AURA Issue Intake Plan" in result.comment_markdown


def test_queue_command_sets_goal(project_root) -> None:
    context = IssueContext(
        number=23,
        title="[agent-task] Queue this issue",
        body="### Goal\nQueue this goal in AURA.",
        labels=["agent-task"],
    )
    result = CommandRouter(project_root=str(project_root)).handle(
        context,
        "/queue aura",
        response_marker="<!-- aura-command-response:2 -->",
    )
    assert result.recognized is True
    assert result.should_queue is True
    assert result.queue_goal == "Queue this goal in AURA."
