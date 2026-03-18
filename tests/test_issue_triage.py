from __future__ import annotations

from core.github_automation.issue_triage import IssueContext, parse_issue_sections, triage_issue


def test_parse_issue_sections_extracts_form_fields() -> None:
    body = """
### Summary
CI fails on merge queue.

### Preferred provider
aura
""".strip()
    sections = parse_issue_sections(body)
    assert sections["summary"] == "CI fails on merge queue."
    assert sections["preferred provider"] == "aura"


def test_triage_agent_task_prefers_requested_provider() -> None:
    context = IssueContext(
        number=12,
        title="[agent-task] Add issue intake planning",
        body="### Goal\nCreate a new issue intake workflow.\n\n### Preferred provider\naura",
        labels=["agent-task"],
    )
    triage = triage_issue(context)
    assert triage.issue_type == "agent-task"
    assert triage.preferred_provider == "aura"
    assert triage.queue_candidate is True
    assert "provider:aura" in triage.labels_to_apply


def test_triage_bug_defaults_to_medium_risk() -> None:
    context = IssueContext(
        number=8,
        title="[bug] Queue processing fails",
        body="### Reproduction steps\n1. Run the queue.\n2. Observe the crash.",
        labels=["bug"],
    )
    triage = triage_issue(context)
    assert triage.issue_type == "bug"
    assert triage.risk_level == "medium"


def test_triage_bug_reproduction_does_not_false_positive_code_review() -> None:
    context = IssueContext(
        number=9,
        title="[bug] Fix queue crash on retries",
        body="### Reproduction steps\n1. Trigger the retry path.\n2. Observe the crash.",
        labels=["bug"],
    )
    triage = triage_issue(context)
    assert triage.issue_type == "bug"
    assert triage.automation_lane != "code review"


def test_triage_agent_task_bug_stays_queue_candidate() -> None:
    context = IssueContext(
        number=10,
        title="[bug] Fix agent queue retries",
        body="### Reproduction steps\n1. Trigger retries.\n2. Observe the crash.",
        labels=["agent-task", "bug"],
    )
    triage = triage_issue(context)
    assert triage.issue_type == "bug"
    assert triage.queue_candidate is True
    assert "agent-task" in triage.labels_to_apply
    assert "bug" in triage.labels_to_apply
