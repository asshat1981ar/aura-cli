from __future__ import annotations

from core.github_automation import AgentDispatchPlanner, IssueContext, default_agent_profiles, slugify


def test_default_agent_profiles_are_branch_only() -> None:
    profiles = default_agent_profiles()
    assert {"pr-reviewer", "issue-planner", "bugfix", "agentic-workflows-dev"} <= set(profiles)
    assert all(profile.branch_only for profile in profiles.values())
    assert all(profile.direct_main_allowed is False for profile in profiles.values())


def test_bug_issue_prefers_bugfix_profile_and_codex(project_root) -> None:
    context = IssueContext(
        number=42,
        title="[bug] Fix queue crash on retries",
        body="### Reproduction steps\n1. Trigger the retry path.\n2. Observe the crash.",
        labels=["agent-task", "bug"],
    )
    dispatch = AgentDispatchPlanner(project_root=str(project_root)).dispatch(context)
    assert dispatch.profile_name == "bugfix"
    assert dispatch.provider in {"codex", "claude"}
    assert dispatch.branch_name.startswith("agent/")
    assert dispatch.direct_main_allowed is False


def test_review_lane_prefers_pr_reviewer_profile(project_root) -> None:
    context = IssueContext(
        number=77,
        title="[feature] Review flaky PR automation output",
        body="Need a review-focused pass on pull request summaries and risk notes.",
        labels=["enhancement"],
    )
    dispatch = AgentDispatchPlanner(project_root=str(project_root)).dispatch(context)
    assert dispatch.profile_name == "pr-reviewer"
    assert dispatch.provider in {"copilot", "claude", "gemini"}
    assert "provider:" in " ".join(dispatch.labels_to_apply)


def test_slugify_produces_branch_safe_slug() -> None:
    assert slugify("Fix / weird: branch naming??") == "fix-weird-branch-naming"


def test_dispatch_keeps_only_final_provider_label(project_root) -> None:
    context = IssueContext(
        number=51,
        title="[bug] Fix provider label drift",
        body="### Reproduction steps\n1. Trigger dispatch.\n2. Inspect labels.",
        labels=["agent-task", "bug", "provider:copilot"],
    )
    dispatch = AgentDispatchPlanner(project_root=str(project_root)).dispatch(
        context,
        preferred_provider="codex",
    )
    provider_labels = [label for label in dispatch.labels_to_apply if label.startswith("provider:")]
    assert provider_labels == ["provider:codex"]
