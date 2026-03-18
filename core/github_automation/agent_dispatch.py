"""Dispatch planning for coding-agent execution lanes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

from core.github_automation.agent_profiles import AgentProfile, default_agent_profiles
from core.github_automation.issue_planner import IssuePlan, IssuePlanner
from core.github_automation.issue_triage import IssueContext, IssueTriageResult, triage_issue


@dataclass(slots=True)
class AgentDispatchPlan:
    """Repo-owned dispatch metadata for a coding-agent task."""

    provider: str
    profile_name: str
    branch_name: str
    base_branch: str
    requires_pr: bool
    branch_only: bool
    direct_main_allowed: bool
    human_review_required: bool
    labels_to_apply: list[str]
    queue_goal: str
    dispatch_reason: str
    artifact_names: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def render_markdown(self, *, marker: str = "<!-- aura-agent-dispatch -->") -> str:
        lines = [
            marker,
            "## AURA Agent Dispatch",
            "",
            f"**Provider:** `{self.provider}`",
            f"**Profile:** `{self.profile_name}`",
            f"**Base branch:** `{self.base_branch}`",
            f"**Suggested branch:** `{self.branch_name}`",
            f"**Requires PR:** `{'yes' if self.requires_pr else 'no'}`",
            f"**Direct writes to main allowed:** `{'yes' if self.direct_main_allowed else 'no'}`",
            f"**Human review required:** `{'yes' if self.human_review_required else 'no'}`",
            "",
            "### Dispatch Reason",
            "",
            self.dispatch_reason,
            "",
            "### Labels",
            "",
        ]
        lines.extend(f"- `{label}`" for label in self.labels_to_apply)
        lines.extend(["", "### Expected Artifacts", ""])
        lines.extend(f"- `{artifact}`" for artifact in self.artifact_names)
        if self.notes:
            lines.extend(["", "### Notes", ""])
            lines.extend(f"- {note}" for note in self.notes)
        lines.extend(["", f"_Queue goal: {self.queue_goal}_", ""])
        return "\n".join(lines)


class AgentDispatchPlanner:
    """Choose provider, profile, and branch policy for coding-agent tasks."""

    def __init__(self, project_root: str = "."):
        self.issue_planner = IssuePlanner(project_root=project_root)
        self.profiles = default_agent_profiles()

    def dispatch(
        self,
        context: IssueContext,
        *,
        preferred_provider: str | None = None,
        requested_profile: str | None = None,
        base_branch: str = "main",
    ) -> AgentDispatchPlan:
        triage = triage_issue(context)
        plan = self.issue_planner.plan(context, triage)
        profile = self._select_profile(context, triage, requested_profile)
        provider = self._select_provider(plan, triage, profile, preferred_provider)
        branch_name = self._branch_name(context, provider, profile.name)
        human_review_required = triage.risk_level == "high" or profile.name == "agentic-workflows-dev"
        inherited_labels = [
            label
            for label in plan.labels_to_apply
            if not label.startswith("provider:") and not label.startswith("profile:")
        ]
        labels = _dedupe(
            inherited_labels
            + [
                "agent-task",
                f"provider:{provider}",
                f"profile:{profile.name}",
            ]
            + (["needs-human-review"] if human_review_required else [])
        )
        notes = [
            "All coding-agent work must land on a branch and through a pull request.",
            "Generated patches should be attached as artifacts for auditability.",
        ]
        if human_review_required:
            notes.append("This dispatch requires a human reviewer before merge.")
        if provider == "aura":
            notes.append("AURA can queue the task, but should still produce branch-oriented changes only.")

        return AgentDispatchPlan(
            provider=provider,
            profile_name=profile.name,
            branch_name=branch_name,
            base_branch=base_branch,
            requires_pr=profile.requires_pr,
            branch_only=profile.branch_only,
            direct_main_allowed=profile.direct_main_allowed,
            human_review_required=human_review_required,
            labels_to_apply=labels,
            queue_goal=plan.queue_goal,
            dispatch_reason=self._dispatch_reason(triage, plan, profile, provider),
            artifact_names=[
                "agent-dispatch-summary.json",
                "agent-dispatch-comment.md",
                "patch-metadata.json",
            ],
            notes=notes,
        )

    def _select_profile(
        self,
        context: IssueContext,
        triage: IssueTriageResult,
        requested_profile: str | None,
    ) -> AgentProfile:
        if requested_profile and requested_profile in self.profiles:
            return self.profiles[requested_profile]

        text = f"{context.title}\n{context.body}".lower()
        if any(token in text for token in ("workflow", "orchestration", "agent loop", "tool-use")):
            return self.profiles["agentic-workflows-dev"]
        if triage.issue_type == "bug":
            return self.profiles["bugfix"]
        if triage.automation_lane == "code review" or triage.preferred_provider == "gemini":
            return self.profiles["pr-reviewer"]
        return self.profiles["issue-planner"]

    def _select_provider(
        self,
        plan: IssuePlan,
        triage: IssueTriageResult,
        profile: AgentProfile,
        preferred_provider: str | None,
    ) -> str:
        if preferred_provider and preferred_provider in profile.allowed_providers:
            return preferred_provider
        if triage.preferred_provider in profile.allowed_providers:
            return triage.preferred_provider
        if profile.name == "pr-reviewer":
            if triage.risk_level == "high":
                return "claude"
            if triage.automation_lane == "code review":
                return "copilot"
            return "gemini"
        if profile.name == "bugfix":
            if triage.risk_level == "high":
                return "claude"
            return "codex"
        if profile.name == "agentic-workflows-dev":
            return "aura"
        if plan.automation_lane == "merge governance":
            return "aura"
        return "aura"

    def _branch_name(self, context: IssueContext, provider: str, profile_name: str) -> str:
        slug = slugify(context.title or context.body or "agent-task")
        prefix = f"agent/{provider}/{profile_name}"
        if context.number is not None:
            return f"{prefix}/issue-{context.number}-{slug}"
        return f"{prefix}/{slug}"

    def _dispatch_reason(
        self,
        triage: IssueTriageResult,
        plan: IssuePlan,
        profile: AgentProfile,
        provider: str,
    ) -> str:
        return (
            f"Selected `{provider}` with the `{profile.name}` profile for a "
            f"`{triage.issue_type}` issue in the `{plan.automation_lane}` lane "
            f"at `{triage.risk_level}` risk."
        )


def slugify(value: str) -> str:
    """Return a git-branch-safe slug."""
    collapsed = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    if not collapsed:
        return "task"
    return collapsed[:48]


def _dedupe(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return seen
