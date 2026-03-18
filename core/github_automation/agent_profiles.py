"""Role-specific agent profiles used by GitHub dispatch planning."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class AgentProfile:
    """Metadata describing a role-specific coding agent lane."""

    name: str
    role: str
    description: str
    allowed_providers: list[str]
    branch_only: bool = True
    requires_pr: bool = True
    direct_main_allowed: bool = False
    focus_areas: list[str] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def default_agent_profiles() -> dict[str, AgentProfile]:
    """Return the repo's supported role-specific agent profiles."""
    return {
        "pr-reviewer": AgentProfile(
            name="pr-reviewer",
            role="pull-request-review",
            description="Review a pull request, summarize risk, and flag missing tests or rollout gaps.",
            allowed_providers=["copilot", "claude", "gemini", "aura"],
            focus_areas=["review", "risk", "tests", "regressions"],
        ),
        "issue-planner": AgentProfile(
            name="issue-planner",
            role="issue-planning",
            description="Turn an issue into an implementation-ready plan with likely files, tests, and risks.",
            allowed_providers=["aura", "copilot", "codex"],
            focus_areas=["planning", "triage", "task-shaping"],
        ),
        "bugfix": AgentProfile(
            name="bugfix",
            role="implementation",
            description="Investigate a bug, propose a fix path, and prepare branch-oriented implementation metadata.",
            allowed_providers=["codex", "claude", "copilot", "aura"],
            focus_areas=["debugging", "implementation", "tests"],
        ),
        "agentic-workflows-dev": AgentProfile(
            name="agentic-workflows-dev",
            role="workflow-architecture",
            description="Design and debug agentic workflows, routing, retry behavior, and tool-use loops.",
            allowed_providers=["aura", "claude", "codex"],
            focus_areas=["agent loops", "tool orchestration", "state management"],
        ),
    }
