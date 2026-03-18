"""Slash-command parsing and routing for issue_comment workflows."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from core.github_automation.issue_planner import IssuePlanner
from core.github_automation.issue_triage import IssueContext, triage_issue

SUPPORTED_REVIEW_PROVIDERS = {"aura", "copilot", "codex", "claude", "gemini"}


@dataclass(slots=True)
class CommandRequest:
    """Parsed issue comment command."""

    command: str
    args: list[str] = field(default_factory=list)
    raw_line: str = ""


@dataclass(slots=True)
class CommandResult:
    """Outcome of slash-command routing."""

    recognized: bool
    action: str
    labels_to_add: list[str] = field(default_factory=list)
    queue_goal: str = ""
    should_queue: bool = False
    requested_provider: str = ""
    comment_markdown: str = ""
    artifacts: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_slash_command(comment_body: str) -> CommandRequest | None:
    """Return the first recognized slash-command request in *comment_body*."""
    for raw_line in comment_body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not line.startswith("/"):
            return None
        parts = line[1:].split()
        if not parts:
            return None
        return CommandRequest(command=parts[0].lower(), args=[part.lower() for part in parts[1:]], raw_line=line)
    return None


class CommandRouter:
    """Routes issue comment commands to deterministic local handlers."""

    def __init__(self, project_root: str = "."):
        self.planner = IssuePlanner(project_root=project_root)

    def handle(self, context: IssueContext, comment_body: str, *, response_marker: str) -> CommandResult:
        request = parse_slash_command(comment_body)
        if request is None:
            return CommandResult(recognized=False, action="ignore")

        triage = triage_issue(context)
        if request.command == "plan":
            plan = self.planner.plan(context, triage)
            return CommandResult(
                recognized=True,
                action="plan",
                labels_to_add=plan.labels_to_apply,
                requested_provider=plan.recommended_provider,
                comment_markdown=plan.render_markdown(marker=response_marker),
                artifacts={
                    "triage": triage.to_dict(),
                    "plan": plan.to_dict(),
                    "command": request.raw_line,
                },
            )

        if request.command == "queue":
            provider = request.args[0] if request.args else "aura"
            if provider != "aura":
                return _unsupported_provider_result(provider, response_marker)
            return CommandResult(
                recognized=True,
                action="queue",
                labels_to_add=_dedupe(["agent-task", "provider:aura"]),
                queue_goal=triage.queue_goal,
                should_queue=True,
                requested_provider="aura",
                comment_markdown="\n".join(
                    [
                        response_marker,
                        "## AURA Command Response",
                        "",
                        f"Queued goal candidate for AURA: `{triage.queue_goal}`",
                        "",
                        "This command only adds the goal to the AURA queue when `AURA_AUTO_QUEUE_ISSUES=1` is enabled for the workflow.",
                        "",
                    ]
                ),
                artifacts={
                    "command": request.raw_line,
                    "queue_goal": triage.queue_goal,
                },
            )

        if request.command == "review":
            provider = request.args[0] if request.args else "aura"
            if provider not in SUPPORTED_REVIEW_PROVIDERS:
                return _unsupported_provider_result(provider, response_marker)

            if context.is_pull_request:
                body = [
                    response_marker,
                    "## AURA Command Response",
                    "",
                    f"Recorded provider hint `{provider}` for this pull request.",
                    "",
                    "The unified PR review orchestrator remains the source of truth, but this label can guide the next manual or automated review pass.",
                    "",
                ]
            else:
                plan = self.planner.plan(context, triage)
                body = [
                    response_marker,
                    "## AURA Command Response",
                    "",
                    f"Recorded provider hint `{provider}` for this issue.",
                    "",
                    f"Recommended next lane: `{plan.automation_lane}`",
                    f"Queue goal candidate: `{plan.queue_goal}`",
                    "",
                ]
            return CommandResult(
                recognized=True,
                action="review",
                labels_to_add=_dedupe([f"provider:{provider}", "agent-review"]),
                requested_provider=provider,
                comment_markdown="\n".join(body),
                artifacts={
                    "command": request.raw_line,
                    "requested_provider": provider,
                    "issue_is_pull_request": context.is_pull_request,
                },
            )

        return CommandResult(
            recognized=False,
            action="unsupported",
            comment_markdown="",
        )


def _unsupported_provider_result(provider: str, marker: str) -> CommandResult:
    return CommandResult(
        recognized=True,
        action="unsupported",
        comment_markdown="\n".join(
            [
                marker,
                "## AURA Command Response",
                "",
                f"`{provider}` is not a supported provider hint here.",
                "",
                f"Supported providers: {', '.join(sorted(SUPPORTED_REVIEW_PROVIDERS))}.",
                "",
            ]
        ),
        artifacts={"requested_provider": provider},
    )


def _dedupe(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return seen
