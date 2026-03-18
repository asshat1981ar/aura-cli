"""Issue triage helpers for GitHub issue intake."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

KNOWN_PROVIDERS = {"aura", "copilot", "codex", "claude", "gemini", "no preference"}
KNOWN_RISK_LEVELS = {"low", "medium", "high"}

_SECTION_PATTERN = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)


def parse_issue_sections(body: str) -> dict[str, str]:
    """Parse GitHub issue form markdown into ``heading -> body`` sections."""
    if not body:
        return {}

    matches = list(_SECTION_PATTERN.finditer(body))
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        title = _normalize_section_name(match.group(1))
        value = body[start:end].strip()
        if value:
            sections[title] = value
    return sections


@dataclass(slots=True)
class IssueContext:
    """Minimal issue payload used by local planners and workflow scripts."""

    number: int | None
    title: str
    body: str
    labels: list[str] = field(default_factory=list)
    author: str = ""
    is_pull_request: bool = False

    @classmethod
    def from_github_event(cls, payload: dict) -> "IssueContext":
        issue = payload.get("issue", {})
        labels = [label.get("name", "") for label in issue.get("labels", []) if label.get("name")]
        return cls(
            number=issue.get("number"),
            title=issue.get("title", ""),
            body=issue.get("body", "") or "",
            labels=labels,
            author=(issue.get("user") or {}).get("login", ""),
            is_pull_request="pull_request" in issue,
        )

    def to_workflow_inputs(self) -> dict[str, object]:
        return {
            "issue_number": self.number,
            "issue_title": self.title,
            "issue_body": self.body,
            "issue_labels": self.labels,
            "issue_author": self.author,
            "issue_is_pull_request": self.is_pull_request,
        }


@dataclass(slots=True)
class IssueTriageResult:
    """Deterministic triage output for issue planning."""

    issue_type: str
    risk_level: str
    preferred_provider: str
    automation_lane: str
    summary: str
    problem_statement: str
    labels_to_apply: list[str]
    queue_candidate: bool
    queue_goal: str
    sections: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def triage_issue(context: IssueContext) -> IssueTriageResult:
    """Classify and normalize an issue into repo-local planning fields."""
    sections = parse_issue_sections(context.body)
    normalized_labels = {label.lower() for label in context.labels}
    issue_type = _infer_issue_type(context, sections)
    risk_level = _infer_risk_level(context, sections, issue_type)
    preferred_provider = _infer_preferred_provider(context, sections)
    automation_lane = _infer_automation_lane(context, sections, issue_type)
    summary = _first_non_empty(sections.get("summary"), context.title, "Untitled issue")
    problem_statement = _first_non_empty(
        sections.get("problem statement"),
        sections.get("expected result"),
        sections.get("actual result"),
        sections.get("proposed solution"),
        context.body.strip(),
        context.title,
    )
    queue_goal = _first_non_empty(sections.get("goal"), sections.get("summary"), context.title)
    is_agent_task = "agent-task" in normalized_labels or issue_type == "agent-task"
    queue_candidate = is_agent_task or "aura-goal" in normalized_labels or preferred_provider == "aura"

    labels = []
    if is_agent_task:
        labels.append("agent-task")
    if issue_type == "feature":
        labels.append("enhancement")
    elif issue_type == "bug":
        labels.append("bug")
    labels.append(f"risk:{risk_level}")
    if risk_level == "high":
        labels.append("needs-human-review")
    if preferred_provider in {"aura", "copilot", "codex", "claude", "gemini"}:
        labels.append(f"provider:{preferred_provider}")

    return IssueTriageResult(
        issue_type=issue_type,
        risk_level=risk_level,
        preferred_provider=preferred_provider,
        automation_lane=automation_lane,
        summary=summary,
        problem_statement=problem_statement,
        labels_to_apply=_dedupe(labels),
        queue_candidate=queue_candidate,
        queue_goal=queue_goal,
        sections=sections,
    )


def _infer_issue_type(context: IssueContext, sections: dict[str, str]) -> str:
    title = context.title.lower()
    labels = {label.lower() for label in context.labels}
    if "bug" in labels or title.startswith("[bug]") or "reproduction steps" in sections:
        return "bug"
    if "enhancement" in labels or title.startswith("[feature]") or "acceptance criteria" in sections:
        return "feature"
    if "agent-task" in labels or title.startswith("[agent-task]"):
        return "agent-task"
    return "general"


def _infer_risk_level(context: IssueContext, sections: dict[str, str], issue_type: str) -> str:
    for label in context.labels:
        if label.startswith("risk:"):
            value = label.split(":", 1)[1].strip().lower()
            if value in KNOWN_RISK_LEVELS:
                return value
    for key in ("risk level", "severity"):
        value = _normalize_choice(sections.get(key, ""))
        if value in KNOWN_RISK_LEVELS:
            return value
    if issue_type in {"bug", "agent-task"}:
        return "medium"
    return "low"


def _infer_preferred_provider(context: IssueContext, sections: dict[str, str]) -> str:
    for label in context.labels:
        if label.startswith("provider:"):
            provider = label.split(":", 1)[1].strip().lower()
            if provider in KNOWN_PROVIDERS:
                return provider
    provider = _normalize_choice(sections.get("preferred provider", ""))
    if provider in KNOWN_PROVIDERS:
        return provider
    return "no preference"


def _infer_automation_lane(context: IssueContext, sections: dict[str, str], issue_type: str) -> str:
    explicit = _normalize_choice(sections.get("best-fit automation lane", ""))
    if explicit:
        return explicit

    text = " ".join(
        filter(
            None,
            [context.title.lower(), context.body.lower(), issue_type],
        )
    )
    if any(phrase in text for phrase in ("branch protection", "auto-merge", "merge queue")):
        return "merge governance"
    if re.search(r"\bmerge\b", text) or re.search(r"\bruleset\b", text):
        return "merge governance"
    if "pull request" in text or re.search(r"\breview\b", text) or re.search(r"\bpr\b", text):
        return "code review"
    if any(re.search(pattern, text) for pattern in (r"\bagent\b", r"\bautomation\b", r"\bqueue\b", r"\bworkflow\b")):
        return "coding agent"
    return "issue intake"


def _normalize_section_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _normalize_choice(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""
    return cleaned.splitlines()[0].strip().lower()


def _first_non_empty(*values: str) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""


def _dedupe(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return seen
