"""Structured issue planning for GitHub issue intake."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
import re

from core.github_automation.issue_triage import IssueContext, IssueTriageResult

_SEARCH_ROOTS = (".github", "aura_cli", "agents", "core", "docs", "scripts", "tests", "tools")


@dataclass(slots=True)
class IssuePlan:
    """Implementation-ready plan generated from a triaged issue."""

    summary: str
    issue_type: str
    automation_lane: str
    risk_level: str
    recommended_provider: str
    problem_statement: str
    affected_areas: list[str]
    candidate_files: list[str]
    suggested_tests: list[str]
    implementation_steps: list[str]
    notes: list[str]
    queue_goal: str
    labels_to_apply: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def render_markdown(self, *, marker: str = "<!-- aura-issue-plan -->") -> str:
        lines = [
            marker,
            "## AURA Issue Intake Plan",
            "",
            f"**Summary:** {self.summary}",
            f"**Issue type:** `{self.issue_type}`",
            f"**Automation lane:** `{self.automation_lane}`",
            f"**Risk:** `{self.risk_level}`",
            f"**Recommended provider:** `{self.recommended_provider}`",
            "",
            "### Problem Statement",
            "",
            self.problem_statement,
            "",
            "### Candidate Files",
            "",
        ]
        if self.candidate_files:
            lines.extend(f"- `{path}`" for path in self.candidate_files)
        else:
            lines.append("- No high-confidence candidate files found yet.")
        lines.extend(["", "### Suggested Steps", ""])
        lines.extend(f"{index}. {step}" for index, step in enumerate(self.implementation_steps, start=1))
        lines.extend(["", "### Suggested Validation", ""])
        lines.extend(f"- `{command}`" for command in self.suggested_tests)
        if self.notes:
            lines.extend(["", "### Notes", ""])
            lines.extend(f"- {note}" for note in self.notes)
        lines.extend(["", f"_Queue goal: {self.queue_goal}_", ""])
        return "\n".join(lines)


class IssuePlanner:
    """Builds a deterministic implementation plan for an issue."""

    def __init__(self, project_root: str | Path = "."):
        self.project_root = Path(project_root)

    def plan(self, context: IssueContext, triage: IssueTriageResult) -> IssuePlan:
        candidate_files = self._candidate_files(context, triage)
        affected_areas = self._affected_areas(candidate_files)
        suggested_tests = self._suggested_tests(candidate_files, triage)
        implementation_steps = self._implementation_steps(candidate_files, triage)
        notes = self._notes(triage, candidate_files)
        recommended_provider = self._recommended_provider(triage)
        labels_to_apply = triage.labels_to_apply + (
            [f"provider:{recommended_provider}"]
            if recommended_provider in {"aura", "copilot", "codex", "claude", "gemini"}
            else []
        )
        return IssuePlan(
            summary=triage.summary,
            issue_type=triage.issue_type,
            automation_lane=triage.automation_lane,
            risk_level=triage.risk_level,
            recommended_provider=recommended_provider,
            problem_statement=triage.problem_statement,
            affected_areas=affected_areas,
            candidate_files=candidate_files,
            suggested_tests=suggested_tests,
            implementation_steps=implementation_steps,
            notes=notes,
            queue_goal=triage.queue_goal,
            labels_to_apply=_dedupe(labels_to_apply),
        )

    def _candidate_files(self, context: IssueContext, triage: IssueTriageResult) -> list[str]:
        keywords = _keywords(context.title, triage.problem_statement, " ".join(triage.sections.values()))
        fallback = self._fallback_candidates(triage)
        scored: list[tuple[int, str]] = []
        for root in _SEARCH_ROOTS:
            base = self.project_root / root
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(self.project_root).as_posix()
                score = _score_path(rel, keywords)
                if score > 0:
                    scored.append((score, rel))
        scored.sort(key=lambda item: (-item[0], item[1]))

        candidates = [path for _, path in scored[:5]]
        if not candidates:
            return fallback

        for path in fallback:
            if path not in candidates and len(candidates) < 5:
                candidates.append(path)
        return candidates[:5]

    def _fallback_candidates(self, triage: IssueTriageResult) -> list[str]:
        if triage.automation_lane == "merge governance":
            return [".github/workflows/ci.yml", ".github/workflows/pr-review-orchestrator.yml"]
        if triage.automation_lane == "code review":
            return ["core/github_automation/policy.py", "core/github_automation/review_synthesizer.py"]
        if triage.issue_type == "agent-task":
            return ["core/workflow_engine.py", ".github/workflows/issue-intake.yml"]
        if triage.issue_type == "bug":
            return ["core/", "tests/"]
        return ["README.md", "docs/INTEGRATION_MAP.md"]

    def _affected_areas(self, candidate_files: list[str]) -> list[str]:
        areas = []
        for path in candidate_files:
            first = path.split("/", 1)[0]
            if first not in areas:
                areas.append(first)
        return areas

    def _suggested_tests(self, candidate_files: list[str], triage: IssueTriageResult) -> list[str]:
        commands = ["python -m pytest -q tests/ -k \"not integration\""]
        if any(path.startswith("core/") or path.startswith("aura_cli/") for path in candidate_files):
            commands.append("python -m pytest -q tests/test_workflow_engine.py")
        if any(path.startswith(".github/") for path in candidate_files):
            commands.append("python -m pytest -q tests/test_github_provider_router.py tests/test_github_review_policy.py tests/test_github_review_synthesizer.py")
        if triage.risk_level != "low":
            commands.append("ruff check .")
        return _dedupe(commands)

    def _implementation_steps(self, candidate_files: list[str], triage: IssueTriageResult) -> list[str]:
        steps = [
            "Confirm the issue scope and reproduce the current behavior or gap.",
            "Update the most likely implementation files and keep the change set focused.",
            "Add or adjust tests and snapshots for the affected behavior.",
            "Run the suggested validation commands and capture any follow-up work.",
        ]
        if candidate_files:
            steps.insert(1, f"Start by inspecting `{candidate_files[0]}` and adjacent tests.")
        if triage.automation_lane == "merge governance":
            steps.insert(2, "Validate branch protection, check-run naming, and merge-queue compatibility.")
        return steps

    def _notes(self, triage: IssueTriageResult, candidate_files: list[str]) -> list[str]:
        notes = []
        if triage.risk_level == "high":
            notes.append("High-risk issues should keep a human reviewer in the loop before merge.")
        if not any(path.startswith("tests/") for path in candidate_files):
            notes.append("No direct test file match was found; expect to add or update tests explicitly.")
        if triage.preferred_provider != "no preference":
            notes.append(f"Requester preferred `{triage.preferred_provider}` for this issue.")
        return notes

    def _recommended_provider(self, triage: IssueTriageResult) -> str:
        if triage.preferred_provider != "no preference":
            return triage.preferred_provider
        if triage.automation_lane == "code review":
            return "copilot"
        if triage.automation_lane == "merge governance":
            return "aura"
        if triage.issue_type == "agent-task":
            return "aura"
        if triage.issue_type == "bug":
            return "codex"
        if triage.issue_type == "feature":
            return "codex"
        return "aura"


def _keywords(*texts: str) -> list[str]:
    words = set()
    for text in texts:
        for token in re.split(r"[^a-zA-Z0-9_]+", text or ""):
            token = token.strip().lower()
            if len(token) >= 3:
                words.add(token)
    return sorted(words)


def _score_path(path: str, keywords: list[str]) -> int:
    parts = {part.lower() for part in PurePosixPath(path).parts}
    lowered = path.lower()
    score = 0
    for keyword in keywords:
        if keyword in parts:
            score += 3
        elif keyword in lowered:
            score += 1
    return score


def _dedupe(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return seen
