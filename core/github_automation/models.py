"""Normalized data models for GitHub review automation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Severity = Literal["critical", "high", "medium", "low", "info"]
RecommendedAction = Literal["approve", "comment", "request_changes", "escalate"]

_SEVERITY_RANK = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
}


@dataclass(slots=True)
class ReviewFinding:
    """A single normalized review finding."""

    severity: Severity
    path: str
    line: int | None
    title: str
    detail: str
    confidence: float
    category: str

    def sort_key(self) -> tuple[int, str, int, str]:
        return (
            _SEVERITY_RANK.get(self.severity, 99),
            self.path,
            self.line or 0,
            self.title.lower(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProviderReview:
    """Normalized output from a single provider or local review lane."""

    provider: str
    summary: str
    findings: list[ReviewFinding] = field(default_factory=list)
    recommended_action: RecommendedAction = "comment"
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["findings"] = [finding.to_dict() for finding in self.findings]
        return payload


@dataclass(slots=True)
class PolicyDecision:
    """Policy verdict after combining findings and repo risk rules."""

    recommended_action: RecommendedAction
    human_review_required: bool = False
    blocked_reasons: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SynthesisResult:
    """Single synthesized view posted back to GitHub."""

    summary: str
    recommended_action: RecommendedAction
    providers_consulted: list[str]
    finding_counts: dict[str, int]
    labels: list[str]
    human_review_required: bool
    blocked_reasons: list[str]
    comment_markdown: str
    findings: list[ReviewFinding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "recommended_action": self.recommended_action,
            "providers_consulted": self.providers_consulted,
            "finding_counts": self.finding_counts,
            "labels": self.labels,
            "human_review_required": self.human_review_required,
            "blocked_reasons": self.blocked_reasons,
            "comment_markdown": self.comment_markdown,
            "findings": [finding.to_dict() for finding in self.findings],
        }
