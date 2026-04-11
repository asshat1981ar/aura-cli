"""Shared types for the error resolution subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResolutionConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class CacheKey:
    """Deterministic cache key for an error + context pair."""

    error_type: str
    error_message: str
    command_hash: str

    def __str__(self) -> str:
        return f"{self.error_type}:{self.error_message}:{self.command_hash}"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CacheKey):
            return NotImplemented
        return str(self) == str(other)


@dataclass
class ResolutionResult:
    """The outcome of a single error resolution attempt."""

    original_error: str
    explanation: str
    suggested_fix: str
    confidence: ResolutionConfidence
    auto_applied: bool
    cache_hit: bool
    provider: str
    execution_time_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_error": self.original_error,
            "explanation": self.explanation,
            "suggested_fix": self.suggested_fix,
            "confidence": self.confidence.value,
            "auto_applied": self.auto_applied,
            "cache_hit": self.cache_hit,
            "provider": self.provider,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResolutionResult":
        return cls(
            original_error=data["original_error"],
            explanation=data["explanation"],
            suggested_fix=data["suggested_fix"],
            confidence=ResolutionConfidence(data["confidence"]),
            auto_applied=data["auto_applied"],
            cache_hit=data["cache_hit"],
            provider=data["provider"],
            execution_time_ms=data["execution_time_ms"],
            metadata=data.get("metadata", {}),
        )
