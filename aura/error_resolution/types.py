"""Type definitions for error resolution."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ResolutionConfidence(Enum):
    """Confidence level for resolution suggestions."""
    HIGH = "high"       # Verified fix, safe to auto-apply
    MEDIUM = "medium"   # Likely correct, user confirmation recommended
    LOW = "low"         # Suggestion, manual review required


@dataclass(frozen=True)
class ResolutionResult:
    """Result of an error resolution attempt.
    
    Attributes:
        original_error: String representation of the original error
        explanation: Human-readable explanation of the error
        suggested_fix: Suggested command or fix
        confidence: Confidence level in the suggestion
        auto_applied: Whether the fix was automatically applied
        cache_hit: Whether result came from cache
        provider: Which provider generated the suggestion
        execution_time_ms: Time taken to resolve in milliseconds
    """
    original_error: str
    explanation: str
    suggested_fix: str
    confidence: ResolutionConfidence
    auto_applied: bool
    cache_hit: bool
    provider: str       # "openai", "ollama", "cache", "known_fix"
    execution_time_ms: int


@dataclass(frozen=True)
class ErrorContext:
    """Context information for error resolution.
    
    Attributes:
        command: The command that was being executed
        working_dir: Current working directory
        environment: Relevant environment variables
        recent_commands: Recently executed commands
        system_info: OS, Python version, etc.
    """
    command: str = ""
    working_dir: str = ""
    environment: dict = None
    recent_commands: list = None
    system_info: dict = None
    
    def __post_init__(self):
        # Handle mutable defaults
        if self.environment is None:
            object.__setattr__(self, 'environment', {})
        if self.recent_commands is None:
            object.__setattr__(self, 'recent_commands', [])
        if self.system_info is None:
            object.__setattr__(self, 'system_info', {})


@dataclass(frozen=True)
class CacheKey:
    """Key for caching resolution results.
    
    Combines error type, message, and relevant context for cache lookup.
    """
    error_type: str
    error_message: str
    command_hash: str = ""  # Hash of relevant command context
    
    def __str__(self) -> str:
        return f"{self.error_type}:{self.error_message}:{self.command_hash}"
