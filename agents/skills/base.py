"""Base class for all AURA skill modules."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time

from core.logging_utils import log_json


@dataclass
class SkillResult(dict):
    """Typed result envelope returned by every skill execution."""

    status: str  # "success" | "error" | "partial"
    skill_name: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0.0
    _initialized: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        self._refresh_mapping()
        object.__setattr__(self, "_initialized", True)

    @property
    def ok(self) -> bool:
        return self.status != "error"

    def as_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (backward-compatible with old callers)."""
        return dict(self)

    def __setattr__(self, key: str, value: Any) -> None:
        object.__setattr__(self, key, value)
        if key != "_initialized" and getattr(self, "_initialized", False):
            self._refresh_mapping()

    def _refresh_mapping(self) -> None:
        """Mirror attribute fields into the legacy dict-shaped result payload."""
        dict.clear(self)
        dict.update(self, {"status": self.status, "skill": self.skill_name, **self.data})
        if self.error:
            dict.__setitem__(self, "error", self.error)


class SkillBase(ABC):
    """
    Abstract base for all skill modules.

    Subclasses must set `name` and implement `_run()`.
    All exceptions must be caught internally; run() always returns a SkillResult.
    """

    name: str = "base_skill"

    def __init__(self, brain=None, model=None):
        self.brain = brain
        self.model = model

    def run(self, input_data: Dict[str, Any]) -> SkillResult:
        """Execute the skill. Returns a SkillResult with timing."""
        start = time.monotonic()
        try:
            raw = self._run(input_data)
            elapsed = (time.monotonic() - start) * 1000
            # Respect the skill's own status field if it sets one
            status = raw.pop("status", "success") if isinstance(raw, dict) else "success"
            return SkillResult(
                status=status,
                skill_name=self.name,
                data=raw if isinstance(raw, dict) else {},
                duration_ms=elapsed,
            )
        except Exception as exc:  # pylint: disable=broad-except
            elapsed = (time.monotonic() - start) * 1000
            log_json("ERROR", f"{self.name}_failed", details={"error": str(exc)})
            return SkillResult(
                status="error",
                skill_name=self.name,
                error=str(exc),
                duration_ms=elapsed,
            )

    @abstractmethod
    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Override in subclass. Called by run() with exception guard."""
