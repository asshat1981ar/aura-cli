"""Base class for all AURA skill modules."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict
from core.logging_utils import log_json


class SkillBase(ABC):
    """
    Abstract base for all skill modules.

    Subclasses must set `name` and implement `run()`.
    All exceptions must be caught internally; run() must always return a dict.
    """

    name: str = "base_skill"

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the skill. Returns a JSON-serialisable dict."""
        try:
            return self._run(input_data)
        except Exception as exc:  # pylint: disable=broad-except
            log_json("ERROR", f"{self.name}_failed", details={"error": str(exc)})
            return {"error": str(exc), "skill": self.name}

    @abstractmethod
    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Override in subclass. Called by run() with exception guard."""
