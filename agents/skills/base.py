"""Base class for all AURA skill modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator

from core.logging_utils import log_json

# ---------------------------------------------------------------------------
# Shared filesystem helpers
# ---------------------------------------------------------------------------

#: Directories that should always be excluded when walking project source trees.
SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".tox",
        ".venv",
        "venv",
        "dist",
        "build",
        "env",
        ".env",
        "site-packages",
        "test-aura-env",
        "aura_cli.egg-info",
        "tmp_out",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
    }
)


def iter_py_files(root: Path) -> Iterator[Path]:
    """Yield all ``*.py`` files under *root*, skipping :data:`SKIP_DIRS`.

    Compared to a bare ``root.rglob("*.py")``, this helper avoids scanning
    large virtual-environment or generated directories that are never part of
    the project's own source code.

    Args:
        root: Directory to scan.

    Yields:
        :class:`~pathlib.Path` objects for every ``.py`` file found.
    """
    for f in root.rglob("*.py"):
        if not any(part in SKIP_DIRS for part in f.parts):
            yield f


# ---------------------------------------------------------------------------
# Skill base class
# ---------------------------------------------------------------------------


class SkillBase(ABC):
    """
    Abstract base for all skill modules.

    Subclasses must set `name` and implement `run()`.
    All exceptions must be caught internally; run() must always return a dict.
    """

    name: str = "base_skill"

    def __init__(self, brain=None, model=None):
        self.brain = brain
        self.model = model

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
