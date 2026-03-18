"""Pull-request context helpers for GitHub automation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath

_DOC_PREFIXES = ("docs/", "plans/")
_DOC_EXTENSIONS = {".md", ".rst", ".txt", ".adoc"}
_DEPENDENCY_FILES = {
    "requirements.txt",
    "pyproject.toml",
    "package.json",
    "package-lock.json",
    "poetry.lock",
}


def _normalize_path(path: str) -> str:
    normalized = path.strip()
    if normalized.startswith("./"):
        return normalized[2:]
    return normalized


def _normalize_paths(paths: list[str]) -> list[str]:
    return [_normalize_path(path) for path in paths if path and path.strip()]


@dataclass(slots=True)
class PRContext:
    """Small summary of a PR used by routing and policy code."""

    number: int | None = None
    title: str = ""
    base_ref: str = "main"
    head_ref: str = ""
    draft: bool = False
    changed_files: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    @classmethod
    def from_changed_files(
        cls,
        changed_files: list[str],
        *,
        number: int | None = None,
        title: str = "",
        base_ref: str = "main",
        head_ref: str = "",
        draft: bool = False,
        labels: list[str] | None = None,
    ) -> "PRContext":
        return cls(
            number=number,
            title=title,
            base_ref=base_ref,
            head_ref=head_ref,
            draft=draft,
            changed_files=_normalize_paths(changed_files),
            labels=labels or [],
        )

    @property
    def touched_file_count(self) -> int:
        return len(self.changed_files)

    @property
    def docs_only(self) -> bool:
        if not self.changed_files:
            return False
        return all(self._is_doc_path(path) for path in self.changed_files)

    @property
    def touches_python(self) -> bool:
        return any(path.endswith(".py") for path in self.changed_files)

    @property
    def touches_core(self) -> bool:
        prefixes = ("core/", "agents/", "aura_cli/", "tools/")
        return any(path.endswith(".py") and path.startswith(prefixes) for path in self.changed_files)

    @property
    def touches_workflows(self) -> bool:
        return any(path.startswith(".github/workflows/") for path in self.changed_files)

    @property
    def touches_dependencies(self) -> bool:
        return any(PurePosixPath(path).name in _DEPENDENCY_FILES for path in self.changed_files)

    @property
    def touches_tests(self) -> bool:
        return any(path.startswith("tests/") for path in self.changed_files)

    @property
    def touches_protected_paths(self) -> bool:
        prefixes = (".github/", "tools/", "core/", "aura_cli/")
        protected_files = {"requirements.txt", "pyproject.toml"}
        return any(
            path.startswith(prefixes) or PurePosixPath(path).name in protected_files
            for path in self.changed_files
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @staticmethod
    def _is_doc_path(path: str) -> bool:
        normalized = _normalize_path(path)
        if normalized.startswith(_DOC_PREFIXES):
            return True
        return PurePosixPath(normalized).suffix in _DOC_EXTENSIONS
