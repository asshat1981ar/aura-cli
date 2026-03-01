from __future__ import annotations

from pathlib import Path


def resolve_project_path(project_root: Path, raw_path: object, default: str) -> Path:
    """Resolve a configured storage path against the inspected project root."""
    path_value = raw_path if isinstance(raw_path, str) and raw_path.strip() else default
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return Path(project_root) / candidate
