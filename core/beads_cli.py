"""Helpers for resolving the repository-scoped BEADS CLI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


def local_beads_cli_path(project_root: Path | str | None = None) -> Path:
    root = Path(project_root) if project_root is not None else Path.cwd()
    executable = "bd.cmd" if os.name == "nt" else "bd"
    return root / "node_modules" / ".bin" / executable


def resolve_beads_cli(
    project_root: Path | str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    env_map = env or os.environ
    for key in ("BEADS_CLI", "BD_COMMAND"):
        value = env_map.get(key)
        if value and value.strip():
            return value.strip()

    roots: list[Path] = []
    if project_root is not None:
        roots.append(Path(project_root))

    cwd = Path.cwd()
    if cwd not in roots:
        roots.append(cwd)

    for root in roots:
        candidate = local_beads_cli_path(root)
        if candidate.exists():
            return str(candidate)

    return "bd"


def uses_repo_local_beads_cli(command: str, project_root: Path | str | None = None) -> bool:
    if not command or command == "bd":
        return False

    candidate = local_beads_cli_path(project_root)
    try:
        return Path(command).resolve() == candidate.resolve()
    except OSError:
        return Path(command) == candidate
