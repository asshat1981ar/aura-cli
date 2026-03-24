"""Filesystem jailing, secure directory creation, and cleanup for environment isolation."""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path


def jail_path(path: Path, workspace_root: Path) -> Path:
    """Resolve *path* and verify it stays within *workspace_root*.

    Follows the same pattern as ``_jail()`` in ``tools/mcp_server.py``.

    Raises:
        ValueError: If the resolved path escapes the workspace root.
    """
    resolved = (workspace_root / path).resolve()
    try:
        resolved.relative_to(workspace_root.resolve())
    except ValueError:
        raise ValueError(
            f"Path '{path}' escapes workspace root '{workspace_root}'. "
            "Only paths within the workspace are allowed."
        )
    return resolved


def create_secure_secrets_dir(path: Path) -> Path:
    """Create a secrets directory with mode 0700 (owner-only access).

    Returns:
        The created directory path.
    """
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(str(path), 0o700)
    return path


def create_workspace_tree(workspace_root: Path) -> dict:
    """Create the full workspace directory tree for an environment.

    Layout::

        workspace_root/
            config/
            logs/
            temp/
            secrets/   (mode 0700)
            deps/

    Returns:
        Dict mapping directory names to their paths.
    """
    dirs = {}
    for subdir in ("config", "logs", "temp", "deps"):
        d = workspace_root / subdir
        d.mkdir(parents=True, exist_ok=True)
        dirs[subdir] = d

    dirs["secrets"] = create_secure_secrets_dir(workspace_root / "secrets")
    return dirs


def cleanup_stale(workspace: Path, max_age_hours: int = 24) -> dict:
    """Remove files in temp/ and logs/ older than *max_age_hours*.

    Returns:
        Dict with counts of removed files per directory.
    """
    cutoff = time.time() - (max_age_hours * 3600)
    removed = {"temp": 0, "logs": 0}

    for subdir in ("temp", "logs"):
        target = workspace / subdir
        if not target.is_dir():
            continue
        for item in target.iterdir():
            try:
                if item.stat().st_mtime < cutoff:
                    if item.is_file():
                        item.unlink()
                        removed[subdir] += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        removed[subdir] += 1
            except OSError:
                continue

    return removed


def teardown_workspace(workspace: Path, *, preserve_secrets: bool = True) -> dict:
    """Remove a workspace, optionally preserving the secrets directory.

    Returns:
        Dict describing what was removed.
    """
    if not workspace.exists():
        return {"status": "not_found", "path": str(workspace)}

    if preserve_secrets:
        workspace / "secrets"
        # Remove everything except secrets
        for item in workspace.iterdir():
            if item.name == "secrets":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        return {"status": "cleaned", "preserved": "secrets", "path": str(workspace)}
    else:
        shutil.rmtree(workspace)
        return {"status": "removed", "path": str(workspace)}
