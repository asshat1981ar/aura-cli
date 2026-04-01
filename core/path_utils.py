"""Centralized path filtering utilities for AURA CLI.

Provides shared file filtering logic to reduce code duplication
across skills and agents.
"""
from pathlib import Path
from typing import Set, Union


# Default patterns to skip
DEFAULT_SKIP_PATTERNS: Set[str] = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "*.egg-info",
    "build",
    "dist",
    ".coverage",
    "htmlcov",
    ".DS_Store",
    "Thumbs.db",
}


def should_skip_path(
    path: Union[str, Path],
    skip_patterns: Set[str] = None,
    additional_patterns: Set[str] = None,
) -> bool:
    """Check if a path should be skipped based on patterns.

    Args:
        path: The file or directory path to check
        skip_patterns: Custom patterns to skip (defaults to DEFAULT_SKIP_PATTERNS)
        additional_patterns: Extra patterns to add to defaults

    Returns:
        True if the path should be skipped, False otherwise

    Examples:
        >>> should_skip_path("/project/.git/config")
        True
        >>> should_skip_path("/project/src/main.py")
        False
        >>> should_skip_path("/project/node_modules/lodash", additional_paths={"custom_dir"})
        True
    """
    path_str = str(path)
    path_obj = Path(path)

    patterns = skip_patterns if skip_patterns is not None else DEFAULT_SKIP_PATTERNS.copy()
    if additional_patterns:
        patterns = patterns | additional_patterns

    # Check exact name match
    if path_obj.name in patterns:
        return True

    # Check suffix patterns (e.g., *.egg-info)
    for pattern in patterns:
        if pattern.startswith("*") and pattern[1:] in path_str:
            return True

    # Check if any part of the path matches
    for part in path_obj.parts:
        if part in patterns:
            return True

    return False


def filter_paths(
    paths: list[Union[str, Path]],
    skip_patterns: Set[str] = None,
    additional_patterns: Set[str] = None,
) -> list[Path]:
    """Filter a list of paths, removing those that should be skipped.

    Args:
        paths: List of paths to filter
        skip_patterns: Custom patterns to skip
        additional_patterns: Extra patterns to add

    Returns:
        Filtered list of paths (as Path objects)
    """
    return [
        Path(p) for p in paths
        if not should_skip_path(p, skip_patterns, additional_patterns)
    ]


def get_project_files(
    project_root: Union[str, Path],
    extensions: Set[str] = None,
    skip_patterns: Set[str] = None,
) -> list[Path]:
    """Get all relevant files in a project directory.

    Args:
        project_root: Root directory to scan
        extensions: File extensions to include (e.g., {'.py', '.js'})
        skip_patterns: Patterns to skip

    Returns:
        List of file paths
    """
    root = Path(project_root)
    files = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if should_skip_path(path, skip_patterns):
            continue
        if extensions and path.suffix not in extensions:
            continue
        files.append(path)

    return files
