#!/usr/bin/env python3
"""Secrets scanner for AURA CLI.

Searches source files for patterns that look like hardcoded secrets:
API keys, tokens, passwords, Bearer auth headers, AWS keys, OpenAI keys,
and GitHub PATs.

Usage:
    python3 scripts/scan_secrets.py <directory>

Exits 1 if any findings are detected, 0 if clean.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCANNED_EXTENSIONS: frozenset[str] = frozenset({".py", ".json", ".yml", ".yaml", ".toml", ".cfg", ".sh", ".env"})

IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        # Test suites contain intentional secret-like strings as fixtures
        "tests",
        # Local virtualenvs bundled in the repo
        "test-aura-env",
        # User-local Claude Code config (not part of the committed codebase)
        ".claude",
    }
)

# Patterns that indicate a real secret value (not a placeholder).
# Each pattern is a compiled regex that matches a line containing a finding.
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str], bool]] = [
    # Assignment-style: api_key = "...", secret = '...', token = '...', password = '...'
    # Value must be at least 6 characters to skip very short permission keywords like
    # 'write', 'read', 'none' used in GitHub Actions permission blocks.
    # GitHub Actions expression syntax (${{ ... }}) is excluded at the line level.
    # check_placeholder=True — skip common example/placeholder values
    (
        "api_key/secret/token/password assignment",
        re.compile(
            r"""(?i)\b(api[_-]?key|secret|token|password)\s*[=:]\s*['"][^'"]{6,}['"]""",
            re.IGNORECASE,
        ),
        True,
    ),
    # Bearer token in header value
    (
        "Bearer token",
        re.compile(r"""Bearer\s+[A-Za-z0-9\-_\.]{16,}"""),
        True,
    ),
    # AWS access key ID — AKIA prefix is specific enough; don't apply generic placeholder check
    (
        "AWS access key (AKIA...)",
        re.compile(r"""\bAKIA[0-9A-Z]{16}\b"""),
        False,
    ),
    # OpenAI key  sk-...
    (
        "OpenAI API key (sk-...)",
        re.compile(r"""\bsk-[A-Za-z0-9]{16,}\b"""),
        False,
    ),
    # GitHub PAT  ghp_...
    (
        "GitHub PAT (ghp_...)",
        re.compile(r"""\bghp_[A-Za-z0-9]{16,}\b"""),
        False,
    ),
]

# Values that indicate the match is a placeholder, not a real secret.
# Only applied to patterns that have check_placeholder=True.
_PLACEHOLDER_FRAGMENTS: tuple[str, ...] = (
    "PLACEHOLDER",
    "YOUR_API_KEY",
    "YOUR_",
    "<",
    "example",
    "test",
    "fake",
    "dummy",
    "changeme",
    # Docstring / inline ellipsis placeholders  api_key="..."
    "...",
)

# Line-level skip patterns — if a line matches any of these, skip it entirely
# regardless of whether a secret pattern matches.
_SKIP_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # GitHub Actions expression syntax  ${{ secrets.FOO }}
    re.compile(r"""\$\{\{"""),
    # f-string or string that talks *about* a secret rather than containing one
    re.compile(r"""appears to contain a secret"""),
)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _is_placeholder(value: str) -> bool:
    """Return True if the matched text looks like a known placeholder."""
    for fragment in _PLACEHOLDER_FRAGMENTS:
        if fragment.lower() in value.lower():
            return True
    return False


def _scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Scan a single file and return (line_number, pattern_name, line) tuples."""
    findings: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return findings

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        # Skip comment lines
        if stripped.startswith("#") or stripped.startswith("//"):
            continue
        # Skip lines that match known-safe patterns (e.g. CI expressions, error messages)
        if any(p.search(line) for p in _SKIP_LINE_PATTERNS):
            continue

        for pattern_name, pattern, check_placeholder in _SECRET_PATTERNS:
            match = pattern.search(line)
            if match:
                if check_placeholder and _is_placeholder(match.group(0)):
                    continue
                findings.append((lineno, pattern_name, line.rstrip()))
                break  # one finding per line is enough

    return findings


def _iter_files(root: Path):
    """Yield all files under root, skipping ignored directories."""
    for item in root.rglob("*"):
        # Skip ignored directory trees
        if any(part in IGNORED_DIRS for part in item.parts):
            continue
        if not item.is_file():
            continue
        if item.suffix.lower() not in SCANNED_EXTENSIONS and item.name != ".env":
            # Also scan files named exactly ".env" (no suffix)
            if item.name not in {".env"}:
                continue
        yield item


def scan(root: Path) -> list[tuple[Path, int, str, str]]:
    """Scan all eligible files under root.

    Returns list of (file_path, line_number, pattern_name, line) tuples.
    """
    all_findings: list[tuple[Path, int, str, str]] = []
    for file_path in sorted(_iter_files(root)):
        for lineno, pattern_name, line in _scan_file(file_path):
            all_findings.append((file_path, lineno, pattern_name, line))
    return all_findings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]

    if not args:
        print(f"Usage: {sys.argv[0]} <directory>", file=sys.stderr)
        return 2

    root = Path(args[0]).resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        return 2

    findings = scan(root)

    if not findings:
        print("Secrets scan: clean — no findings.")
        return 0

    print(f"Secrets scan: {len(findings)} finding(s) detected!\n")
    for file_path, lineno, pattern_name, line in findings:
        try:
            display_path = file_path.relative_to(root)
        except ValueError:
            display_path = file_path
        print(f"  {display_path}:{lineno}  [{pattern_name}]")
        print(f"    {line}")
    print(f"\nTotal: {len(findings)} finding(s). Fix or mark as PLACEHOLDER before committing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
