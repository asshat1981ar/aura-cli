#!/usr/bin/env python3
"""Pre-merge snapshot gate.

Runs three CI checks to ensure all auto-generated snapshot files are current
before a branch is merged into main.  Exits non-zero if any check fails.

Usage::

    python3 scripts/pre_merge_check.py

Environment variables:
    NO_COLOR   When set to any non-empty value, disables ANSI colour output.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

_NO_COLOR: bool = bool(os.environ.get("NO_COLOR", ""))

_GREEN = "" if _NO_COLOR else "\033[32m"
_RED = "" if _NO_COLOR else "\033[31m"
_BOLD = "" if _NO_COLOR else "\033[1m"
_RESET = "" if _NO_COLOR else "\033[0m"

TICK = f"{_GREEN}✓{_RESET}"
CROSS = f"{_RED}✗{_RESET}"

# ---------------------------------------------------------------------------
# Check definitions
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Check:
    """A single gate check."""

    name: str
    cmd: list[str]
    description: str


CHECKS: list[Check] = [
    Check(
        name="CLI reference docs current",
        cmd=[sys.executable, "scripts/generate_cli_reference.py", "--check"],
        description="docs/CLI_REFERENCE.md must reflect the current CLI schema",
    ),
    Check(
        name="CLI help snapshots current",
        cmd=[sys.executable, "-m", "pytest", "tests/test_cli_help_snapshots.py", "-q"],
        description="Help text snapshot files must match actual CLI output",
    ),
    Check(
        name="Sweep artifact tests pass",
        cmd=[
            sys.executable,
            "-m",
            "pytest",
            "tests/test_generate_active_sweep_artifacts.py",
            "-q",
        ],
        description="Active-sweep artifact generation tests must pass",
    ),
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_check(check: Check, *, cwd: Path = REPO_ROOT) -> tuple[bool, str]:
    """Run *check* and return ``(passed, combined_output)``."""
    result = subprocess.run(
        check.cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (result.stdout + result.stderr).strip()
    return result.returncode == 0, combined


def main(argv: list[str] | None = None) -> int:  # noqa: ARG001  (reserved for future flags)
    """Run all checks and return the exit code."""
    print(f"\n{_BOLD}=== Pre-Merge Snapshot Gate ==={_RESET}\n")

    results: list[tuple[Check, bool, str]] = []

    for check in CHECKS:
        print(f"  Running: {check.name}…", end=" ", flush=True)
        passed, output = run_check(check)
        symbol = TICK if passed else CROSS
        print(symbol)
        if not passed and output:
            # Indent the subprocess output for readability
            indented = "\n".join(f"    {line}" for line in output.splitlines())
            print(f"\n{indented}\n")
        results.append((check, passed, output))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    passed_count = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    all_passed = passed_count == total

    print()
    print(f"{_BOLD}--- Summary ---{_RESET}")
    for check, ok, _ in results:
        symbol = TICK if ok else CROSS
        print(f"  {symbol}  {check.name}")

    print()
    summary_color = _GREEN if all_passed else _RED
    print(
        f"{_BOLD}{summary_color}{passed_count}/{total} checks passed{_RESET}"
    )

    if not all_passed:
        failed_names = [c.name for c, ok, _ in results if not ok]
        print(
            f"\n{_RED}Fix the above failures before merging. "
            f"Failed: {', '.join(failed_names)}{_RESET}"
        )
        print(
            "\nQuick fix hints:"
            "\n  • Regen CLI reference : python3 scripts/generate_cli_reference.py"
            "\n  • Update snapshots    : python3 update_snapshots.py"
            "\n  • Run sweep artifacts : python3 scripts/generate_active_sweep_artifacts.py"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
