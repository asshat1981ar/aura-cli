"""Lint skill: runs flake8 over a configurable list of Python files."""
from __future__ import annotations

import subprocess
from typing import Any, Dict, List

from agents.skills.base import SkillBase
from agents.skills.test_and_observe import parse_flake8_output


class LintSkill(SkillBase):
    """Run flake8 over a list of files and return structured diagnostics.

    Input keys:
        files (List[str], optional): Explicit list of paths to lint.
            When omitted, falls back to git-staged Python files.
        config (str, optional): Path to a flake8 config file passed
            as ``--config=<path>``.

    Output keys:
        status (str): ``"success"`` | ``"violations_found"`` | ``"error"``
        files_checked (int): Number of files passed to flake8.
        violation_count (int): Total number of violations found.
        violations (List[dict]): Serialised Diagnostic objects.
        raw (str): Raw flake8 stdout for debugging.
    """

    name = "lint"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if "files" in input_data:
            files: List[str] = input_data["files"]
        else:
            files: List[str] = self._get_staged_files()
        config: str = input_data.get("config", "")

        if not files:
            return {
                "status": "success",
                "files_checked": 0,
                "violation_count": 0,
                "violations": [],
                "raw": "",
            }

        cmd = ["flake8"] + ([f"--config={config}"] if config else []) + files
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )
        except FileNotFoundError:
            return {
                "status": "error",
                "error": "flake8 not installed",
                "files_checked": len(files),
                "violation_count": 0,
                "violations": [],
                "raw": "",
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": "lint timed out after 60s",
                "files_checked": len(files),
                "violation_count": 0,
                "violations": [],
                "raw": "",
            }

        violations = parse_flake8_output(proc.stdout + proc.stderr)

        return {
            "status": "success" if proc.returncode == 0 else "violations_found",
            "files_checked": len(files),
            "violation_count": len(violations),
            "violations": [
                {
                    "severity": v.severity,
                    "kind": v.kind,
                    "message": v.message,
                    "file": v.primary_location.file,
                    "line": v.primary_location.line,
                    "col": v.primary_location.col,
                    "suggested_next_commands": v.suggested_next_commands,
                }
                for v in violations
            ],
            "raw": proc.stdout,
        }

    def _get_staged_files(self) -> List[str]:
        """Return git-staged Python files; empty list on any error."""
        try:
            out = subprocess.check_output(
                ["git", "diff", "--name-only", "--cached"],
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).decode()
            return [f for f in out.splitlines() if f.endswith(".py")]
        except Exception:
            return []
