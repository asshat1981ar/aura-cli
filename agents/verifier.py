import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Dict

from agents.base import Agent
from core.sanitizer import sanitize_command
from core.logging_utils import log_json


class VerifierAgent(Agent):
    name = "verify"

    def __init__(self, timeout: int = 120):
        self.timeout = timeout

    def _available_test_files(self, project_root: Path) -> list[str]:
        tests_root = project_root / "tests"
        if not tests_root.is_dir():
            return []
        return sorted(
            str(path.relative_to(project_root))
            for path in tests_root.rglob("test_*.py")
            if path.is_file()
        )

    def _tokenize_path(self, value: str) -> list[str]:
        return [token for token in re.findall(r"[a-zA-Z0-9]+", (value or "").lower()) if len(token) > 1]

    def _related_test_files(self, project_root: Path, file_path: str, available_tests: list[str], limit: int = 3) -> list[str]:
        rel = str(Path(file_path))
        source = Path(rel)

        direct_matches = []
        if rel.startswith("tests/") and (project_root / rel).is_file():
            direct_matches.append(rel)

        preferred_names = {
            f"test_{source.stem}.py",
            f"test_{source.stem}_wrapper.py",
        }
        scored: list[tuple[int, str]] = []
        source_tokens = set(self._tokenize_path(rel))

        for candidate in available_tests:
            candidate_path = Path(candidate)
            score = 0
            if candidate_path.name in preferred_names:
                score += 10
            if candidate_path.stem == f"test_{source.stem}":
                score += 8
            overlap = source_tokens & set(self._tokenize_path(candidate))
            score += len(overlap) * 2
            if source.parent.name and source.parent.name in candidate:
                score += 1
            if score > 0:
                scored.append((score, candidate))

        scored.sort(key=lambda item: (-item[0], len(Path(item[1]).parts), item[1]))
        ordered = direct_matches + [candidate for _, candidate in scored if candidate not in direct_matches]
        return ordered[:limit]

    def _changed_files_from_git(self, project_root: Path) -> list[str]:
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=str(project_root), capture_output=True, text=True, timeout=10
            )
            return [line.strip() for line in result.stdout.splitlines() if line.strip()]
        except Exception as e:
            log_json("WARN", "verifier_changed_files_failed", details={"error": str(e)})
            return []

    def _changed_test_files(self, project_root: Path, change_set: Dict | None = None) -> list[str]:
        """Infer the smallest useful pytest target set from change_set or git diff."""
        changed: list[str] = []
        if isinstance(change_set, dict):
            for change in change_set.get("changes", []):
                if not isinstance(change, dict):
                    continue
                file_path = change.get("file_path")
                if isinstance(file_path, str) and file_path.strip():
                    changed.append(str(Path(file_path)))
        if not changed:
            changed = self._changed_files_from_git(project_root)
        if not changed:
            return []

        available_tests = self._available_test_files(project_root)
        if not available_tests:
            return []

        test_files: list[str] = []
        seen: set[str] = set()
        for file_path in changed:
            for candidate in self._related_test_files(project_root, file_path, available_tests):
                if candidate in seen:
                    continue
                seen.add(candidate)
                test_files.append(candidate)
        return test_files

    def _normalize_test_command(self, tests) -> list[str]:
        if isinstance(tests, list) and tests:
            first = tests[0]
            if isinstance(first, str) and first.strip():
                return shlex.split(first)
        elif isinstance(tests, str) and tests.strip():
            return shlex.split(tests)
        return []

    def _is_repo_wide_pytest_command(self, cmd: list[str]) -> bool:
        normalized = [part.strip() for part in cmd if isinstance(part, str) and part.strip()]
        if not normalized:
            return False
        if normalized in (["pytest"], ["pytest", "-q"]):
            return True
        return normalized in (["python", "-m", "pytest", "-q"], ["python3", "-m", "pytest", "-q"])

    def run(self, input_data: Dict) -> Dict:
        if input_data.get("dry_run"):
            return {"status": "skip", "failures": [], "logs": "dry_run"}

        project_root = Path(input_data.get("project_root", "."))
        cmd = ["python3", "-m", "pytest", "-q"]
        tests = input_data.get("tests")
        explicit_cmd = self._normalize_test_command(tests)
        incremental = self._changed_test_files(project_root, change_set=input_data.get("change_set"))

        if explicit_cmd and not self._is_repo_wide_pytest_command(explicit_cmd):
            cmd = explicit_cmd
        elif incremental:
            log_json("INFO", "verifier_incremental_tests", details={"files": incremental})
            cmd = ["python3", "-m", "pytest", "-q"] + incremental
        elif explicit_cmd:
            cmd = explicit_cmd

        sanitize_command(cmd)

        try:
            env = dict(os.environ)
            env.setdefault("AURA_SKIP_CHDIR", "1")
            proc = subprocess.run(
                cmd,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "fail",
                "failures": ["pytest_timeout"],
                "logs": f"timeout after {self.timeout}s while running: {' '.join(cmd)}",
            }

        status = "pass" if proc.returncode == 0 else "fail"
        failures = [] if proc.returncode == 0 else ["pytest_failed"]
        logs = (proc.stdout or "") + (proc.stderr or "")
        return {"status": status, "failures": failures, "logs": logs.strip()}
