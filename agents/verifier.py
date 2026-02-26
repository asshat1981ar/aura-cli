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

    def _changed_test_files(self, project_root: Path) -> list:
        """Use git diff to find changed files and map to related test files."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=str(project_root), capture_output=True, text=True, timeout=10
            )
            changed = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if not changed:
                return []
            # Map each changed source file to a likely test file
            test_files = []
            for f in changed:
                p = Path(f)
                # e.g. agents/coder.py → tests/test_coder.py
                candidate = project_root / "tests" / f"test_{p.stem}.py"
                if candidate.exists():
                    test_files.append(str(candidate))
                # Also include directly changed test files
                if p.stem.startswith("test_") and (project_root / f).exists():
                    test_files.append(str(project_root / f))
            return list(set(test_files))
        except Exception as e:
            log_json("WARN", "verifier_changed_files_failed", details={"error": str(e)})
            return []

    def run(self, input_data: Dict) -> Dict:
        if input_data.get("dry_run"):
            return {"status": "skip", "failures": [], "logs": "dry_run"}

        project_root = Path(input_data.get("project_root", "."))
        cmd = ["python3", "-m", "pytest", "-q"]
        tests = input_data.get("tests")
        if tests:
            if isinstance(tests, list):
                cmd = tests[0].split()
            elif isinstance(tests, str):
                cmd = tests.split()
        else:
            # Incremental: only run tests related to changed files
            incremental = self._changed_test_files(project_root)
            if incremental:
                log_json("INFO", "verifier_incremental_tests", details={"files": incremental})
                cmd = ["python3", "-m", "pytest", "-q"] + incremental

        sanitize_command(cmd)

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return {"status": "fail", "failures": ["pytest_timeout"], "logs": "timeout"}

        status = "pass" if proc.returncode == 0 else "fail"
        failures = [] if proc.returncode == 0 else ["pytest_failed"]
        logs = (proc.stdout or "") + (proc.stderr or "")
        return {"status": status, "failures": failures, "logs": logs.strip()}
