import os
import re
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from core.logging_utils import log_json # Import log_json


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    success: bool           # True iff exit_code == 0
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    execution_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    PYTHONDONTWRITEBYTECODE_ENV = {"PYTHONDONTWRITEBYTECODE": "1"}

    # Convenience
    @property
    def passed(self) -> bool:
        return self.success and not self.timed_out

    def summary(self) -> str:
        status = "PASS" if self.passed else ("TIMEOUT" if self.timed_out else "FAIL")
        return (
            f"[{status}] exit={self.exit_code} | "
            f"stdout={len(self.stdout)}c | stderr={len(self.stderr)}c"
        )

    def __str__(self):
        return (
            f"SandboxResult(passed={self.passed}, exit={self.exit_code})\n"
            f"  stdout: {self.stdout[:300]!r}\n"
            f"  stderr: {self.stderr[:300]!r}"
        )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SandboxAgent:
    """
    Real code execution in an isolated subprocess.

    Usage
    -----
    sandbox = SandboxAgent(brain, timeout=30)
    result  = sandbox.run_code("print('hello')")
    result  = sandbox.run_tests(code_str, test_str)

    The agent always returns a SandboxResult — it never raises.
    All execution evidence is persisted to brain.
    """

    # Matches pytest summary lines e.g. "5 passed, 1 failed"
    PYTEST_SUMMARY_RE = re.compile(
        r"(\d+)\s+passed|(\d+)\s+failed|(\d+)\s+error", re.IGNORECASE
    )

    def __init__(self, brain, timeout: int = 30, python_exec: Optional[str] = None):
        self.brain = brain
        self.timeout = timeout
        # Prefer the same interpreter running AURA
        self.python_exec = python_exec or self._find_python()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_code(self, code: str, extra_files: Optional[dict] = None) -> SandboxResult:
        """
        Execute a raw Python code string.
        extra_files: {filename: content} — written alongside the main script.
        """
        with tempfile.TemporaryDirectory(prefix="aura_sandbox_") as tmpdir:
            script = Path(tmpdir) / "aura_exec.py"
            script.write_text(code, encoding="utf-8")

            if extra_files:
                for fname, content in extra_files.items():
                    (Path(tmpdir) / fname).write_text(content, encoding="utf-8")

            result = self._run(str(script), cwd=tmpdir)
            self._record(result, label="run_code", code_snippet=code[:120])
            return result

    def run_file(self, file_path: str) -> SandboxResult:
        """Execute an existing Python file on disk."""
        result = self._run(file_path, cwd=str(Path(file_path).parent))
        self._record(result, label="run_file", code_snippet=file_path)
        return result

    def run_tests(self, code: str, tests: str) -> SandboxResult:
        """
        Write code + tests to a temp dir, then run pytest.
        Returns structured SandboxResult with parsed pass/fail counts.
        """
        with tempfile.TemporaryDirectory(prefix="aura_test_") as tmpdir:
            src = Path(tmpdir) / "source.py"
            tst = Path(tmpdir) / "test_source.py"
            src.write_text(code, encoding="utf-8")

            # Inject source import at top of test file if not present
            if "from source import" not in tests and "import source" not in tests:
                header = "import sys, os\nsys.path.insert(0, os.path.dirname(__file__))\n"
                tests = header + tests
            tst.write_text(tests, encoding="utf-8")

            result = self._run_pytest(tmpdir)
            result.metadata.update(self._parse_pytest_summary(result.stdout + result.stderr))
            self._record(result, label="run_tests", code_snippet=code[:120])
            return result

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    def _run(self, script_path: str, cwd: str) -> SandboxResult:
        """Run a single Python script via subprocess."""
        try:
            proc = subprocess.Popen(
                [self.python_exec, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
                return SandboxResult(
                    success=proc.returncode == 0,
                    exit_code=proc.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    execution_path=script_path,
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                return SandboxResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Execution timed out after {self.timeout}s",
                    timed_out=True,
                    execution_path=script_path,
                )
        except Exception as exc:
            log_json("ERROR", "sandbox_internal_error", details={"method": "_run", "error": str(exc), "script_path": script_path})
            return SandboxResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"SandboxAgent internal error: {exc}",
                execution_path=script_path,
            )

    def _run_pytest(self, tmpdir: str) -> SandboxResult:
        """Run pytest inside tmpdir."""
        try:
            proc = subprocess.Popen(
                [self.python_exec, "-m", "pytest", tmpdir, "-v", "--tb=short", "--no-header"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmpdir,
                text=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
                return SandboxResult(
                    success=proc.returncode == 0,
                    exit_code=proc.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    execution_path=tmpdir,
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                return SandboxResult(
                    success=False, exit_code=-1,
                    stdout="", stderr=f"pytest timed out after {self.timeout}s",
                    timed_out=True, execution_path=tmpdir,
                )
        except FileNotFoundError:
            # pytest not installed — fall back to unittest discovery
            return self._run_unittest(tmpdir)
        except Exception as exc:
            log_json("ERROR", "pytest_runner_error", details={"method": "_run_pytest", "error": str(exc), "tmpdir": tmpdir})
            return SandboxResult(
                success=False, exit_code=-1, stdout="",
                stderr=f"pytest runner error: {exc}",
                execution_path=tmpdir,
            )

    def _run_unittest(self, tmpdir: str) -> SandboxResult:
        """Fallback: run unittest discover when pytest is unavailable."""
        try:
            proc = subprocess.Popen(
                [self.python_exec, "-m", "unittest", "discover", "-s", tmpdir, "-v"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=tmpdir, text=True,
            )
            stdout, stderr = proc.communicate(timeout=self.timeout)
            return SandboxResult(
                success=proc.returncode == 0, exit_code=proc.returncode,
                stdout=stdout, stderr=stderr, execution_path=tmpdir,
            )
        except Exception as exc:
            log_json("ERROR", "unittest_runner_error", details={"method": "_run_unittest", "error": str(exc), "tmpdir": tmpdir})
            return SandboxResult(
                success=False, exit_code=-1, stdout="",
                stderr=f"unittest runner error: {exc}",
                execution_path=tmpdir,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_pytest_summary(self, output: str) -> dict:
        """Extract pass/fail/error counts from pytest output."""
        counts = {"passed": 0, "failed": 0, "errors": 0}
        for match in self.PYTEST_SUMMARY_RE.finditer(output):
            if match.group(1):
                counts["passed"] = int(match.group(1))
            if match.group(2):
                counts["failed"] = int(match.group(2))
            if match.group(3):
                counts["errors"] = int(match.group(3))
        return counts

    def _record(self, result: SandboxResult, label: str, code_snippet: str):
        self.brain.remember(
            f"SandboxAgent [{label}]: {result.summary()} | "
            f"code='{code_snippet[:80]}...'"
        )

    @staticmethod
    def _find_python() -> str:
        import sys
        return sys.executable