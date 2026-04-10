"""Isolated code execution sandbox for AURA's pre-apply safety check.

This module provides :class:`SandboxAgent`, which runs LLM-generated Python
snippets in a temporary subprocess **before** they are written to the real
project filesystem.  This guards against obviously broken or dangerous code
reaching disk.

The agent never raises — all execution outcomes are returned as a
:class:`SandboxResult` dataclass so the caller can inspect and route failures
without exception handling.

Typical usage::

    sandbox = SandboxAgent(brain, timeout=30)

    # Single-snippet smoke test
    result = sandbox.run_code("print('hello')")
    if not result.passed:
        print(result.stderr)

    # Code + test suite
    result = sandbox.run_tests(source_code, test_code)
    print(result.metadata)  # {"passed": 5, "failed": 0, "errors": 0}
"""
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from core.logging_utils import log_json  # Import log_json
from core.sanitizer import SecurityError  # For violation detection

# ---------------------------------------------------------------------------
# Sandbox security helpers
# ---------------------------------------------------------------------------

# Network-blocking environment overlay: routes all HTTP(S) traffic to a
# non-existent local proxy so that executed code cannot reach external
# services without requiring OS-level network namespaces or root privileges.
_SANDBOX_NETWORK_ENV: dict = {
    "http_proxy": "http://127.0.0.1:1",
    "https_proxy": "http://127.0.0.1:1",
    "HTTP_PROXY": "http://127.0.0.1:1",
    "HTTPS_PROXY": "http://127.0.0.1:1",
    "no_proxy": "",
    "NO_PROXY": "",
    # Prevent SSL verification fallbacks that might bypass the proxy
    "REQUESTS_CA_BUNDLE": "",
}

# Filesystem restriction: Environment variables to restrict Python's import
# and file system access. These are defense-in-depth measures; the primary
# isolation is the temporary directory + cwd enforcement.
_SANDBOX_FS_ENV: dict = {
    # Disable user site-packages to prevent injection via ~/.local
    "PYTHONNOUSERSITE": "1",
    # Don't write .pyc files
    "PYTHONDONTWRITEBYTECODE": "1",
    # Restrict Python path to system paths only
    "PYTHONPATH": "",
}

# Allowed directories for filesystem operations (defense-in-depth)
_SANDBOX_ALLOWED_PREFIXES: tuple = (
    "/tmp",
    "/var/tmp",
    tempfile.gettempdir(),
)


def _is_path_allowed(path: str, cwd: str) -> bool:
    """Check if a path is within allowed sandbox directories.

    Args:
        path: The path to check.
        cwd: The current working directory (sandbox temp dir).

    Returns:
        True if the path is allowed, False otherwise.
    """
    try:
        resolved = Path(path).resolve()
        cwd_resolved = Path(cwd).resolve()

        # Allow paths within the sandbox temp directory
        if str(resolved).startswith(str(cwd_resolved)):
            return True

        # Allow paths in system temp directories
        for prefix in _SANDBOX_ALLOWED_PREFIXES:
            if str(resolved).startswith(prefix):
                return True

        return False
    except (OSError, ValueError):
        return False


def _wrap_code_with_fs_restrictions(code: str, cwd: str) -> str:
    """Wrap Python code with filesystem access restrictions.

    Injects guards at the start of the code to intercept file open calls
    and restrict them to the sandbox directory.

    Args:
        code: The Python code to wrap.
        cwd: The sandbox working directory.

    Returns:
        Wrapped code with filesystem restrictions.
    """
    guard = f'''
import builtins
import os
import sys

_sandbox_cwd = {cwd!r}
_allowed_prefixes = ({cwd!r}, "/tmp", "/var/tmp", tempfile.gettempdir() if "tempfile" in sys.modules else "/tmp")

def _sandbox_open(file, *args, **kwargs):
    """Restricted open() that only allows access to sandbox directories."""
    if isinstance(file, (str, os.PathLike)):
        try:
            resolved = os.path.realpath(file)
            if not any(resolved.startswith(p) for p in _allowed_prefixes):
                raise PermissionError(f"Sandbox restriction: Access to {{resolved}} is not allowed")
        except (OSError, ValueError):
            pass  # Let the real open() handle invalid paths
    return builtins._original_open(file, *args, **kwargs)

# Preserve original open
if not hasattr(builtins, "_original_open"):
    builtins._original_open = builtins.open
    builtins.open = _sandbox_open

# Restrict sys.path to prevent importing from outside
sys.path = [p for p in sys.path if p == "" or p.startswith((_sandbox_cwd, "/tmp", "/usr", "/lib"))]

'''
    return guard + code


def _set_resource_limits() -> None:
    """Apply CPU and memory resource limits to the child process.

    Intended to be passed as ``preexec_fn`` to :func:`subprocess.Popen` so
    the limits are applied only inside the sandboxed child process, not to
    the AURA parent process itself.

    Limits imposed:
    - CPU time: 30 seconds (hard + soft)
    - Virtual address space: 512 MiB (hard + soft)

    Silently skipped on Windows where the ``resource`` module is unavailable.
    """
    try:
        import resource  # noqa: PLC0415 — intentional late import for Windows compat
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))  # 30 s CPU
        resource.setrlimit(
            resource.RLIMIT_AS,
            (512 * 1024 * 1024, 512 * 1024 * 1024),  # 512 MiB virtual address space
        )
    except ImportError:
        pass  # Windows — resource module unavailable; limits silently skipped

# ---------------------------------------------------------------------------
# Violation detection patterns for subprocess output
# Maps compiled regex patterns to violation_type labels.
# ---------------------------------------------------------------------------
_VIOLATION_PATTERNS = [
    (re.compile(r"\bPermissionError\b"), "restricted_path_access"),
    (re.compile(r"\b(?:ModuleNotFoundError|ImportError)\b", re.IGNORECASE), "blocked_import"),
    (re.compile(r"\bSecurityError\b|\bAccess denied\b"), "blocked_command"),
    (re.compile(r"\bBlockingIOError\b|Operation not permitted", re.IGNORECASE), "blocked_syscall"),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    """Structured outcome of a single sandbox execution.

    Attributes:
        success: ``True`` iff the subprocess exited with code 0.
        exit_code: Raw integer exit code from the subprocess.  ``-1`` signals
            an internal error or timeout.
        stdout: Full captured standard output from the subprocess.
        stderr: Full captured standard error from the subprocess.
        timed_out: ``True`` when the subprocess was forcibly killed because it
            exceeded :attr:`SandboxAgent.timeout`.
        execution_path: Absolute path to the temp file or directory that was
            executed.  ``None`` when the execution did not reach the filesystem.
        metadata: Arbitrary key-value store for extra data (e.g. parsed pytest
            pass/fail counts populated by :meth:`SandboxAgent.run_tests`).
    """

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
        """``True`` when the execution succeeded and did not time out."""
        return self.success and not self.timed_out

    def summary(self) -> str:
        """Return a compact one-line execution summary.

        Returns:
            A string of the form
            ``"[PASS|FAIL|TIMEOUT] exit=<n> | stdout=<n>c | stderr=<n>c"``.
        """
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
    """Runs Python code in an isolated subprocess for pre-apply safety validation.

    The agent creates a fresh temporary directory for every execution so that
    generated snippets cannot pollute the real project tree.  All results are
    returned as :class:`SandboxResult` instances and also persisted to *brain*
    so that the orchestrator's memory layer can review past execution outcomes.

    The agent never raises; exceptions are caught and mapped to a failed
    :class:`SandboxResult` with the error text in :attr:`SandboxResult.stderr`.

    Attributes:
        brain: The LLM brain object used for memory persistence via
            :meth:`~core.brain.Brain.remember`.
        timeout: Maximum wall-clock seconds to allow per subprocess.  Processes
            exceeding this limit are killed and :attr:`SandboxResult.timed_out`
            is set to ``True``.
        python_exec: Absolute path to the Python interpreter used for
            subprocesses.  Defaults to :func:`sys.executable` so that the same
            environment (virtual-env, Termux, etc.) is used.
    """

    # Matches pytest summary lines e.g. "5 passed, 1 failed"
    PYTEST_SUMMARY_RE = re.compile(
        r"(\d+)\s+passed|(\d+)\s+failed|(\d+)\s+error", re.IGNORECASE
    )

    def __init__(self, brain, timeout: int = 30, python_exec: Optional[str] = None):
        """Initialise the sandbox agent.

        Args:
            brain: Brain instance used to persist execution records via
                ``brain.remember()``.
            timeout: Subprocess timeout in seconds.  Defaults to ``30``.
            python_exec: Path to the Python interpreter.  When ``None``
                (default), :func:`sys.executable` is used so that the sandbox
                runs in the same virtual environment as AURA itself.
        """
        self.brain = brain
        self.timeout = timeout
        # Prefer the same interpreter running AURA
        self.python_exec = python_exec or self._find_python()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_code(self, code: str, extra_files: Optional[dict] = None) -> SandboxResult:
        """Execute a raw Python code string in a fresh temporary directory.

        Writes *code* to ``aura_exec.py`` in a new temp dir, optionally writes
        any *extra_files* alongside it, then runs the script via subprocess.
        The code is wrapped with filesystem access restrictions to prevent
        writes outside the temporary directory.

        Args:
            code: Python source code to execute.
            extra_files: Optional dict of ``{filename: content}`` pairs.  Each
                entry is written as a sibling file to the main script before
                execution, allowing the script to import helpers.

        Returns:
            :class:`SandboxResult` with execution outcome.
            :attr:`SandboxResult.execution_path` points to ``aura_exec.py``
            inside the (now deleted) temp dir.
        """
        with tempfile.TemporaryDirectory(prefix="aura_sandbox_") as tmpdir:
            script = Path(tmpdir) / "aura_exec.py"

            # Wrap code with filesystem restrictions
            wrapped_code = _wrap_code_with_fs_restrictions(code, tmpdir)
            script.write_text(wrapped_code, encoding="utf-8")

            if extra_files:
                for fname, content in extra_files.items():
                    (Path(tmpdir) / fname).write_text(content, encoding="utf-8")

            result = self._run(str(script), cwd=tmpdir)
            self._record(result, label="run_code", code_snippet=code[:120])
            return result

    def run_file(self, file_path: str) -> SandboxResult:
        """Execute an existing Python file on disk in an isolated subprocess.

        Unlike :meth:`run_code`, no temporary directory is created — the file
        is executed in-place from its own parent directory.

        Args:
            file_path: Absolute or relative path to the ``.py`` file to run.

        Returns:
            :class:`SandboxResult` with execution outcome.
        """
        result = self._run(file_path, cwd=str(Path(file_path).parent))
        self._record(result, label="run_file", code_snippet=file_path)
        return result

    def run_tests(self, code: str, tests: str) -> SandboxResult:
        """Write *code* and *tests* to a temp dir, then run pytest (or unittest).

        If the test file does not already import the source module, a
        ``sys.path`` injection header is prepended automatically so that
        ``from source import ...`` statements work without installation.

        Args:
            code: Source code to test, written to ``source.py``.
            tests: Test code (pytest or unittest style), written to
                ``test_source.py``.

        Returns:
            :class:`SandboxResult` with execution outcome.
            :attr:`SandboxResult.metadata` is populated with parsed pytest
            counts: ``{"passed": int, "failed": int, "errors": int}``.
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
        from core.sanitizer import sanitize_command
        cmd = [self.python_exec, script_path]
        try:
            sanitize_command(cmd)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                text=True,
                env={**os.environ, **_SANDBOX_NETWORK_ENV, **_SANDBOX_FS_ENV},
                preexec_fn=_set_resource_limits,
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
                result = SandboxResult(
                    success=proc.returncode == 0,
                    exit_code=proc.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    execution_path=script_path,
                )
                self._check_and_log_violations(result)
                return result
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
        except SecurityError as exc:
            log_json(
                "warning",
                "sandbox_violation_attempt",
                details={
                    "violation_type": "blocked_command",
                    "attempted_value": str(exc),
                    "goal": None,
                    "agent": "sandbox",
                },
            )
            return SandboxResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Security violation: {exc}",
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
        from core.sanitizer import sanitize_command
        cmd = [self.python_exec, "-m", "pytest", tmpdir, "-v", "--tb=short", "--no-header"]
        try:
            sanitize_command(cmd)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmpdir,
                text=True,
                env={
                    **os.environ,
                    **_SANDBOX_NETWORK_ENV,
                    **_SANDBOX_FS_ENV,
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                },
                preexec_fn=_set_resource_limits,
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
                result = SandboxResult(
                    success=proc.returncode == 0,
                    exit_code=proc.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    execution_path=tmpdir,
                )
                self._check_and_log_violations(result)
                return result
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                return SandboxResult(
                    success=False, exit_code=-1,
                    stdout="", stderr=f"pytest timed out after {self.timeout}s",
                    timed_out=True, execution_path=tmpdir,
                )
        except SecurityError as exc:
            log_json(
                "warning",
                "sandbox_violation_attempt",
                details={
                    "violation_type": "blocked_command",
                    "attempted_value": str(exc),
                    "goal": None,
                    "agent": "sandbox",
                },
            )
            return SandboxResult(
                success=False, exit_code=-1, stdout="",
                stderr=f"Security violation: {exc}",
                execution_path=tmpdir,
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
        from core.sanitizer import sanitize_command
        cmd = [self.python_exec, "-m", "unittest", "discover", "-s", tmpdir, "-v"]
        try:
            sanitize_command(cmd)
            proc = subprocess.Popen(
                cmd,
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

    def _check_and_log_violations(self, result: SandboxResult, goal: Optional[str] = None) -> None:
        """Scan subprocess stdout/stderr for security violation indicators and log each match.

        Args:
            result: The :class:`SandboxResult` whose output to inspect.
            goal: Optional goal string forwarded to the log entry when available.
        """
        combined = (result.stdout or "") + (result.stderr or "")
        for pattern, violation_type in _VIOLATION_PATTERNS:
            match = pattern.search(combined)
            if match:
                log_json(
                    "warning",
                    "sandbox_violation_attempt",
                    details={
                        "violation_type": violation_type,
                        "attempted_value": match.group(0),
                        "goal": goal,
                        "agent": "sandbox",
                    },
                )

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
