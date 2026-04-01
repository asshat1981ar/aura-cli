"""Tests for scripts/pre_merge_check.py.

Covers:
  1. All checks pass  → exit code 0
  2. One check fails  → exit code 1
  3. All checks fail  → exit code 1
  4. Output contains each check name in the summary
  5. NO_COLOR env suppresses ANSI escape sequences
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Load the module under test without executing __main__
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "pre_merge_check.py"


_MODULE_NAME = "pre_merge_check"


def _load_module(env_override: dict[str, str] | None = None) -> types.ModuleType:
    """Load the script as a module, optionally with patched environment variables.

    The module is registered in *sys.modules* under a unique key each call so
    that module-level constants (e.g. NO_COLOR) are re-evaluated every time.
    """
    # Give each load a unique name so module-level state is not shared between tests.
    import uuid
    unique_name = f"{_MODULE_NAME}_{uuid.uuid4().hex}"

    spec = importlib.util.spec_from_file_location(unique_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so @dataclass can resolve cls.__module__
    sys.modules[unique_name] = mod

    if env_override is not None:
        with patch.dict(os.environ, env_override, clear=False):
            spec.loader.exec_module(mod)
    else:
        spec.loader.exec_module(mod)

    return mod


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_completed_process(returncode: int, stdout: str = "", stderr: str = "") -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPreMergeCheckAllPass(unittest.TestCase):
    """All three subprocess calls succeed → exit code 0."""

    def test_returns_zero_when_all_checks_pass(self) -> None:
        mod = _load_module()
        always_ok = _make_completed_process(returncode=0, stdout="ok")

        with patch.object(mod.subprocess, "run", return_value=always_ok):
            exit_code = mod.main()

        self.assertEqual(exit_code, 0)

    def test_summary_shows_three_of_three(self, capsys=None) -> None:
        """3/3 checks passed must appear in printed output."""
        mod = _load_module()
        always_ok = _make_completed_process(returncode=0)

        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with patch.object(mod.subprocess, "run", return_value=always_ok):
            with redirect_stdout(buf):
                exit_code = mod.main()

        output = buf.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("3/3 checks passed", output)


class TestPreMergeCheckOneFailure(unittest.TestCase):
    """Second check fails → exit code 1."""

    def _side_effect_second_fails(self, *args, **kwargs) -> MagicMock:
        cmd = args[0] if args else kwargs.get("args", [])
        # Fail the help-snapshots check (contains 'test_cli_help_snapshots')
        if any("test_cli_help_snapshots" in str(c) for c in cmd):
            return _make_completed_process(returncode=1, stdout="FAILED snapshot mismatch")
        return _make_completed_process(returncode=0, stdout="ok")

    def test_returns_one_when_one_check_fails(self) -> None:
        mod = _load_module()

        with patch.object(mod.subprocess, "run", side_effect=self._side_effect_second_fails):
            exit_code = mod.main()

        self.assertEqual(exit_code, 1)

    def test_output_contains_all_check_names(self) -> None:
        """Every check name must appear in the printed summary regardless of pass/fail."""
        mod = _load_module()
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with patch.object(mod.subprocess, "run", side_effect=self._side_effect_second_fails):
            with redirect_stdout(buf):
                mod.main()

        output = buf.getvalue()
        for check in mod.CHECKS:
            self.assertIn(check.name, output, f"Check name '{check.name}' missing from output")


class TestPreMergeCheckAllFail(unittest.TestCase):
    """All checks fail → exit code 1, summary shows 0/3."""

    def test_returns_one_and_zero_of_three(self) -> None:
        mod = _load_module()
        always_fail = _make_completed_process(returncode=1, stderr="error")

        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with patch.object(mod.subprocess, "run", return_value=always_fail):
            with redirect_stdout(buf):
                exit_code = mod.main()

        output = buf.getvalue()
        self.assertEqual(exit_code, 1)
        self.assertIn("0/3 checks passed", output)


class TestPreMergeCheckNoColor(unittest.TestCase):
    """When NO_COLOR is set the module must not emit ANSI escape codes."""

    def test_no_ansi_when_no_color_set(self) -> None:
        mod = _load_module(env_override={"NO_COLOR": "1"})

        # The colour constants should be empty strings
        self.assertEqual(mod._GREEN, "")
        self.assertEqual(mod._RED, "")
        self.assertEqual(mod._BOLD, "")
        self.assertEqual(mod._RESET, "")


class TestPreMergeCheckSubprocessCommand(unittest.TestCase):
    """Verify the exact subprocess commands issued match the spec."""

    def test_three_subprocesses_called(self) -> None:
        mod = _load_module()
        always_ok = _make_completed_process(returncode=0)
        call_log: list = []

        def recording_run(*args, **kwargs):
            call_log.append(args[0] if args else kwargs.get("args"))
            return always_ok

        with patch.object(mod.subprocess, "run", side_effect=recording_run):
            mod.main()

        self.assertEqual(len(call_log), 3, "Expected exactly 3 subprocess calls")

    def test_cli_reference_check_flag_present(self) -> None:
        """The generate_cli_reference invocation must include --check."""
        mod = _load_module()
        always_ok = _make_completed_process(returncode=0)
        call_log: list = []

        def recording_run(*args, **kwargs):
            call_log.append(args[0] if args else kwargs.get("args"))
            return always_ok

        with patch.object(mod.subprocess, "run", side_effect=recording_run):
            mod.main()

        ref_cmd = call_log[0]
        self.assertIn("--check", ref_cmd)
        self.assertIn("generate_cli_reference.py", str(ref_cmd))


if __name__ == "__main__":
    unittest.main()
