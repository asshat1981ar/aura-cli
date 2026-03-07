"""
Smoke tests for the main CLI entrypoint.

These tests are intentionally light and do not duplicate the comprehensive
snapshot-based test suites. Their primary purpose is to confirm that the
`python3 main.py ...` entrypoint is wired correctly and that basic
invocations succeed with a status code of 0.
"""
import io
import json
import os
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch, MagicMock

# Add repo root to path to allow importing from `tests`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.cli_entrypoint_test_utils import run_main_subprocess

# Set AURA_SKIP_CHDIR to prevent chdir to a tmp dir, which breaks imports
os.environ["AURA_SKIP_CHDIR"] = "1"


class TestCLIEntrypointSmoke(TestCase):
    def test_main_help_smoke_test(self):
        proc = run_main_subprocess("help")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("AURA CLI", proc.stdout)

    def test_main_json_help_smoke_test(self):
        proc = run_main_subprocess("--json-help")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        
        # Assert that the output is valid JSON and contains top-level keys
        payload = json.loads(proc.stdout)
        self.assertIn("commands", payload)
        self.assertIn("json_contracts", payload)


class TestDispatchCommandImportErrorHandling(TestCase):
    """Tests for ImportError handling at the dispatch boundary."""

    def _make_failing_rule(self, action: str, err_msg: str):
        """Create a DispatchRule whose handler raises ImportError."""
        from aura_cli.cli_main import DispatchRule

        def _raising_handler(_ctx):
            raise ImportError(err_msg)

        return DispatchRule(action, False, _raising_handler)

    def test_dispatch_command_catches_import_error_from_handler_and_returns_1(self):
        """dispatch_command() catches ImportError from a handler and returns exit code 1."""
        from aura_cli.cli_main import dispatch_command, COMMAND_DISPATCH_REGISTRY
        from aura_cli.cli_options import parse_cli_args

        err_msg = "Package 'numpy' is required but not installed."
        parsed = parse_cli_args(["help"])
        stderr_capture = io.StringIO()

        failing_registry = dict(COMMAND_DISPATCH_REGISTRY)
        failing_registry["help"] = self._make_failing_rule("help", err_msg)

        with patch("aura_cli.cli_main.COMMAND_DISPATCH_REGISTRY", failing_registry), \
             patch("sys.stderr", stderr_capture):
            rc = dispatch_command(parsed, project_root=REPO_ROOT)

        self.assertEqual(rc, 1)
        self.assertIn(err_msg, stderr_capture.getvalue())

    def test_dispatch_command_reraises_import_error_when_json_flag_set(self):
        """dispatch_command() re-raises ImportError when --json is set so main() can emit JSON."""
        from aura_cli.cli_main import dispatch_command, COMMAND_DISPATCH_REGISTRY
        from aura_cli.cli_options import parse_cli_args

        err_msg = "Package 'requests' is required but not installed."
        parsed = parse_cli_args(["help", "--json"])

        failing_registry = dict(COMMAND_DISPATCH_REGISTRY)
        failing_registry["help"] = self._make_failing_rule("help", err_msg)

        with patch("aura_cli.cli_main.COMMAND_DISPATCH_REGISTRY", failing_registry):
            with self.assertRaises(ImportError):
                dispatch_command(parsed, project_root=REPO_ROOT)

    def test_main_returns_json_error_on_import_error_when_json_flag_set(self):
        """main() emits a JSON error payload matching the CLI contract when --json is set and a handler raises ImportError."""
        from aura_cli.cli_main import main, COMMAND_DISPATCH_REGISTRY

        err_msg = "Package 'numpy' is required but not installed."
        stdout_capture = io.StringIO()

        failing_registry = dict(COMMAND_DISPATCH_REGISTRY)
        failing_registry["help"] = self._make_failing_rule("help", err_msg)

        with patch("aura_cli.cli_main.COMMAND_DISPATCH_REGISTRY", failing_registry), \
             patch("sys.stdout", stdout_capture):
            rc = main(project_root_override=str(REPO_ROOT), argv=["help", "--json"])

        self.assertEqual(rc, 1)
        payload = json.loads(stdout_capture.getvalue().strip())
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["code"], "missing_dependency")
        self.assertEqual(payload["message"], err_msg)

    def test_main_returns_1_and_prints_to_stderr_on_import_error_without_json(self):
        """main() prints an error to stderr and returns 1 when a handler raises ImportError (no --json)."""
        from aura_cli.cli_main import main, COMMAND_DISPATCH_REGISTRY

        err_msg = "Package 'gitpython' is required but not installed."
        stderr_capture = io.StringIO()

        failing_registry = dict(COMMAND_DISPATCH_REGISTRY)
        failing_registry["help"] = self._make_failing_rule("help", err_msg)

        with patch("aura_cli.cli_main.COMMAND_DISPATCH_REGISTRY", failing_registry), \
             patch("sys.stderr", stderr_capture):
            rc = main(project_root_override=str(REPO_ROOT), argv=["help"])

        self.assertEqual(rc, 1)
        self.assertIn(err_msg, stderr_capture.getvalue())
