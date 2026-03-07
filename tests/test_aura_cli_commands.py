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
import aura_cli.cli_main as cli_main

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


class TestDispatchImportErrorHandling(TestCase):
    """Tests that verify ImportError from missing optional dependencies is caught
    at the dispatch boundary and results in a graceful error instead of a traceback."""

    def _make_parsed(self, action="help", json_mode=False):
        """Build a minimal parsed namespace for dispatch_command."""
        parsed = MagicMock()
        parsed.namespace = MagicMock()
        parsed.warning_records = []
        parsed.warnings = []
        parsed.namespace.json = json_mode
        # Make _resolve_dispatch_action return the desired action
        parsed.subcommand = action
        parsed.namespace.subcommand = action
        return parsed

    def test_dispatch_command_catches_import_error_from_handler(self):
        """When a handler raises ImportError, dispatch_command returns exit code 1."""
        parsed = self._make_parsed(action="doctor")

        def _raising_handler(_ctx):
            raise ImportError("No module named 'some_optional_dep'")

        with patch.object(cli_main, "COMMAND_DISPATCH_REGISTRY", {
            "doctor": cli_main._dispatch_rule("doctor", _raising_handler),
        }), patch("aura_cli.cli_main._resolve_dispatch_action", return_value="doctor"):
            rc = cli_main.dispatch_command(
                parsed,
                project_root=Path("."),
            )

        self.assertEqual(rc, 1)

    def test_dispatch_command_prints_error_message_on_import_error(self):
        """When ImportError is raised, an informative message is written to stderr."""
        parsed = self._make_parsed(action="doctor")

        def _raising_handler(_ctx):
            raise ImportError("No module named 'some_optional_dep'")

        with patch.object(cli_main, "COMMAND_DISPATCH_REGISTRY", {
            "doctor": cli_main._dispatch_rule("doctor", _raising_handler),
        }), patch("aura_cli.cli_main._resolve_dispatch_action", return_value="doctor"), \
             patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            cli_main.dispatch_command(parsed, project_root=Path("."))
            err_output = mock_stderr.getvalue()

        self.assertIn("some_optional_dep", err_output)
        self.assertIn("pip install", err_output)

    def test_dispatch_command_json_mode_outputs_json_on_import_error(self):
        """In JSON mode, ImportError produces a JSON error object on stdout."""
        parsed = self._make_parsed(action="doctor", json_mode=True)

        def _raising_handler(_ctx):
            raise ImportError("No module named 'missing_pkg'")

        with patch.object(cli_main, "COMMAND_DISPATCH_REGISTRY", {
            "doctor": cli_main._dispatch_rule("doctor", _raising_handler),
        }), patch("aura_cli.cli_main._resolve_dispatch_action", return_value="doctor"), \
             patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            rc = cli_main.dispatch_command(parsed, project_root=Path("."))
            out = mock_stdout.getvalue()

        self.assertEqual(rc, 1)
        payload = json.loads(out)
        self.assertIn("error", payload)
        self.assertIn("missing_pkg", payload["error"])
        self.assertEqual(payload["type"], "missing_dependency")

    def test_dispatch_command_does_not_catch_non_import_errors(self):
        """Non-ImportError exceptions propagate normally from the handler."""
        parsed = self._make_parsed(action="doctor")

        def _raising_handler(_ctx):
            raise ValueError("unexpected value error")

        with patch.object(cli_main, "COMMAND_DISPATCH_REGISTRY", {
            "doctor": cli_main._dispatch_rule("doctor", _raising_handler),
        }), patch("aura_cli.cli_main._resolve_dispatch_action", return_value="doctor"):
            with self.assertRaises(ValueError):
                cli_main.dispatch_command(parsed, project_root=Path("."))
