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
from unittest.mock import MagicMock, patch

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
    """Verify that ImportError raised by a command handler is caught
    gracefully and returns exit code 1 instead of propagating as a traceback."""

    def _make_dispatch_context(self, action="help"):
        """Return a minimal parsed namespace that resolves to the given action."""
        from aura_cli.cli_options import parse_cli_args
        return parse_cli_args([action])

    def test_dispatch_command_catches_import_error_from_handler(self):
        """A handler that raises ImportError must return 1 and print to stderr."""
        from aura_cli.cli_main import dispatch_command, COMMAND_DISPATCH_REGISTRY, DispatchRule

        parsed = self._make_dispatch_context("help")

        # Patch the help handler to raise an ImportError (simulates _MissingPackage)
        original_rule = COMMAND_DISPATCH_REGISTRY.get("help")
        self.assertIsNotNone(original_rule)

        mock_handler = MagicMock(side_effect=ImportError(
            "Package 'numpy' is required but not installed. Install it with: pip install numpy"
        ))
        patched_rule = DispatchRule(
            action=original_rule.action,
            requires_runtime=original_rule.requires_runtime,
            handler=mock_handler,
        )

        with patch.dict(COMMAND_DISPATCH_REGISTRY, {"help": patched_rule}):
            stderr_capture = io.StringIO()
            with patch("sys.stderr", stderr_capture):
                rc = dispatch_command(parsed, project_root=REPO_ROOT)

        self.assertEqual(rc, 1)
        self.assertIn("numpy", stderr_capture.getvalue())

    def test_main_catches_import_error_from_dispatch_command(self):
        """main() must return 1 and print to stderr when dispatch_command raises ImportError."""
        from aura_cli.cli_main import main

        err_msg = "Package 'requests' is required but not installed. Install it with: pip install requests"

        stderr_capture = io.StringIO()
        with patch("aura_cli.cli_main.dispatch_command", side_effect=ImportError(err_msg)), \
             patch("sys.stderr", stderr_capture):
            rc = main(project_root_override=str(REPO_ROOT), argv=["help"])

        self.assertEqual(rc, 1)
        self.assertIn("requests", stderr_capture.getvalue())

    def test_main_returns_import_error_as_json_when_json_flag_set(self):
        """main() with --json flag must return JSON-encoded error on ImportError."""
        from aura_cli.cli_main import main

        err_msg = "Package 'numpy' is required but not installed."

        stdout_capture = io.StringIO()
        # Use an argv where --json is parsed by the subcommand (goal add ... --json)
        with patch("aura_cli.cli_main.dispatch_command", side_effect=ImportError(err_msg)), \
             patch("sys.stdout", stdout_capture):
            rc = main(project_root_override=str(REPO_ROOT), argv=["goal", "add", "my goal", "--json"])

        self.assertEqual(rc, 1)
        payload = json.loads(stdout_capture.getvalue())
        self.assertIn("error", payload)
        self.assertIn("numpy", payload["error"])
