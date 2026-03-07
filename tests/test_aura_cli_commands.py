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
    def test_dispatch_command_returns_1_on_import_error(self):
        """dispatch_command returns 1 and prints a message when handler raises ImportError."""
        from aura_cli.cli_main import dispatch_command, COMMAND_DISPATCH_REGISTRY

        parsed = MagicMock()
        parsed.action = "help"
        parsed.warning_records = []
        parsed.warnings = []
        parsed.namespace = MagicMock()

        failing_handler = MagicMock(side_effect=ImportError("fake_missing_module"))
        fake_rule = MagicMock(requires_runtime=False, handler=failing_handler)

        stderr_capture = io.StringIO()
        with patch.dict(COMMAND_DISPATCH_REGISTRY, {"help": fake_rule}):
            with patch("sys.stderr", stderr_capture):
                rc = dispatch_command(parsed, project_root=Path.cwd())

        self.assertEqual(rc, 1)
        self.assertIn("fake_missing_module", stderr_capture.getvalue())

    def test_main_returns_1_on_import_error(self):
        """main() returns 1 when dispatch_command raises ImportError."""
        from aura_cli.cli_main import main

        with patch("aura_cli.cli_main.dispatch_command", side_effect=ImportError("missing_pkg")):
            stderr_capture = io.StringIO()
            with patch("sys.stderr", stderr_capture):
                rc = main(argv=["help"])

        self.assertEqual(rc, 1)
        self.assertIn("missing_pkg", stderr_capture.getvalue())

    def test_main_json_error_on_import_error(self):
        """main() emits JSON error payload when --json flag is present and ImportError occurs."""
        from aura_cli.cli_main import main

        with patch("aura_cli.cli_main.dispatch_command", side_effect=ImportError("missing_pkg")):
            stdout_capture = io.StringIO()
            with patch("sys.stdout", stdout_capture):
                rc = main(argv=["help", "--json"])

        self.assertEqual(rc, 1)
        payload = json.loads(stdout_capture.getvalue())
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["code"], "missing_dependency")
        self.assertIn("missing_pkg", payload["message"])
