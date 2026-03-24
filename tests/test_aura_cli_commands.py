"""
Smoke tests for the main CLI entrypoint.

These tests are intentionally light and do not duplicate the comprehensive
snapshot-based test suites. Their primary purpose is to confirm that the
`python3 main.py ...` entrypoint is wired correctly and that basic
invocations succeed with a status code of 0.
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import TestCase

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

    def test_main_json_runtime_failures_preserve_structured_error_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sitecustomize = Path(tmpdir) / "sitecustomize.py"
            sitecustomize.write_text(
                "\n".join(
                    (
                        "import sys",
                        "import types",
                        "",
                        "module = types.ModuleType('aura_cli.cli_main')",
                        "",
                        "def main(argv=None):",
                        "    raise RuntimeError('boom')",
                        "",
                        "module.main = main",
                        "sys.modules['aura_cli.cli_main'] = module",
                    )
                ),
                encoding="utf-8",
            )

            env_pythonpath = tmpdir
            if existing := os.environ.get("PYTHONPATH"):
                env_pythonpath = os.pathsep.join((tmpdir, existing))

            proc = run_main_subprocess(
                "goal",
                "status",
                "--json",
                env_overrides={"PYTHONPATH": env_pythonpath},
            )

        self.assertEqual(proc.returncode, 1)
        self.assertEqual(proc.stderr, "")

        payload = json.loads(proc.stdout)
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["code"], "unexpected_runtime_error")
        self.assertEqual(payload["message"], "boom")
