import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"


def _run_main(*argv: str):
    env = os.environ.copy()
    env.setdefault("AURA_SKIP_CHDIR", "1")
    return subprocess.run(
        [sys.executable, "main.py", *argv],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


class TestCLIErrorSnapshots(unittest.TestCase):
    def _snapshot_text(self, name: str) -> str:
        return (SNAPSHOT_DIR / name).read_text(encoding="utf-8")

    def test_unknown_command_text_error_matches_snapshot(self):
        proc = _run_main("goa")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, self._snapshot_text("cli_error_unknown_command.txt"))

    def test_unknown_command_json_error_matches_snapshot(self):
        proc = _run_main("goa", "--json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stderr, "")
        actual = json.dumps(json.loads(proc.stdout), indent=2, sort_keys=True) + "\n"
        self.assertEqual(actual, self._snapshot_text("cli_error_unknown_command.json"))

    def test_unknown_help_topic_text_error_matches_snapshot(self):
        proc = _run_main("help", "nope")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, self._snapshot_text("cli_error_unknown_help_topic.txt"))

    def test_unknown_help_topic_json_error_matches_snapshot(self):
        proc = _run_main("help", "nope", "--json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stderr, "")
        actual = json.dumps(json.loads(proc.stdout), indent=2, sort_keys=True) + "\n"
        self.assertEqual(actual, self._snapshot_text("cli_error_unknown_help_topic.json"))

    def test_mixed_canonical_and_legacy_text_error_matches_snapshot(self):
        proc = _run_main("--run-goals", "goal", "status")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, self._snapshot_text("cli_error_mixed_subcommand_legacy.txt"))

    def test_mixed_canonical_and_legacy_json_error_matches_snapshot(self):
        proc = _run_main("--run-goals", "goal", "status", "--json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stderr, "")
        actual = json.dumps(json.loads(proc.stdout), indent=2, sort_keys=True) + "\n"
        self.assertEqual(actual, self._snapshot_text("cli_error_mixed_subcommand_legacy.json"))

    def test_conflicting_legacy_actions_text_error_matches_snapshot(self):
        proc = _run_main("--diag", "--status")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, self._snapshot_text("cli_error_conflicting_legacy_actions.txt"))

    def test_conflicting_legacy_actions_json_error_matches_snapshot(self):
        proc = _run_main("--diag", "--status", "--json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stderr, "")
        actual = json.dumps(json.loads(proc.stdout), indent=2, sort_keys=True) + "\n"
        self.assertEqual(actual, self._snapshot_text("cli_error_conflicting_legacy_actions.json"))

    def test_canonical_subcommand_with_unsupported_legacy_flag_text_error_matches_snapshot(self):
        proc = _run_main("goal", "status", "--run-goals")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, self._snapshot_text("cli_error_unrecognized_subcommand_argument.txt"))

    def test_canonical_subcommand_with_unsupported_legacy_flag_json_error_matches_snapshot(self):
        proc = _run_main("goal", "status", "--run-goals", "--json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stderr, "")
        actual = json.dumps(json.loads(proc.stdout), indent=2, sort_keys=True) + "\n"
        self.assertEqual(actual, self._snapshot_text("cli_error_unrecognized_subcommand_argument.json"))

    def test_missing_goal_subcommand_text_error_matches_snapshot(self):
        proc = _run_main("goal")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, self._snapshot_text("cli_error_missing_goal_subcommand.txt"))

    def test_missing_goal_subcommand_json_error_matches_snapshot(self):
        proc = _run_main("goal", "--json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stderr, "")
        actual = json.dumps(json.loads(proc.stdout), indent=2, sort_keys=True) + "\n"
        self.assertEqual(actual, self._snapshot_text("cli_error_missing_goal_subcommand.json"))

    def test_missing_mcp_subcommand_text_error_matches_snapshot(self):
        proc = _run_main("mcp")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, self._snapshot_text("cli_error_missing_mcp_subcommand.txt"))

    def test_missing_mcp_subcommand_json_error_matches_snapshot(self):
        proc = _run_main("mcp", "--json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stderr, "")
        actual = json.dumps(json.loads(proc.stdout), indent=2, sort_keys=True) + "\n"
        self.assertEqual(actual, self._snapshot_text("cli_error_missing_mcp_subcommand.json"))

    def test_missing_workflow_subcommand_text_error_matches_snapshot(self):
        proc = _run_main("workflow")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, self._snapshot_text("cli_error_missing_workflow_subcommand.txt"))

    def test_missing_workflow_subcommand_json_error_matches_snapshot(self):
        proc = _run_main("workflow", "--json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stderr, "")
        actual = json.dumps(json.loads(proc.stdout), indent=2, sort_keys=True) + "\n"
        self.assertEqual(actual, self._snapshot_text("cli_error_missing_workflow_subcommand.json"))


if __name__ == "__main__":
    unittest.main()
