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


if __name__ == "__main__":
    unittest.main()
