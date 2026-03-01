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


class TestCLIHelpSnapshots(unittest.TestCase):
    def _snapshot_text(self, name: str) -> str:
        return (SNAPSHOT_DIR / name).read_text(encoding="utf-8")

    def test_main_help_matches_snapshot(self):
        proc = _run_main("help")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(proc.stdout, self._snapshot_text("cli_help_top_level.txt"))

    def test_main_help_goal_add_matches_snapshot(self):
        proc = _run_main("help", "goal", "add")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(proc.stdout, self._snapshot_text("cli_help_goal_add.txt"))

    def test_main_json_help_matches_snapshot(self):
        proc = _run_main("--json-help")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")

        actual = json.dumps(json.loads(proc.stdout), indent=2, sort_keys=True) + "\n"
        expected = self._snapshot_text("cli_json_help.json")
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
