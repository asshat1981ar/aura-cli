import json
import unittest

from tests.cli_entrypoint_test_utils import run_main_subprocess
from tests.cli_snapshot_utils import normalized_json_text, read_snapshot_text, snapshot_dir_for

SNAPSHOT_DIR = snapshot_dir_for(__file__)


class TestCLIHelpSnapshots(unittest.TestCase):
    _HELP_TEXT_SNAPSHOT_CONTRACTS = (
        ("cli_help_top_level.txt", "AURA CLI", True, False, True),
        ("cli_help_goal_add.txt", "aura goal add", True, True, False),
        ("cli_help_watch.txt", "aura watch", True, False, False),
        ("cli_help_studio.txt", "aura studio", True, False, False),
        ("cli_help_config.txt", "aura config", True, False, False),
        ("cli_help_contract_report.txt", "aura contract-report", True, False, False),
    )
    _JSON_HELP_COMMAND_CONTRACTS = (
        ("help", "help", False, 2),
        ("config", "show_config", False, 1),
        ("contract-report", "contract_report", False, 2),
        ("watch", "watch", True, 2),
        ("studio", "studio", True, 2),
        ("goal add", "goal_add", True, 2),
        ("goal run", "goal_run", True, 1),
        ("goal status", "goal_status", True, 1),
        ("workflow run", "workflow_run", True, 1),
        ("mcp tools", "mcp_tools", False, 1),
    )

    def test_main_help_matches_snapshot(self):
        proc = run_main_subprocess("help")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(proc.stdout, read_snapshot_text(SNAPSHOT_DIR, "cli_help_top_level.txt"))

    def test_main_help_goal_add_matches_snapshot(self):
        proc = run_main_subprocess("help", "goal", "add")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(proc.stdout, read_snapshot_text(SNAPSHOT_DIR, "cli_help_goal_add.txt"))

    def test_main_help_watch_matches_snapshot(self):
        proc = run_main_subprocess("help", "watch")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(proc.stdout, read_snapshot_text(SNAPSHOT_DIR, "cli_help_watch.txt"))

    def test_main_help_studio_matches_snapshot(self):
        proc = run_main_subprocess("help", "studio")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(proc.stdout, read_snapshot_text(SNAPSHOT_DIR, "cli_help_studio.txt"))

    def test_main_help_config_matches_snapshot(self):
        proc = run_main_subprocess("help", "config")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(proc.stdout, read_snapshot_text(SNAPSHOT_DIR, "cli_help_config.txt"))

    def test_main_help_contract_report_matches_snapshot(self):
        proc = run_main_subprocess("help", "contract-report")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(proc.stdout, read_snapshot_text(SNAPSHOT_DIR, "cli_help_contract_report.txt"))

    def test_main_json_help_matches_snapshot(self):
        proc = run_main_subprocess("--json-help")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")

        actual = normalized_json_text(proc.stdout)
        expected = read_snapshot_text(SNAPSHOT_DIR, "cli_json_help.json")
        self.assertEqual(actual, expected)

    def test_json_help_snapshot_follows_command_contract(self):
        payload = json.loads(read_snapshot_text(SNAPSHOT_DIR, "cli_json_help.json"))

        self.assertIn("commands", payload)
        self.assertIn("json_contracts", payload)
        self.assertTrue(payload.get("deterministic"))

        command_map = {
            " ".join(item["path"]): item
            for item in payload["commands"]
        }

        self.assertIn("cli_errors", payload["json_contracts"])
        self.assertIn("cli_warnings", payload["json_contracts"])

        for path_label, expected_action, expected_requires_runtime, expected_example_count in self._JSON_HELP_COMMAND_CONTRACTS:
            with self.subTest(path=path_label):
                item = command_map[path_label]
                self.assertEqual(item["action"], expected_action)
                self.assertEqual(item["requires_runtime"], expected_requires_runtime)
                self.assertEqual(len(item["examples"]), expected_example_count)
                self.assertIsInstance(item["description"], str)
                self.assertTrue(item["description"])

    def test_help_text_snapshots_follow_text_contract(self):
        for snapshot_name, expected_header, expects_examples, expects_legacy_flags, expects_commands in self._HELP_TEXT_SNAPSHOT_CONTRACTS:
            with self.subTest(snapshot=snapshot_name):
                payload = read_snapshot_text(SNAPSHOT_DIR, snapshot_name)

                self.assertTrue(payload.startswith(expected_header + "\n"))
                self.assertTrue(payload.endswith("\n"))

                if expects_examples:
                    self.assertIn("\nExamples:\n", payload)
                else:
                    self.assertNotIn("\nExamples:\n", payload)

                if expects_legacy_flags:
                    self.assertIn("\nLegacy flags:\n", payload)
                else:
                    self.assertNotIn("\nLegacy flags:\n", payload)

                if expects_commands:
                    self.assertIn("\nCommands:\n", payload)
                else:
                    self.assertNotIn("\nCommands:\n", payload)


if __name__ == "__main__":
    unittest.main()
