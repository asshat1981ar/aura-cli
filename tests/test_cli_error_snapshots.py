import json
import unittest

import aura_cli.options as cli_options_meta
from tests.cli_entrypoint_test_utils import run_main_subprocess
from tests.cli_snapshot_utils import normalized_json_text, read_snapshot_json, read_snapshot_text, snapshot_dir_for

SNAPSHOT_DIR = snapshot_dir_for(__file__)


class TestCLIErrorSnapshots(unittest.TestCase):
    _JSON_ERROR_SNAPSHOT_CONTRACTS = (
        ("cli_error_unknown_command.json", cli_options_meta.CLI_PARSE_ERROR_CODE, True),
        ("cli_error_unknown_help_topic.json", cli_options_meta.UNKNOWN_COMMAND_HELP_TOPIC_CODE, False),
        ("cli_error_mixed_subcommand_legacy.json", cli_options_meta.CLI_PARSE_ERROR_CODE, True),
        ("cli_error_conflicting_legacy_actions.json", cli_options_meta.CLI_PARSE_ERROR_CODE, True),
        ("cli_error_unrecognized_subcommand_argument.json", cli_options_meta.CLI_PARSE_ERROR_CODE, True),
        ("cli_error_missing_goal_subcommand.json", cli_options_meta.CLI_PARSE_ERROR_CODE, True),
        ("cli_error_missing_mcp_subcommand.json", cli_options_meta.CLI_PARSE_ERROR_CODE, True),
        ("cli_error_missing_workflow_subcommand.json", cli_options_meta.CLI_PARSE_ERROR_CODE, True),
    )
    _TEXT_ERROR_SNAPSHOT_CONTRACTS = (
        ("cli_error_unknown_command.txt", True),
        ("cli_error_unknown_help_topic.txt", False),
        ("cli_error_mixed_subcommand_legacy.txt", True),
        ("cli_error_conflicting_legacy_actions.txt", True),
        ("cli_error_unrecognized_subcommand_argument.txt", True),
        ("cli_error_missing_goal_subcommand.txt", True),
        ("cli_error_missing_mcp_subcommand.txt", True),
        ("cli_error_missing_workflow_subcommand.txt", True),
    )

    def test_unknown_command_text_error_matches_snapshot(self):
        proc = run_main_subprocess("goa")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, read_snapshot_text(SNAPSHOT_DIR, "cli_error_unknown_command.txt"))

    def test_unknown_command_json_error_matches_snapshot(self):
        proc = run_main_subprocess("goa", "--json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stderr, "")
        actual = normalized_json_text(proc.stdout)
        self.assertEqual(actual, read_snapshot_text(SNAPSHOT_DIR, "cli_error_unknown_command.json"))

    def test_unknown_help_topic_text_error_matches_snapshot(self):
        proc = run_main_subprocess("help", "nope")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, read_snapshot_text(SNAPSHOT_DIR, "cli_error_unknown_help_topic.txt"))

    def test_unknown_help_topic_json_error_matches_snapshot(self):
        proc = run_main_subprocess("help", "nope", "--json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stderr, "")
        actual = normalized_json_text(proc.stdout)
        self.assertEqual(actual, read_snapshot_text(SNAPSHOT_DIR, "cli_error_unknown_help_topic.json"))

    def test_mixed_canonical_and_legacy_text_error_matches_snapshot(self):
        proc = run_main_subprocess("--run-goals", "goal", "status")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, read_snapshot_text(SNAPSHOT_DIR, "cli_error_mixed_subcommand_legacy.txt"))

    def test_mixed_canonical_and_legacy_json_error_matches_snapshot(self):
        proc = run_main_subprocess("--run-goals", "goal", "status", "--json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stderr, "")
        actual = normalized_json_text(proc.stdout)
        self.assertEqual(actual, read_snapshot_text(SNAPSHOT_DIR, "cli_error_mixed_subcommand_legacy.json"))

    def test_conflicting_legacy_actions_text_error_matches_snapshot(self):
        proc = run_main_subprocess("--diag", "--status")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, read_snapshot_text(SNAPSHOT_DIR, "cli_error_conflicting_legacy_actions.txt"))

    def test_conflicting_legacy_actions_json_error_matches_snapshot(self):
        proc = run_main_subprocess("--diag", "--status", "--json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stderr, "")
        actual = normalized_json_text(proc.stdout)
        self.assertEqual(actual, read_snapshot_text(SNAPSHOT_DIR, "cli_error_conflicting_legacy_actions.json"))

    def test_canonical_subcommand_with_unsupported_legacy_flag_text_error_matches_snapshot(self):
        proc = run_main_subprocess("goal", "status", "--run-goals")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, read_snapshot_text(SNAPSHOT_DIR, "cli_error_unrecognized_subcommand_argument.txt"))

    def test_canonical_subcommand_with_unsupported_legacy_flag_json_error_matches_snapshot(self):
        proc = run_main_subprocess("goal", "status", "--run-goals", "--json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stderr, "")
        actual = normalized_json_text(proc.stdout)
        self.assertEqual(actual, read_snapshot_text(SNAPSHOT_DIR, "cli_error_unrecognized_subcommand_argument.json"))

    def test_missing_goal_subcommand_text_error_matches_snapshot(self):
        proc = run_main_subprocess("goal")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, read_snapshot_text(SNAPSHOT_DIR, "cli_error_missing_goal_subcommand.txt"))

    def test_missing_goal_subcommand_json_error_matches_snapshot(self):
        proc = run_main_subprocess("goal", "--json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stderr, "")
        actual = normalized_json_text(proc.stdout)
        self.assertEqual(actual, read_snapshot_text(SNAPSHOT_DIR, "cli_error_missing_goal_subcommand.json"))

    def test_missing_mcp_subcommand_text_error_matches_snapshot(self):
        proc = run_main_subprocess("mcp")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, read_snapshot_text(SNAPSHOT_DIR, "cli_error_missing_mcp_subcommand.txt"))

    def test_missing_mcp_subcommand_json_error_matches_snapshot(self):
        proc = run_main_subprocess("mcp", "--json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stderr, "")
        actual = normalized_json_text(proc.stdout)
        self.assertEqual(actual, read_snapshot_text(SNAPSHOT_DIR, "cli_error_missing_mcp_subcommand.json"))

    def test_missing_workflow_subcommand_text_error_matches_snapshot(self):
        proc = run_main_subprocess("workflow")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, read_snapshot_text(SNAPSHOT_DIR, "cli_error_missing_workflow_subcommand.txt"))

    def test_missing_workflow_subcommand_json_error_matches_snapshot(self):
        proc = run_main_subprocess("workflow", "--json")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stderr, "")
        actual = normalized_json_text(proc.stdout)
        self.assertEqual(actual, read_snapshot_text(SNAPSHOT_DIR, "cli_error_missing_workflow_subcommand.json"))

    def test_json_error_snapshots_follow_documented_contract(self):
        for snapshot_name, expected_code, expects_usage in self._JSON_ERROR_SNAPSHOT_CONTRACTS:
            with self.subTest(snapshot=snapshot_name):
                payload = read_snapshot_json(SNAPSHOT_DIR, snapshot_name)

                self.assertEqual(payload["status"], "error")
                self.assertEqual(payload["code"], expected_code)
                self.assertIsInstance(payload["message"], str)
                self.assertTrue(payload["message"])
                self.assertNotIn("cli_warnings", payload)

                if expects_usage:
                    self.assertIn("usage", payload)
                    self.assertTrue(payload["usage"].startswith("usage: aura "))
                else:
                    self.assertNotIn("usage", payload)

    def test_text_error_snapshots_follow_text_contract(self):
        for snapshot_name, expects_usage in self._TEXT_ERROR_SNAPSHOT_CONTRACTS:
            with self.subTest(snapshot=snapshot_name):
                payload = read_snapshot_text(SNAPSHOT_DIR, snapshot_name)

                self.assertTrue(payload.startswith("Error: "))
                self.assertTrue(payload.endswith("\n"))

                if expects_usage:
                    self.assertIn("\nusage: aura", payload)
                else:
                    self.assertNotIn("\nusage: aura", payload)


if __name__ == "__main__":
    unittest.main()
