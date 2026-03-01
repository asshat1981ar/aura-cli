import json
import unittest

import aura_cli.options as cli_options_meta
from tests.cli_snapshot_utils import read_snapshot_json, snapshot_dir_for

SNAPSHOT_DIR = snapshot_dir_for(__file__)


class TestCLIWarningContracts(unittest.TestCase):
    _SCENARIOS = (
        {
            "action": "mcp_tools",
            "replacement_command": "aura mcp tools",
            "legacy_snapshot": "cli_legacy_mcp_tools_dispatch.json",
            "canonical_snapshot": "cli_canonical_mcp_tools_dispatch.json",
        },
        {
            "action": "diag",
            "replacement_command": "aura diag",
            "legacy_snapshot": "cli_legacy_diag_dispatch.json",
            "canonical_snapshot": "cli_canonical_diag_dispatch.json",
        },
        {
            "action": "mcp_call",
            "replacement_command": "aura mcp call",
            "legacy_snapshot": "cli_legacy_mcp_call_dispatch.json",
            "canonical_snapshot": "cli_canonical_mcp_call_dispatch.json",
        },
        {
            "action": "workflow_run",
            "replacement_command": "aura workflow run",
            "legacy_snapshot": "cli_legacy_workflow_run_dispatch.json",
            "canonical_snapshot": "cli_canonical_workflow_run_dispatch.json",
        },
        {
            "action": "evolve",
            "replacement_command": "aura evolve",
            "legacy_snapshot": "cli_legacy_evolve_dispatch.json",
            "canonical_snapshot": "cli_canonical_evolve_dispatch.json",
        },
        {
            "action": "goal_status",
            "replacement_command": "aura goal status",
            "legacy_snapshot": "cli_legacy_goal_status_dispatch.json",
            "canonical_snapshot": "cli_canonical_goal_status_dispatch.json",
        },
    )

    def _without_cli_warnings(self, payload: dict) -> dict:
        base = dict(payload)
        base.pop("cli_warnings", None)
        return base

    def test_legacy_warning_snapshot_contracts_match_canonical_payloads(self):
        for scenario in self._SCENARIOS:
            with self.subTest(action=scenario["action"]):
                legacy_payload = read_snapshot_json(SNAPSHOT_DIR, scenario["legacy_snapshot"])
                canonical_payload = read_snapshot_json(SNAPSHOT_DIR, scenario["canonical_snapshot"])

                self.assertIn("cli_warnings", legacy_payload)
                self.assertEqual(len(legacy_payload["cli_warnings"]), 1)
                warning = legacy_payload["cli_warnings"][0]

                self.assertEqual(warning["code"], cli_options_meta.CLI_WARNINGS_CODE_LEGACY_FLAGS_DEPRECATED)
                self.assertEqual(warning["category"], "deprecation")
                self.assertEqual(warning["phase"], "compatibility")
                self.assertEqual(warning["action"], scenario["action"])
                self.assertEqual(warning["replacement_command"], scenario["replacement_command"])
                self.assertTrue(warning["legacy_flags"])
                self.assertIn(scenario["replacement_command"], warning["message"])

                self.assertEqual(canonical_payload, self._without_cli_warnings(legacy_payload))
