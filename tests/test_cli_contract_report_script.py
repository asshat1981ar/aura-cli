import json
import unittest
from unittest.mock import patch

from tests.cli_contract_report_test_utils import (
    REPO_ROOT,
    compact_json_text,
    normalized_json_text,
    run_dispatch_with_report,
    run_main_main_with_report,
    run_main_subprocess,
    run_patched_main_subprocess,
    run_script_main,
    run_script_main_with_report,
    run_script_subprocess,
)
from tests.cli_snapshot_utils import read_snapshot_text, snapshot_dir_for
SNAPSHOT_DIR = snapshot_dir_for(__file__)


class TestCLIContractReportScript(unittest.TestCase):
    _CLEAN_REPORT = {"ok": True, "failure_keys": []}
    _DIRTY_REPORT = {
        "ok": False,
        "failure_keys": ["missing_in_help_schema"],
        "missing_in_help_schema": [["contract-report"]],
    }

    def test_script_main_prints_ok_report(self):
        code, out, err = run_script_main()

        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        payload = json.loads(out)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["failure_keys"], [])
        self.assertIn("missing_in_help_schema", payload)
        self.assertIn("smoke_dispatch_mismatches", payload)

    def test_script_check_output_matches_snapshot(self):
        proc = run_script_subprocess("--check")

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(
            normalized_json_text(proc.stdout),
            read_snapshot_text(SNAPSHOT_DIR, "cli_contract_report.json"),
        )

    def test_main_contract_report_check_output_matches_snapshot(self):
        proc = run_main_subprocess("contract-report", "--check")

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(
            normalized_json_text(proc.stdout),
            read_snapshot_text(SNAPSHOT_DIR, "cli_contract_report.json"),
        )

    def test_script_check_mode_returns_zero_when_clean(self):
        code, out, err = run_script_main("--check")

        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        payload = json.loads(out)
        self.assertTrue(payload["ok"])

    def test_script_compact_mode_emits_single_line_json(self):
        proc = run_script_subprocess("--compact")

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(proc.stdout.count("\n"), 1)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["ok"])

    def test_main_compact_mode_matches_compacted_snapshot(self):
        proc = run_main_subprocess("contract-report", "--compact")

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(proc.stdout, compact_json_text(read_snapshot_text(SNAPSHOT_DIR, "cli_contract_report.json")))

    def test_script_no_dispatch_mode_skips_dispatch_only_keys(self):
        code, out, _ = run_script_main("--no-dispatch")

        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertTrue(payload["ok"])
        self.assertNotIn("missing_in_dispatch", payload)
        self.assertNotIn("smoke_dispatch_mismatches", payload)

    def test_main_no_dispatch_mode_skips_dispatch_only_keys(self):
        proc = run_main_subprocess("contract-report", "--no-dispatch")

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["ok"])
        self.assertNotIn("missing_in_dispatch", payload)
        self.assertNotIn("smoke_dispatch_mismatches", payload)

    def test_script_check_mode_returns_nonzero_for_dirty_report(self):
        code, out, err = run_script_main_with_report(self._DIRTY_REPORT, "--check")

        self.assertEqual(code, 1)
        self.assertEqual(
            normalized_json_text(out),
            read_snapshot_text(SNAPSHOT_DIR, "cli_contract_report_dirty.json"),
        )
        self.assertEqual(err, "CLI contract failures: missing_in_help_schema\n")

    def test_script_compact_check_mode_returns_nonzero_for_dirty_report(self):
        code, out, err = run_script_main_with_report(self._DIRTY_REPORT, "--compact", "--check")

        self.assertEqual(code, 1)
        self.assertEqual(out, compact_json_text(read_snapshot_text(SNAPSHOT_DIR, "cli_contract_report_dirty.json")))
        self.assertEqual(err, "CLI contract failures: missing_in_help_schema\n")

    def test_main_check_mode_returns_nonzero_for_dirty_report_subprocess(self):
        proc = run_patched_main_subprocess(self._DIRTY_REPORT, "contract-report", "--check")

        self.assertEqual(proc.returncode, 1)
        self.assertEqual(proc.stderr, "CLI contract failures: missing_in_help_schema\n")
        self.assertEqual(
            normalized_json_text(proc.stdout),
            read_snapshot_text(SNAPSHOT_DIR, "cli_contract_report_dirty.json"),
        )

    def test_main_compact_check_mode_returns_nonzero_for_dirty_report_subprocess(self):
        proc = run_patched_main_subprocess(self._DIRTY_REPORT, "contract-report", "--compact", "--check")

        self.assertEqual(proc.returncode, 1)
        self.assertEqual(proc.stderr, "CLI contract failures: missing_in_help_schema\n")
        self.assertEqual(
            proc.stdout,
            compact_json_text(read_snapshot_text(SNAPSHOT_DIR, "cli_contract_report_dirty.json")),
        )

    def test_check_exit_codes_match_across_dispatch_script_and_main(self):
        scenarios = [
            ("clean", self._CLEAN_REPORT, 0, ""),
            ("dirty", self._DIRTY_REPORT, 1, "CLI contract failures: missing_in_help_schema\n"),
        ]
        runners = [
            ("dispatch", lambda report: run_dispatch_with_report(report, "contract-report", "--check")),
            ("script", lambda report: run_script_main_with_report(report, "--check")),
            ("main", lambda report: run_main_main_with_report(report, "contract-report", "--check")),
        ]

        for scenario_name, report, expected_code, expected_stderr in scenarios:
            for runner_name, runner in runners:
                with self.subTest(scenario=scenario_name, runner=runner_name):
                    code, out, err = runner(report)
                    self.assertEqual(code, expected_code)
                    self.assertEqual(err, expected_stderr)
                    payload = json.loads(out)
                    self.assertEqual(payload["ok"], report["ok"])


if __name__ == "__main__":
    unittest.main()
