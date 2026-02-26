import unittest

from aura_cli.cli_options import cli_contract_report, parse_cli_args
import aura_cli.cli_main as cli_main


class TestCLIContract(unittest.TestCase):
    def test_help_specs_cover_parser_commands(self):
        report = cli_contract_report()
        self.assertEqual(report["missing_in_specs"], [], f"Parser commands missing from help specs: {report['missing_in_specs']}")
        self.assertEqual(report["missing_in_parser"], [], f"Help specs missing from parser: {report['missing_in_parser']}")
        self.assertEqual(report["unknown_spec_lookups"], [])

    def test_all_canonical_commands_resolve_to_registered_dispatch_action(self):
        canonical_invocations = [
            ["bootstrap"],
            ["diag"],
            ["doctor"],
            ["help"],
            ["goal", "add", "x"],
            ["goal", "run"],
            ["goal", "status"],
            ["goal", "once", "x"],
            ["workflow", "run", "x"],
            ["mcp", "tools"],
            ["mcp", "call", "limits"],
            ["scaffold", "demo"],
            ["evolve"],
            ["watch"],
            ["logs"],
        ]
        for argv in canonical_invocations:
            with self.subTest(argv=argv):
                parsed = parse_cli_args(argv)
                action = cli_main._resolve_dispatch_action(parsed)  # internal contract test
                self.assertIn(action, cli_main.COMMAND_DISPATCH_REGISTRY)


if __name__ == "__main__":
    unittest.main()
