import json
import shlex
from dataclasses import fields
import unittest

import aura_cli.cli_options as cli_parser_options
from aura_cli.cli_options import (
    cli_contract_report,
    parse_cli_args,
    parser_customizer_paths,
    parser_leaf_command_paths,
)
import aura_cli.cli_main as cli_main
import aura_cli.options as cli_options_meta


class TestCLIContract(unittest.TestCase):
    def _argv_from_example(self, example: str) -> list[str]:
        tokens = shlex.split(example)
        self.assertGreaterEqual(len(tokens), 3, f"Example too short: {example}")
        self.assertEqual(tokens[0], "python3", f"Example must start with 'python3': {example}")
        self.assertEqual(tokens[1], "main.py", f"Example must target main.py: {example}")
        return tokens[2:]

    def test_help_specs_cover_parser_commands(self):
        report = cli_contract_report()
        self.assertEqual(report["missing_in_specs"], [], f"Parser commands missing from help specs: {report['missing_in_specs']}")
        self.assertEqual(report["missing_in_parser"], [], f"Help specs missing from parser: {report['missing_in_parser']}")
        self.assertEqual(report["unknown_spec_lookups"], [])

    def test_all_actions_have_smoke_invocation_contracts(self):
        action_names = set(cli_options_meta.CLI_ACTION_SPECS_BY_ACTION)
        for action in action_names:
            with self.subTest(action=action):
                argv = cli_options_meta.action_smoke_argv(action)
                self.assertIsInstance(argv, tuple)

    def test_smoke_invocations_resolve_to_registered_dispatch_actions(self):
        for action in sorted(cli_options_meta.CLI_ACTION_SPECS_BY_ACTION):
            with self.subTest(action=action):
                argv = list(cli_options_meta.action_smoke_argv(action))
                parsed = parse_cli_args(argv)
                resolved = cli_main._resolve_dispatch_action(parsed)  # internal contract test
                self.assertEqual(resolved, action)
                self.assertIn(resolved, cli_main.COMMAND_DISPATCH_REGISTRY)

    def test_parser_customizers_reference_only_leaf_command_paths(self):
        customizer_paths = parser_customizer_paths()
        leaf_paths = parser_leaf_command_paths()
        self.assertEqual(
            sorted(customizer_paths - leaf_paths),
            [],
            "Parser customizers must reference leaf command paths only.",
        )

    def test_actions_with_positional_smoke_args_have_parser_customizers(self):
        customizer_paths = parser_customizer_paths()
        for action, spec in cli_options_meta.CLI_ACTION_SPECS_BY_ACTION.items():
            canonical_path = spec.canonical_path
            if canonical_path is None:
                continue

            smoke_argv = cli_options_meta.action_smoke_argv(action)
            tail_tokens = smoke_argv[len(canonical_path):]
            has_positional_tail = any(token and not token.startswith("-") for token in tail_tokens)
            if has_positional_tail:
                self.assertIn(
                    canonical_path,
                    customizer_paths,
                    f"Action '{action}' has positional smoke args but no parser customizer for {canonical_path}.",
                )

    def test_help_schema_has_versioning_metadata_and_is_deterministic(self):
        first = cli_options_meta.help_schema()
        second = cli_options_meta.help_schema()

        self.assertEqual(first["version"], cli_options_meta.HELP_SCHEMA_VERSION)
        self.assertEqual(first["generated_by"], cli_options_meta.HELP_SCHEMA_GENERATED_BY)
        self.assertTrue(first["deterministic"])
        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )

    def test_help_schema_action_metadata_matches_action_specs(self):
        payload = cli_options_meta.help_schema()

        paths = [tuple(item["path"]) for item in payload["commands"]]
        self.assertEqual(len(paths), len(set(paths)), "Duplicate help schema command paths found")

        for item in payload["commands"]:
            action = item.get("action")
            requires_runtime = item.get("requires_runtime")
            if action is None:
                self.assertIsNone(requires_runtime)
                continue

            self.assertIn(action, cli_options_meta.CLI_ACTION_SPECS_BY_ACTION)
            self.assertEqual(requires_runtime, cli_options_meta.action_runtime_required(action))

    def test_help_schema_documents_cli_warnings_json_contract(self):
        payload = cli_options_meta.help_schema()
        json_contracts = payload.get("json_contracts") or {}
        self.assertIn(cli_options_meta.CLI_WARNINGS_JSON_FIELD, json_contracts)

        contract = json_contracts[cli_options_meta.CLI_WARNINGS_JSON_FIELD]
        self.assertEqual(contract["field"], cli_options_meta.CLI_WARNINGS_JSON_FIELD)
        self.assertEqual(contract["version"], cli_options_meta.CLI_WARNINGS_JSON_CONTRACT_VERSION)

        expected_fields = [f.name for f in fields(cli_parser_options.CLIWarningRecord)]
        self.assertEqual(contract["record_fields"], expected_fields)

        codes = contract.get("record_codes") or []
        self.assertTrue(codes, "Expected at least one documented cli_warnings record code")
        code_names = {item["code"] for item in codes}
        self.assertIn("legacy_cli_flags_deprecated", code_names)

    def test_action_specs_canonical_paths_map_to_command_specs(self):
        for spec in cli_options_meta.CLI_ACTION_SPECS:
            if spec.canonical_path is None:
                continue
            with self.subTest(action=spec.action):
                self.assertIn(spec.canonical_path, cli_options_meta.COMMAND_SPECS_BY_PATH)

    def test_dispatch_registry_matches_action_specs_and_runtime_flags(self):
        registry_actions = set(cli_main.COMMAND_DISPATCH_REGISTRY)
        action_spec_actions = set(cli_options_meta.CLI_ACTION_SPECS_BY_ACTION)
        self.assertEqual(registry_actions, action_spec_actions)

        for action, rule in cli_main.COMMAND_DISPATCH_REGISTRY.items():
            with self.subTest(action=action):
                self.assertEqual(rule.requires_runtime, cli_options_meta.action_runtime_required(action))

    def test_watch_and_studio_actions_share_dispatch_handler(self):
        watch_rule = cli_main.COMMAND_DISPATCH_REGISTRY["watch"]
        studio_rule = cli_main.COMMAND_DISPATCH_REGISTRY["studio"]
        self.assertIs(watch_rule.handler, studio_rule.handler)
        self.assertEqual(watch_rule.requires_runtime, studio_rule.requires_runtime)

    def test_command_spec_examples_parse_successfully(self):
        for spec in cli_options_meta.COMMAND_SPECS:
            for example in spec.examples:
                with self.subTest(path=spec.path, example=example):
                    argv = self._argv_from_example(example)
                    parse_cli_args(argv)


if __name__ == "__main__":
    unittest.main()
