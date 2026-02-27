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
    parser_parent_command_paths,
    parser_required_subcommand_parent_paths,
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

    def test_required_subcommand_parent_paths_match_non_leaf_command_paths(self):
        parent_paths = parser_parent_command_paths()
        required_paths = parser_required_subcommand_parent_paths()
        self.assertEqual(
            sorted(required_paths),
            sorted(parent_paths),
            "Required-subcommand parent paths must stay in sync with non-leaf command paths.",
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
        self.assertIn(cli_options_meta.CLI_WARNINGS_CODE_LEGACY_FLAGS_DEPRECATED, code_names)

    def test_help_schema_documents_cli_errors_json_contract(self):
        payload = cli_options_meta.help_schema()
        json_contracts = payload.get("json_contracts") or {}
        self.assertIn(cli_options_meta.CLI_ERRORS_JSON_CONTRACT_NAME, json_contracts)

        contract = json_contracts[cli_options_meta.CLI_ERRORS_JSON_CONTRACT_NAME]
        self.assertEqual(contract["version"], cli_options_meta.CLI_ERRORS_JSON_CONTRACT_VERSION)
        self.assertEqual(contract["record_fields"], list(cli_options_meta.CLI_ERRORS_RECORD_FIELDS))
        self.assertEqual(contract["optional_fields"], list(cli_options_meta.CLI_ERRORS_OPTIONAL_FIELDS))

        codes = contract.get("record_codes") or []
        self.assertEqual(
            {item["code"] for item in codes},
            {
                cli_options_meta.CLI_PARSE_ERROR_CODE,
                cli_options_meta.UNKNOWN_COMMAND_HELP_TOPIC_CODE,
            },
        )

    def test_action_specs_canonical_paths_map_to_command_specs(self):
        for spec in cli_options_meta.CLI_ACTION_SPECS:
            if spec.canonical_path is None:
                continue
            with self.subTest(action=spec.action):
                self.assertIn(spec.canonical_path, cli_options_meta.COMMAND_SPECS_BY_PATH)

    def test_command_spec_legacy_flags_are_known_legacy_options(self):
        known_flags = {
            *{f"--{name.replace('_', '-')}" for name in cli_options_meta.legacy_primary_flag_names()},
            *{f"--{name.replace('_', '-')}" for name in cli_options_meta.legacy_auxiliary_flag_names()},
        }
        for spec in cli_options_meta.COMMAND_SPECS:
            for legacy_flag in spec.legacy_flags:
                with self.subTest(path=spec.path, flag=legacy_flag):
                    self.assertIn(legacy_flag, known_flags)

    def test_action_spec_legacy_flags_are_documented_on_canonical_command(self):
        for action_spec in cli_options_meta.CLI_ACTION_SPECS:
            if action_spec.canonical_path is None or not action_spec.legacy_primary_flags:
                continue

            command_spec = cli_options_meta.COMMAND_SPECS_BY_PATH[action_spec.canonical_path]
            documented_flags = set(command_spec.legacy_flags)
            expected_flags = {f"--{name.replace('_', '-')}" for name in action_spec.legacy_primary_flags}

            with self.subTest(action=action_spec.action):
                self.assertTrue(
                    expected_flags.issubset(documented_flags),
                    f"Missing documented legacy flags for action '{action_spec.action}': "
                    f"expected {sorted(expected_flags)}, saw {sorted(documented_flags)}",
                )

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

    def test_leaf_command_examples_resolve_to_documented_actions(self):
        action_by_path = {
            tuple(item["path"]): item.get("action")
            for item in cli_options_meta.help_schema().get("commands", [])
        }
        additional_allowed_actions_by_path = {
            ("goal", "add"): {"goal_add_run"},
        }

        for spec in cli_options_meta.COMMAND_SPECS:
            documented_action = action_by_path.get(spec.path)
            if documented_action is None:
                continue

            allowed_actions = {documented_action}
            allowed_actions.update(additional_allowed_actions_by_path.get(spec.path, set()))

            for example in spec.examples:
                with self.subTest(path=spec.path, example=example):
                    argv = self._argv_from_example(example)
                    parsed = parse_cli_args(argv)
                    resolved_action = cli_main._resolve_dispatch_action(parsed)
                    self.assertIn(
                        resolved_action,
                        allowed_actions,
                        f"Example '{example}' resolved to '{resolved_action}' but expected one of {sorted(allowed_actions)}.",
                    )


if __name__ == "__main__":
    unittest.main()
