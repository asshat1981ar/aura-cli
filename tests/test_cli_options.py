import json
import unittest

from aura_cli.cli_options import (
    CLI_PARSE_ERROR_CODE,
    CLIParseError,
    UNKNOWN_COMMAND_HELP_TOPIC_CODE,
    attach_cli_warnings,
    cli_parse_error_payload,
    parse_cli_args,
    render_help,
    unknown_command_help_topic_payload,
)


class TestCLIOptions(unittest.TestCase):
    def test_goal_add_subcommand_maps_to_legacy_fields(self):
        parsed = parse_cli_args(["goal", "add", "Refactor goal queue"])
        args = parsed.namespace

        self.assertEqual(parsed.command, "goal")
        self.assertEqual(parsed.subcommand, "add")
        self.assertTrue(parsed.uses_subcommand)
        self.assertFalse(parsed.legacy_invocation_used)
        self.assertEqual(args.add_goal, "Refactor goal queue")
        self.assertFalse(args.run_goals)

    def test_goal_add_run_maps_to_add_and_run(self):
        parsed = parse_cli_args(["goal", "add", "Refactor queue", "--run"])
        args = parsed.namespace

        self.assertEqual(parsed.command, "goal")
        self.assertEqual(parsed.subcommand, "add")
        self.assertEqual(args.add_goal, "Refactor queue")
        self.assertTrue(args.run_goals)

    def test_mcp_call_subcommand_maps_to_legacy_fields(self):
        parsed = parse_cli_args(["mcp", "call", "tail_logs", "--args", "{\"lines\": 10}"])
        args = parsed.namespace

        self.assertEqual(parsed.command, "mcp")
        self.assertEqual(parsed.subcommand, "call")
        self.assertEqual(args.mcp_call, "tail_logs")
        self.assertEqual(args.mcp_args, "{\"lines\": 10}")

    def test_workflow_run_uses_max_cycles_global_flag(self):
        parsed = parse_cli_args(["workflow", "run", "Summarize repo", "--max-cycles", "3"])
        args = parsed.namespace

        self.assertEqual(parsed.command, "workflow")
        self.assertEqual(parsed.subcommand, "run")
        self.assertEqual(args.workflow_goal, "Summarize repo")
        self.assertEqual(args.workflow_max_cycles, 3)

    def test_legacy_flags_still_parse_and_warn(self):
        parsed = parse_cli_args(["--add-goal", "Fix tests", "--run-goals"])
        args = parsed.namespace

        self.assertTrue(parsed.legacy_invocation_used)
        self.assertFalse(parsed.uses_subcommand)
        self.assertEqual(parsed.command, "goal")
        self.assertEqual(parsed.subcommand, "add+run")
        self.assertEqual(args.add_goal, "Fix tests")
        self.assertTrue(args.run_goals)
        self.assertTrue(parsed.warnings)
        self.assertTrue(parsed.warning_records)
        warning = parsed.warning_records[0]
        self.assertEqual(warning.code, "legacy_cli_flags_deprecated")
        self.assertEqual(warning.category, "deprecation")
        self.assertEqual(warning.action, "goal_add_run")
        self.assertEqual(warning.replacement_command, "aura goal add")
        self.assertEqual(warning.legacy_flags, ("--add-goal", "--run-goals"))
        self.assertEqual(warning.message, parsed.warnings[0])

    def test_help_subcommand_topic_is_captured(self):
        parsed = parse_cli_args(["help", "goal", "add"])
        self.assertEqual(parsed.command, "help")
        self.assertEqual(parsed.namespace.help_topics, ["goal", "add"])

    def test_watch_and_studio_parse_autonomous_flag(self):
        watch = parse_cli_args(["watch", "--autonomous"])
        studio = parse_cli_args(["studio", "--autonomous"])

        self.assertEqual(watch.action, "watch")
        self.assertEqual(studio.action, "studio")
        self.assertTrue(watch.namespace.autonomous)
        self.assertTrue(studio.namespace.autonomous)

    def test_render_help_json_contains_canonical_command(self):
        payload = json.loads(render_help(format="json"))
        paths = [tuple(item["path"]) for item in payload["commands"]]
        self.assertIn(("goal", "add"), paths)
        self.assertIn(("mcp", "call"), paths)

    def test_attach_cli_warnings_serializes_legacy_warning_records(self):
        parsed = parse_cli_args(["--mcp-tools"])
        payload = attach_cli_warnings({"status": "ok"}, parsed)

        self.assertEqual(payload["status"], "ok")
        self.assertIn("cli_warnings", payload)
        self.assertEqual(len(payload["cli_warnings"]), 1)
        warning = payload["cli_warnings"][0]
        self.assertEqual(warning["code"], "legacy_cli_flags_deprecated")
        self.assertEqual(warning["category"], "deprecation")
        self.assertEqual(warning["action"], "mcp_tools")
        self.assertEqual(warning["replacement_command"], "aura mcp tools")
        self.assertEqual(warning["legacy_flags"], ["--mcp-tools"])

    def test_attach_cli_warnings_omits_key_when_no_warnings(self):
        parsed = parse_cli_args(["mcp", "tools"])
        payload = attach_cli_warnings({"status": "ok"}, parsed)
        self.assertEqual(payload, {"status": "ok"})

    def test_cli_parse_error_payload_uses_shared_contract(self):
        exc = CLIParseError("bad input", usage="usage: aura ...")
        payload = cli_parse_error_payload(exc)

        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["code"], CLI_PARSE_ERROR_CODE)
        self.assertEqual(payload["message"], "bad input")
        self.assertEqual(payload["usage"], "usage: aura ...")

    def test_unknown_help_topic_payload_uses_shared_contract(self):
        payload = unknown_command_help_topic_payload("Unknown command help topic 'x'.")

        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["code"], UNKNOWN_COMMAND_HELP_TOPIC_CODE)
        self.assertEqual(payload["message"], "Unknown command help topic 'x'.")

    def test_parse_error_suggests_top_level_command(self):
        with self.assertRaises(CLIParseError) as ctx:
            parse_cli_args(["goa", "status"])
        self.assertIn("Did you mean 'goal'?", str(ctx.exception))

    def test_rejects_mixing_subcommand_and_legacy_flags(self):
        with self.assertRaises(CLIParseError) as ctx:
            parse_cli_args(["--run-goals", "goal", "status"])
        self.assertIn("Cannot mix canonical subcommands", str(ctx.exception))

    def test_legacy_mcp_args_requires_mcp_call(self):
        with self.assertRaises(CLIParseError) as ctx:
            parse_cli_args(["--mcp-args", "{\"lines\": 5}"])
        self.assertIn("`--mcp-args` requires `--mcp-call", str(ctx.exception))

    def test_rejects_conflicting_legacy_primary_actions(self):
        with self.assertRaises(CLIParseError) as ctx:
            parse_cli_args(["--diag", "--status"])
        self.assertIn("Conflicting legacy actions provided", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
