import json
import unittest

import aura_cli.options as cli_options_meta
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
        parsed = parse_cli_args(["mcp", "call", "tail_logs", "--args", '{"lines": 10}'])
        args = parsed.namespace

        self.assertEqual(parsed.command, "mcp")
        self.assertEqual(parsed.subcommand, "call")
        self.assertEqual(args.mcp_call, "tail_logs")
        self.assertEqual(args.mcp_args, '{"lines": 10}')

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
        self.assertEqual(warning.code, cli_options_meta.CLI_WARNINGS_CODE_LEGACY_FLAGS_DEPRECATED)
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

    def test_evolve_parses_innovation_flags(self):
        parsed = parse_cli_args(["evolve", "--queue-only", "--proposal-limit", "3", "--focus", "research"])
        args = parsed.namespace

        self.assertEqual(parsed.command, "evolve")
        self.assertTrue(args.queue_only)
        self.assertFalse(args.execute_queued)
        self.assertEqual(args.proposal_limit, 3)
        self.assertEqual(args.focus, "research")

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
        self.assertEqual(warning["code"], cli_options_meta.CLI_WARNINGS_CODE_LEGACY_FLAGS_DEPRECATED)
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
            parse_cli_args(["--mcp-args", '{"lines": 5}'])
        self.assertIn("`--mcp-args` requires `--mcp-call", str(ctx.exception))

    def test_rejects_conflicting_legacy_primary_actions(self):
        with self.assertRaises(CLIParseError) as ctx:
            parse_cli_args(["--diag", "--status"])
        self.assertIn("Conflicting legacy actions provided", str(ctx.exception))

    def test_legacy_primary_flags_emit_structured_deprecation_warning_records(self):
        cases = [
            (["--bootstrap"], "bootstrap"),
            (["--diag"], "diag"),
            (["--mcp-tools"], "mcp_tools"),
            (["--mcp-call", "limits"], "mcp_call"),
            (["--workflow-goal", "Summarize repo"], "workflow_run"),
            (["--status"], "goal_status"),
            (["--add-goal", "Fix tests"], "goal_add"),
            (["--goal", "Summarize repo"], "goal_once"),
            (["--run-goals"], "goal_run"),
            (["--scaffold", "demo"], "scaffold"),
            (["--evolve"], "evolve"),
            (["--add-goal", "Fix tests", "--run-goals"], "goal_add_run"),
        ]

        for argv, expected_action in cases:
            with self.subTest(argv=argv):
                parsed = parse_cli_args(argv)
                self.assertTrue(parsed.legacy_invocation_used)
                self.assertFalse(parsed.uses_subcommand)
                self.assertEqual(parsed.action, expected_action)
                self.assertEqual(len(parsed.warning_records), 1)

                warning = parsed.warning_records[0]
                self.assertEqual(warning.code, cli_options_meta.CLI_WARNINGS_CODE_LEGACY_FLAGS_DEPRECATED)
                self.assertEqual(warning.category, "deprecation")
                self.assertEqual(warning.phase, "compatibility")
                self.assertEqual(warning.action, expected_action)
                self.assertEqual(parsed.warnings, [warning.message])

                canonical_path = cli_options_meta.action_default_canonical_path(expected_action)
                expected_replacement = f"aura {' '.join(canonical_path)}" if canonical_path else None
                self.assertEqual(warning.replacement_command, expected_replacement)

                expected_flags = tuple(sorted(token for token in argv if token.startswith("--")))
                self.assertEqual(warning.legacy_flags, expected_flags)

    def test_error_json_payload_codes_are_documented_in_help_schema_contract(self):
        schema = cli_options_meta.help_schema()
        json_contracts = schema.get("json_contracts") or {}
        contract = json_contracts[cli_options_meta.CLI_ERRORS_JSON_CONTRACT_NAME]
        documented_codes = {item["code"]: item for item in (contract.get("record_codes") or [])}

        parse_payload = cli_parse_error_payload(CLIParseError("bad input", usage="usage: aura ..."))
        self.assertEqual(parse_payload["code"], cli_options_meta.CLI_PARSE_ERROR_CODE)
        self.assertIn(parse_payload["code"], documented_codes)
        self.assertEqual(parse_payload["status"], documented_codes[parse_payload["code"]]["status"])
        self.assertIn("usage", parse_payload)

        help_payload = unknown_command_help_topic_payload("Unknown command help topic 'x'.")
        self.assertEqual(help_payload["code"], cli_options_meta.UNKNOWN_COMMAND_HELP_TOPIC_CODE)
        self.assertIn(help_payload["code"], documented_codes)
        self.assertEqual(help_payload["status"], documented_codes[help_payload["code"]]["status"])
        self.assertNotIn("usage", help_payload)


if __name__ == "__main__":
    unittest.main()


# ── Extended pytest-style tests ────────────────────────────────────────────────

import argparse  # noqa: E402
import pytest  # noqa: E402

from aura_cli.cli_options import (  # noqa: E402
    AuraArgumentParser,
    CLIWarningRecord,
    ParsedCLIArgs,
    _add_common_flags,
    _add_root_legacy_flags,
    _children_by_parent,
    _explicit_long_option_names,
    _first_token_unknown_top_level,
    _leaf_command_paths,
    _subparser_dest,
    build_parser,
    cli_contract_report,
    iter_parser_command_paths,
    parser_customizer_paths,
    parser_leaf_command_paths,
    parser_parent_command_paths,
    parser_required_subcommand_parent_paths,
)


# ── CLIWarningRecord ──────────────────────────────────────────────────────────


class TestCLIWarningRecord:
    def test_to_dict_contains_all_fields(self):
        rec = CLIWarningRecord(
            code="DEPRECATION",
            message="use canonical",
            category="deprecation",
            action="goal_add",
            replacement_command="aura goal add",
            legacy_flags=("--add-goal",),
            phase="compatibility",
        )
        d = rec.to_dict()
        assert d["code"] == "DEPRECATION"
        assert d["message"] == "use canonical"
        assert d["category"] == "deprecation"
        assert d["action"] == "goal_add"
        assert d["replacement_command"] == "aura goal add"
        assert d["legacy_flags"] == ["--add-goal"]
        assert d["phase"] == "compatibility"

    def test_to_dict_legacy_flags_is_list(self):
        rec = CLIWarningRecord(code="X", message="m", legacy_flags=("--a", "--b"))
        assert isinstance(rec.to_dict()["legacy_flags"], list)

    def test_to_dict_none_action(self):
        rec = CLIWarningRecord(code="X", message="m", action=None)
        assert rec.to_dict()["action"] is None

    def test_default_category_is_deprecation(self):
        rec = CLIWarningRecord(code="X", message="m")
        assert rec.category == "deprecation"

    def test_default_phase_is_compatibility(self):
        rec = CLIWarningRecord(code="X", message="m")
        assert rec.phase == "compatibility"


# ── AuraArgumentParser ────────────────────────────────────────────────────────


class TestAuraArgumentParser:
    def test_error_raises_cli_parse_error(self):
        parser = AuraArgumentParser()
        with pytest.raises(CLIParseError) as exc_info:
            parser.error("bad argument")
        assert "bad argument" in str(exc_info.value)
        assert exc_info.value.code == 2

    def test_error_includes_usage(self):
        parser = AuraArgumentParser()
        with pytest.raises(CLIParseError) as exc_info:
            parser.error("something wrong")
        assert exc_info.value.usage is not None


# ── ParsedCLIArgs ─────────────────────────────────────────────────────────────


class TestParsedCLIArgs:
    def test_constructed_from_parse(self):
        parsed = parse_cli_args(["goal", "add", "my goal"])
        assert isinstance(parsed, ParsedCLIArgs)
        assert parsed.command == "goal"
        assert parsed.subcommand == "add"
        assert parsed.uses_subcommand is True
        assert parsed.legacy_invocation_used is False
        assert parsed.warnings == []
        assert parsed.warning_records == []

    def test_legacy_invocation_sets_flags(self):
        parsed = parse_cli_args(["--bootstrap"])
        assert parsed.legacy_invocation_used is True
        assert parsed.uses_subcommand is False
        assert len(parsed.warning_records) == 1


# ── Internal helpers ──────────────────────────────────────────────────────────


class TestSubparserDest:
    def test_empty_path_returns_command_1(self):
        assert _subparser_dest(()) == "_command_1"

    def test_single_element_returns_command_2(self):
        assert _subparser_dest(("agent",)) == "_command_2"

    def test_two_elements_returns_command_3(self):
        assert _subparser_dest(("goal", "add")) == "_command_3"


class TestChildrenByParent:
    def test_returns_dict(self):
        result = _children_by_parent()
        assert isinstance(result, dict)

    def test_root_has_children(self):
        result = _children_by_parent()
        root_children = result.get(())
        assert root_children is not None
        assert len(root_children) > 0

    def test_goal_has_subcommands(self):
        result = _children_by_parent()
        goal_children = result.get(("goal",))
        assert goal_children is not None
        assert ("goal", "add") in goal_children


class TestLeafCommandPaths:
    def test_returns_set(self):
        assert isinstance(_leaf_command_paths(), set)

    def test_goal_add_is_leaf(self):
        assert ("goal", "add") in _leaf_command_paths()

    def test_goal_itself_not_leaf(self):
        assert ("goal",) not in _leaf_command_paths()

    def test_mcp_tools_is_leaf(self):
        assert ("mcp", "tools") in _leaf_command_paths()


class TestFirstTokenUnknownTopLevel:
    def test_known_command_returns_none(self):
        assert _first_token_unknown_top_level(["goal", "add"]) is None

    def test_flag_first_returns_none(self):
        assert _first_token_unknown_top_level(["--json", "goal"]) is None

    def test_empty_argv_returns_none(self):
        assert _first_token_unknown_top_level([]) is None

    def test_typo_returns_tuple_with_suggestion(self):
        result = _first_token_unknown_top_level(["gol", "add"])
        assert result is not None
        token, suggestion = result
        assert token == "gol"
        assert suggestion == "goal"

    def test_completely_unknown_returns_tuple_with_none_suggestion(self):
        result = _first_token_unknown_top_level(["zzzzunknown"])
        assert result is not None
        token, suggestion = result
        assert token == "zzzzunknown"
        assert suggestion is None


class TestExplicitLongOptionNames:
    def test_empty_argv_returns_empty_set(self):
        assert _explicit_long_option_names([]) == set()

    def test_long_flags_extracted(self):
        result = _explicit_long_option_names(["--json", "--dry-run"])
        assert "json" in result
        assert "dry_run" in result

    def test_dashes_normalized_to_underscores(self):
        result = _explicit_long_option_names(["--dry-run"])
        assert "dry_run" in result

    def test_equals_sign_option_extracted(self):
        result = _explicit_long_option_names(["--model=gpt4"])
        assert "model" in result

    def test_short_flags_ignored(self):
        result = _explicit_long_option_names(["-j"])
        assert result == set()

    def test_double_dash_sentinel_ignored(self):
        result = _explicit_long_option_names(["--"])
        assert result == set()

    def test_positional_args_ignored(self):
        result = _explicit_long_option_names(["goal", "add", "my goal"])
        assert result == set()


# ── build_parser ──────────────────────────────────────────────────────────────


class TestBuildParser:
    def test_returns_aura_argument_parser(self):
        parser = build_parser()
        assert isinstance(parser, AuraArgumentParser)

    def test_parser_prog_is_aura(self):
        assert build_parser().prog == "aura"

    def test_parser_has_version_action(self):
        parser = build_parser()
        # --version should be defined (will raise SystemExit when invoked)
        with pytest.raises(SystemExit):
            parser.parse_args(["--version"])

    def test_parser_has_json_flag(self):
        parser = build_parser()
        ns = parser.parse_args(["goal", "add", "x", "--json"])
        assert ns.json is True

    def test_parser_has_dry_run_flag(self):
        parser = build_parser()
        ns = parser.parse_args(["goal", "once", "x", "--dry-run"])
        assert ns.dry_run is True


# ── _add_common_flags / _add_root_legacy_flags ────────────────────────────────


class TestAddCommonFlags:
    def test_adds_json_flag(self):
        p = argparse.ArgumentParser()
        _add_common_flags(p)
        ns = p.parse_args(["--json"])
        assert ns.json is True

    def test_adds_dry_run_flag(self):
        p = argparse.ArgumentParser()
        _add_common_flags(p)
        ns = p.parse_args(["--dry-run"])
        assert ns.dry_run is True

    def test_adds_model_flag(self):
        p = argparse.ArgumentParser()
        _add_common_flags(p)
        ns = p.parse_args(["--model", "claude-3"])
        assert ns.model == "claude-3"

    def test_adds_max_cycles_flag(self):
        p = argparse.ArgumentParser()
        _add_common_flags(p)
        ns = p.parse_args(["--max-cycles", "10"])
        assert ns.max_cycles == 10

    def test_idempotent_on_full_state(self):
        p = argparse.ArgumentParser()
        _add_common_flags(p)
        _add_common_flags(p)  # second call should not raise

    def test_no_max_cycles_when_excluded(self):
        p = argparse.ArgumentParser()
        _add_common_flags(p, include_max_cycles=False)
        # Should have partial state, then add max_cycles on second call
        _add_common_flags(p, include_max_cycles=True)
        ns = p.parse_args(["--max-cycles", "5"])
        assert ns.max_cycles == 5


class TestAddRootLegacyFlags:
    def test_adds_add_goal(self):
        p = argparse.ArgumentParser()
        _add_root_legacy_flags(p)
        ns = p.parse_args(["--add-goal", "my goal"])
        assert ns.add_goal == "my goal"

    def test_adds_run_goals(self):
        p = argparse.ArgumentParser()
        _add_root_legacy_flags(p)
        ns = p.parse_args(["--run-goals"])
        assert ns.run_goals is True

    def test_adds_status(self):
        p = argparse.ArgumentParser()
        _add_root_legacy_flags(p)
        ns = p.parse_args(["--status"])
        assert ns.status is True

    def test_adds_goal(self):
        p = argparse.ArgumentParser()
        _add_root_legacy_flags(p)
        ns = p.parse_args(["--goal", "do x"])
        assert ns.goal == "do x"

    def test_adds_mcp_tools(self):
        p = argparse.ArgumentParser()
        _add_root_legacy_flags(p)
        ns = p.parse_args(["--mcp-tools"])
        assert ns.mcp_tools is True

    def test_adds_diag(self):
        p = argparse.ArgumentParser()
        _add_root_legacy_flags(p)
        ns = p.parse_args(["--diag"])
        assert ns.diag is True


# ── parse_cli_args edge cases ─────────────────────────────────────────────────


class TestParseCLIArgsExtended:
    def test_goal_status(self):
        parsed = parse_cli_args(["goal", "status"])
        assert parsed.action == "goal_status"
        assert parsed.command == "goal"

    def test_goal_once(self):
        parsed = parse_cli_args(["goal", "once", "do the thing"])
        assert parsed.action == "goal_once"
        assert parsed.namespace.goal == "do the thing"

    def test_goal_run(self):
        parsed = parse_cli_args(["goal", "run"])
        assert parsed.action == "goal_run"
        assert parsed.namespace.run_goals is True

    def test_goal_run_with_resume(self):
        parsed = parse_cli_args(["goal", "run", "--resume"])
        assert parsed.namespace.resume is True

    def test_mcp_tools(self):
        parsed = parse_cli_args(["mcp", "tools"])
        assert parsed.action == "mcp_tools"

    def test_mcp_call_with_args(self):
        parsed = parse_cli_args(["mcp", "call", "fs/read", "--args", '{"p":"."}'])
        assert parsed.action == "mcp_call"
        assert parsed.namespace.mcp_call == "fs/read"
        assert parsed.namespace.mcp_args == '{"p":"."}'

    def test_mcp_status(self):
        parsed = parse_cli_args(["mcp", "status"])
        assert parsed.action == "mcp_status"

    def test_mcp_restart(self):
        parsed = parse_cli_args(["mcp", "restart", "dev_tools"])
        assert parsed.action == "mcp_restart"

    def test_workflow_run(self):
        parsed = parse_cli_args(["workflow", "run", "Summarize repo"])
        assert parsed.action == "workflow_run"
        assert parsed.namespace.workflow_goal == "Summarize repo"

    def test_workflow_run_no_max_cycles_flag(self):
        parsed = parse_cli_args(["workflow", "run", "x"])
        # workflow run excludes the global --max-cycles per spec
        assert not hasattr(parsed.namespace, "max_cycles") or parsed.namespace.max_cycles is None

    def test_logs_with_tail(self):
        parsed = parse_cli_args(["logs", "--tail", "50"])
        assert parsed.action == "logs"
        assert parsed.namespace.tail == 50

    def test_metrics(self):
        parsed = parse_cli_args(["metrics"])
        assert parsed.action == "metrics_show"

    def test_diag(self):
        parsed = parse_cli_args(["diag"])
        assert parsed.action == "diag"

    def test_bootstrap(self):
        parsed = parse_cli_args(["bootstrap"])
        assert parsed.action == "bootstrap"

    def test_history_with_limit(self):
        parsed = parse_cli_args(["history", "--limit", "5"])
        assert parsed.action == "history"
        assert parsed.namespace.limit == 5

    def test_queue_list(self):
        parsed = parse_cli_args(["queue", "list"])
        assert parsed.action == "queue_list"

    def test_queue_clear(self):
        parsed = parse_cli_args(["queue", "clear"])
        assert parsed.action == "queue_clear"

    def test_memory_search(self):
        parsed = parse_cli_args(["memory", "search", "my query"])
        assert parsed.action == "memory_search"
        assert parsed.namespace.query == "my query"

    def test_memory_reindex(self):
        parsed = parse_cli_args(["memory", "reindex"])
        assert parsed.action == "memory_reindex"

    def test_cancel(self):
        parsed = parse_cli_args(["cancel", "run-id-abc"])
        assert parsed.action == "cancel"
        assert parsed.namespace.run_id == "run-id-abc"

    def test_config_set(self):
        parsed = parse_cli_args(["config", "set", "dry_run", "true"])
        assert parsed.action == "config_set"
        assert parsed.namespace.config_key == "dry_run"
        assert parsed.namespace.config_value == "true"

    def test_credentials_status(self):
        parsed = parse_cli_args(["credentials", "status"])
        assert parsed.action == "credentials_status"

    def test_credentials_migrate_with_yes(self):
        parsed = parse_cli_args(["credentials", "migrate", "--yes"])
        assert parsed.action == "credentials_migrate"
        assert parsed.namespace.yes is True

    def test_credentials_store(self):
        parsed = parse_cli_args(["credentials", "store", "--key", "api_key", "--value", "xyz"])
        assert parsed.action == "credentials_store"
        assert parsed.namespace.key == "api_key"
        assert parsed.namespace.value == "xyz"

    def test_credentials_delete(self):
        parsed = parse_cli_args(["credentials", "delete", "--key", "api_key", "--yes"])
        assert parsed.action == "credentials_delete"
        assert parsed.namespace.key == "api_key"

    def test_sadd_run(self):
        parsed = parse_cli_args(["sadd", "run", "--spec", "design.md"])
        assert parsed.action == "sadd_run"
        assert parsed.namespace.spec == "design.md"

    def test_sadd_run_with_options(self):
        parsed = parse_cli_args(["sadd", "run", "--spec", "d.md", "--max-parallel", "5", "--fail-fast"])
        assert parsed.namespace.max_parallel == 5
        assert parsed.namespace.fail_fast is True

    def test_sadd_status(self):
        parsed = parse_cli_args(["sadd", "status"])
        assert parsed.action == "sadd_status"

    def test_innovate_start(self):
        parsed = parse_cli_args(["innovate", "start", "my problem"])
        assert parsed.action == "innovate_start"

    def test_innovate_list(self):
        parsed = parse_cli_args(["innovate", "list"])
        assert parsed.action == "innovate_list"

    def test_innovate_list_with_limit(self):
        parsed = parse_cli_args(["innovate", "list", "--limit", "5", "--output", "json"])
        assert parsed.namespace.limit == 5
        assert parsed.namespace.output == "json"

    def test_evolve_queue_only(self):
        parsed = parse_cli_args(["evolve", "--queue-only"])
        assert parsed.action == "evolve"
        assert parsed.namespace.queue_only is True

    def test_evolve_with_focus(self):
        parsed = parse_cli_args(["evolve", "--focus", "quality"])
        assert parsed.namespace.focus == "quality"

    def test_beads_schemas(self):
        parsed = parse_cli_args(["beads", "schemas"])
        assert parsed.action == "beads_schemas"

    def test_scaffold(self):
        parsed = parse_cli_args(["scaffold", "mytemplate"])
        assert parsed.action == "scaffold"
        assert parsed.namespace.scaffold == "mytemplate"

    def test_scaffold_with_desc(self):
        parsed = parse_cli_args(["scaffold", "mytemplate", "--desc", "A template"])
        assert parsed.namespace.scaffold_desc == "A template"

    def test_watch_no_args(self):
        parsed = parse_cli_args(["watch"])
        assert parsed.action == "watch"

    def test_studio_autonomous(self):
        parsed = parse_cli_args(["studio", "--autonomous"])
        assert parsed.action == "studio"
        assert parsed.namespace.autonomous is True

    def test_goal_resume(self):
        parsed = parse_cli_args(["goal", "resume"])
        assert parsed.action == "goal_resume"

    def test_goal_resume_with_run(self):
        parsed = parse_cli_args(["goal", "resume", "--run"])
        assert parsed.namespace.run is True

    def test_beads_no_beads_conflict(self):
        with pytest.raises(CLIParseError, match="Cannot pass both"):
            parse_cli_args(["goal", "once", "x", "--beads", "--no-beads"])

    def test_beads_required_optional_conflict(self):
        with pytest.raises(CLIParseError, match="Cannot pass both"):
            parse_cli_args(["goal", "once", "x", "--beads-required", "--beads-optional"])

    def test_no_beads_with_mode_override_conflict(self):
        with pytest.raises(CLIParseError, match="cannot be combined"):
            parse_cli_args(["goal", "once", "x", "--no-beads", "--beads-required"])

    def test_workflow_max_cycles_requires_workflow_goal(self):
        with pytest.raises(CLIParseError, match="requires"):
            parse_cli_args(["--workflow-max-cycles", "5"])

    def test_unknown_command_with_no_suggestion(self):
        with pytest.raises(CLIParseError, match="Unknown command"):
            parse_cli_args(["zzzzxunknown"])

    def test_unknown_command_raises_cli_parse_error(self):
        with pytest.raises(CLIParseError):
            parse_cli_args(["foobar"])

    def test_common_flags_on_goal_once(self):
        parsed = parse_cli_args(["goal", "once", "x", "--dry-run", "--json", "--model", "gpt4"])
        assert parsed.namespace.dry_run is True
        assert parsed.namespace.json is True
        assert parsed.namespace.model == "gpt4"

    def test_common_flags_with_max_cycles(self):
        parsed = parse_cli_args(["goal", "once", "x", "--max-cycles", "3"])
        assert parsed.namespace.max_cycles == 3


# ── render_help ───────────────────────────────────────────────────────────────


class TestRenderHelp:
    def test_text_format_top_level(self):
        result = render_help()
        assert "AURA CLI" in result or "Commands" in result

    def test_text_format_goal_add(self):
        result = render_help(["goal", "add"])
        assert "goal add" in result.lower() or "aura goal add" in result

    def test_json_format_returns_valid_json(self):
        result = render_help(format="json")
        parsed = json.loads(result)
        assert "commands" in parsed

    def test_json_format_contains_goal_add(self):
        payload = json.loads(render_help(format="json"))
        paths = [tuple(item.get("path", [])) for item in payload["commands"]]
        assert ("goal", "add") in paths

    def test_invalid_format_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported help format"):
            render_help(format="html")

    def test_unknown_path_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown command help topic"):
            render_help(["totally", "unknown", "path"])

    def test_unknown_path_with_suggestion(self):
        # Test a path where a suggestion can be generated
        try:
            render_help(["gol", "add"])
        except ValueError as e:
            assert "Unknown command help topic" in str(e)


# ── attach_cli_warnings ───────────────────────────────────────────────────────


class TestAttachCLIWarningsExtended:
    def test_returns_copy_not_original(self):
        original = {"status": "ok"}
        parsed = parse_cli_args(["mcp", "tools"])
        result = attach_cli_warnings(original, parsed)
        assert result is not original

    def test_none_parsed_returns_copy(self):
        payload = {"status": "ok"}
        result = attach_cli_warnings(payload, None)
        assert result == payload

    def test_no_warnings_does_not_add_key(self):
        parsed = parse_cli_args(["goal", "add", "x"])
        result = attach_cli_warnings({"k": "v"}, parsed)
        assert "cli_warnings" not in result


# ── parser contract helpers ───────────────────────────────────────────────────


class TestParserContractHelpers:
    def test_parser_customizer_paths_returns_set(self):
        result = parser_customizer_paths()
        assert isinstance(result, set)
        assert ("goal", "add") in result

    def test_parser_leaf_command_paths_returns_set(self):
        result = parser_leaf_command_paths()
        assert isinstance(result, set)
        assert ("goal", "add") in result

    def test_parser_parent_command_paths_returns_set(self):
        result = parser_parent_command_paths()
        assert isinstance(result, set)
        assert ("goal",) in result

    def test_parser_required_subcommand_parent_paths(self):
        result = parser_required_subcommand_parent_paths()
        assert isinstance(result, set)
        assert ("goal",) in result

    def test_iter_parser_command_paths(self):
        result = iter_parser_command_paths()
        assert isinstance(result, list)
        assert ("goal", "add") in result
        assert len(result) > 10


# ── cli_contract_report ────────────────────────────────────────────────────────


class TestCLIContractReport:
    def test_report_returns_dict(self):
        report = cli_contract_report()
        assert isinstance(report, dict)

    def test_report_has_ok_key(self):
        report = cli_contract_report()
        assert "ok" in report

    def test_report_passes_contract(self):
        report = cli_contract_report()
        assert report["ok"] is True, f"Contract failures: {report.get('failure_keys')}"

    def test_report_has_parser_paths(self):
        report = cli_contract_report()
        assert "parser_paths" in report
        assert len(report["parser_paths"]) > 0

    def test_report_has_spec_paths(self):
        report = cli_contract_report()
        assert "spec_paths" in report

    def test_report_with_dispatch_registry(self):
        report = cli_contract_report(dispatch_registry={})
        assert "missing_in_dispatch" in report

    def test_report_failure_keys_empty_when_ok(self):
        report = cli_contract_report()
        assert report["failure_keys"] == []
