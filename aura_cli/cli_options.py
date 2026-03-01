from __future__ import annotations

import argparse
import difflib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from aura_cli.options import (
    CLI_PARSE_ERROR_CODE,
    CLI_WARNINGS_CODE_LEGACY_FLAGS_DEPRECATED,
    CLI_ACTION_SPECS_BY_ACTION,
    COMMAND_SPECS,
    COMMAND_SPECS_BY_PATH,
    UNKNOWN_COMMAND_HELP_TOPIC_CODE,
    action_default_canonical_path,
    action_display_command,
    action_display_subcommand,
    action_smoke_argv,
    help_schema,
    legacy_auxiliary_flag_names,
    legacy_allowed_multi_primary_flag_sets,
    legacy_primary_flag_names,
    match_action_from_rules,
    spec_argparse_kwargs,
)


_REQUIRED_SUBCOMMAND_PARENT_PATHS: set[tuple[str, ...]] = {
    ("goal",),
    ("mcp",),
    ("memory",),
    ("queue",),
    ("workflow",),
}

_COMMON_FLAG_EXCLUDED_PATHS: set[tuple[str, ...]] = {
    ("logs",),
}

_COMMON_FLAG_MAX_CYCLES_EXCLUDED_PATHS: set[tuple[str, ...]] = {
    ("workflow", "run"),
}

_TOP_LEVEL_COMMANDS: tuple[str, ...] = tuple(dict.fromkeys(spec.path[0] for spec in COMMAND_SPECS if spec.path))


class CLIParseError(Exception):
    def __init__(self, message: str, *, code: int = 2, usage: str | None = None):
        super().__init__(message)
        self.code = code
        self.usage = usage


class AuraArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIParseError(message, usage=self.format_usage())


@dataclass(frozen=True)
class ParsedCLIArgs:
    namespace: argparse.Namespace
    command: str | None
    subcommand: str | None
    action: str
    uses_subcommand: bool
    legacy_invocation_used: bool
    warnings: list[str] = field(default_factory=list)
    warning_records: list["CLIWarningRecord"] = field(default_factory=list)


@dataclass(frozen=True)
class CLIWarningRecord:
    code: str
    message: str
    category: str = "deprecation"
    action: str | None = None
    replacement_command: str | None = None
    legacy_flags: tuple[str, ...] = ()
    phase: str = "compatibility"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "category": self.category,
            "action": self.action,
            "replacement_command": self.replacement_command,
            "legacy_flags": list(self.legacy_flags),
            "phase": self.phase,
        }


def attach_cli_warnings(payload: Mapping[str, Any], parsed: "ParsedCLIArgs | None" = None) -> dict[str, Any]:
    """Attach structured CLI warnings to a JSON payload when present."""
    result = dict(payload)
    if parsed is None:
        return result
    records = getattr(parsed, "warning_records", None) or []
    if not records:
        return result
    result["cli_warnings"] = [record.to_dict() for record in records]
    return result


def cli_parse_error_payload(exc: CLIParseError) -> dict[str, Any]:
    return {
        "status": "error",
        "code": CLI_PARSE_ERROR_CODE,
        "message": str(exc),
        "usage": exc.usage,
    }


def unknown_command_help_topic_payload(message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "code": UNKNOWN_COMMAND_HELP_TOPIC_CODE,
        "message": message,
    }


_PARSER_CUSTOMIZERS: dict[tuple[str, ...], Any] = {}


# Parser construction ---------------------------------------------------------

def _subparser_dest(parent_path: tuple[str, ...]) -> str:
    if not parent_path:
        return "_command_1"
    return f"_command_{len(parent_path) + 1}"



def _children_by_parent() -> dict[tuple[str, ...], set[tuple[str, ...]]]:
    children: dict[tuple[str, ...], set[tuple[str, ...]]] = {}
    for spec in COMMAND_SPECS:
        if not spec.path:
            continue
        children.setdefault(spec.path[:-1], set()).add(spec.path)
    return children



def _leaf_command_paths() -> set[tuple[str, ...]]:
    children = _children_by_parent()
    return {spec.path for spec in COMMAND_SPECS if spec.path and spec.path not in children}



def _add_common_flags(parser: argparse.ArgumentParser, *, include_max_cycles: bool = True) -> None:
    state = getattr(parser, "_aura_common_flags_added", None)
    if state == "full":
        return
    if state == "partial":
        if include_max_cycles:
            parser.add_argument("--max-cycles", dest="max_cycles", type=int, help="Maximum loop cycles for one-off runs.")
            setattr(parser, "_aura_common_flags_added", "full")
        return

    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON where supported.")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Do not apply filesystem changes.")
    parser.add_argument("--decompose", action="store_true", help="Enable task decomposition mode.")
    parser.add_argument("--model", help="Override model name for this run.")
    parser.add_argument("--explain", action="store_true", help="Print decision log for one-off goal runs.")
    if include_max_cycles:
        parser.add_argument("--max-cycles", dest="max_cycles", type=int, help="Maximum loop cycles for one-off runs.")
        setattr(parser, "_aura_common_flags_added", "full")
    else:
        setattr(parser, "_aura_common_flags_added", "partial")



def _add_root_legacy_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--add-goal", dest="add_goal", help="Legacy: add a goal to the queue.")
    parser.add_argument("--run-goals", dest="run_goals", action="store_true", help="Legacy: run queued goals.")
    parser.add_argument("--status", action="store_true", help="Legacy: show goal status.")
    parser.add_argument("--goal", dest="goal", help="Legacy: run one goal immediately.")

    parser.add_argument("--workflow-goal", dest="workflow_goal", help="Legacy: workflow goal text.")
    parser.add_argument("--workflow-max-cycles", dest="workflow_max_cycles", type=int, help="Legacy: workflow max cycles.")

    parser.add_argument("--mcp-tools", dest="mcp_tools", action="store_true", help="Legacy: list MCP tools.")
    parser.add_argument("--mcp-call", dest="mcp_call", help="Legacy: MCP tool name to call.")
    parser.add_argument("--mcp-args", dest="mcp_args", help="Legacy: JSON args for MCP call.")

    parser.add_argument("--diag", action="store_true", help="Legacy: run MCP diagnostics.")
    parser.add_argument("--bootstrap", action="store_true", help="Legacy: bootstrap config.")
    parser.add_argument("--scaffold", dest="scaffold", help="Legacy: scaffold template name.")
    parser.add_argument("--scaffold-desc", dest="scaffold_desc", help="Legacy: scaffold description.")
    parser.add_argument("--evolve", action="store_true", help="Legacy: run evolution loop.")



def _add_spec_parser(
    subparsers: Any,
    *,
    path: tuple[str, ...],
    parser_map: dict[tuple[str, ...], argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        path[-1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        **spec_argparse_kwargs(path),
    )
    parser.set_defaults(_cmd_path=path)
    parser_map[path] = parser
    return parser



def _build_parser_tree_from_specs(root_parser: argparse.ArgumentParser) -> dict[tuple[str, ...], argparse.ArgumentParser]:
    parser_map: dict[tuple[str, ...], argparse.ArgumentParser] = {(): root_parser}
    subparser_map: dict[tuple[str, ...], Any] = {}

    def ensure_subparsers(parent_path: tuple[str, ...]) -> Any:
        if parent_path in subparser_map:
            return subparser_map[parent_path]
        parent_parser = parser_map[parent_path]
        sp = parent_parser.add_subparsers(dest=_subparser_dest(parent_path), metavar="COMMAND")
        if parent_path in _REQUIRED_SUBCOMMAND_PARENT_PATHS:
            sp.required = True
        subparser_map[parent_path] = sp
        return sp

    for spec in sorted(COMMAND_SPECS, key=lambda s: (len(s.path), s.path)):
        parent = spec.path[:-1]
        if parent not in parser_map:
            raise KeyError(f"Missing parent parser for path {spec.path}")
        _add_spec_parser(ensure_subparsers(parent), path=spec.path, parser_map=parser_map)

    leaf_paths = _leaf_command_paths()
    for path, parser in parser_map.items():
        if not path:
            continue
        if path in leaf_paths and path not in _COMMON_FLAG_EXCLUDED_PATHS:
            _add_common_flags(parser, include_max_cycles=path not in _COMMON_FLAG_MAX_CYCLES_EXCLUDED_PATHS)

    return parser_map


# Parser customizers ----------------------------------------------------------

def _customize_help(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("help_topics", nargs="*", help="Command path to show help for.")



def _customize_goal_add(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("add_goal", help="Goal text to add to the queue.")
    parser.add_argument("--run", dest="run_goals", action="store_true", help="Run the queue after adding.")



def _customize_goal_run(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(run_goals=True)



def _customize_goal_status(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(status=True)



def _customize_goal_once(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("goal", help="Goal text to run once.")



def _customize_workflow_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("workflow_goal", help="Workflow goal text.")
    parser.add_argument("--max-cycles", dest="workflow_max_cycles", type=int, help="Workflow-specific max cycles.")



def _customize_mcp_tools(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(mcp_tools=True)



def _customize_mcp_call(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("mcp_call", help="MCP tool name.")
    parser.add_argument("--args", dest="mcp_args", help="JSON arguments for the MCP tool.")



def _customize_diag(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(diag=True)



def _customize_bootstrap(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(bootstrap=True)



def _customize_contract_report(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when the report contains contract failures.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact single-line JSON instead of pretty-printed output.",
    )
    parser.add_argument(
        "--no-dispatch",
        action="store_true",
        help="Skip dispatch-registry parity checks.",
    )



def _customize_scaffold(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("scaffold", help="Scaffold template name.")
    parser.add_argument("--desc", dest="scaffold_desc", help="Scaffold description.")



def _customize_evolve(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(evolve=True)
    parser.add_argument("goal", nargs="?", help="Optional evolution goal override.")



def _customize_logs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tail", type=int, help="Show only the last N log lines.")
    parser.add_argument("--level", default="info", help="Level filter.")
    parser.add_argument("--file", help="Read logs from a file instead of stdin.")
    parser.add_argument("--follow", action="store_true", help="Follow the file when using --file.")


def _customize_studio(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(watch=True)
    parser.add_argument("--autonomous", action="store_true", help="Start the goal loop in the background.")


def _customize_queue_list(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(queue_list=True)


def _customize_queue_clear(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(queue_clear=True)


def _customize_memory_search(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("query", help="Text query for semantic search.")
    parser.add_argument("--limit", type=int, default=5, help="Max results to return.")


def _customize_metrics(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(metrics_show=True)


_PARSER_CUSTOMIZERS.update(
    {
        ("help",): _customize_help,
        ("bootstrap",): _customize_bootstrap,
        ("contract-report",): _customize_contract_report,
        ("diag",): _customize_diag,
        ("watch",): _customize_studio,
        ("studio",): _customize_studio,
        ("goal", "add"): _customize_goal_add,
        ("goal", "run"): _customize_goal_run,
        ("goal", "status"): _customize_goal_status,
        ("goal", "once"): _customize_goal_once,
        ("workflow", "run"): _customize_workflow_run,
        ("mcp", "tools"): _customize_mcp_tools,
        ("mcp", "call"): _customize_mcp_call,
        ("scaffold",): _customize_scaffold,
        ("evolve",): _customize_evolve,
        ("logs",): _customize_logs,
        ("queue", "list"): _customize_queue_list,
        ("queue", "clear"): _customize_queue_clear,
        ("memory", "search"): _customize_memory_search,
        ("metrics",): _customize_metrics,
    }
)



def _apply_parser_customizations(parser_map: Mapping[tuple[str, ...], argparse.ArgumentParser]) -> None:
    for path, customizer in _PARSER_CUSTOMIZERS.items():
        parser = parser_map.get(path)
        if parser is not None:
            customizer(parser)


def parser_customizer_paths() -> set[tuple[str, ...]]:
    return set(_PARSER_CUSTOMIZERS)


def parser_leaf_command_paths() -> set[tuple[str, ...]]:
    return set(_leaf_command_paths())


def parser_parent_command_paths() -> set[tuple[str, ...]]:
    return {path for path in _children_by_parent() if path}


def parser_required_subcommand_parent_paths() -> set[tuple[str, ...]]:
    return set(_REQUIRED_SUBCOMMAND_PARENT_PATHS)



def build_parser() -> AuraArgumentParser:
    parser = AuraArgumentParser(prog="aura", allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json-help", dest="json_help", action="store_true", help="Emit machine-readable CLI help schema.")
    _add_common_flags(parser)
    _add_root_legacy_flags(parser)
    parser_map = _build_parser_tree_from_specs(parser)
    _apply_parser_customizations(parser_map)
    return parser


# Parsing and normalization ---------------------------------------------------

def _usage(parser: argparse.ArgumentParser) -> str:
    try:
        return parser.format_usage()
    except Exception:
        return ""



def _first_token_unknown_top_level(argv: Sequence[str]) -> tuple[str, str | None] | None:
    if not argv:
        return None
    token = argv[0]
    if token.startswith("-"):
        return None
    if token in _TOP_LEVEL_COMMANDS:
        return None
    matches = difflib.get_close_matches(token, _TOP_LEVEL_COMMANDS, n=1, cutoff=0.6)
    return token, (matches[0] if matches else None)



def _explicit_long_option_names(argv: Sequence[str]) -> set[str]:
    names: set[str] = set()
    for token in argv:
        if not token.startswith("--") or token == "--":
            continue
        option = token[2:].split("=", 1)[0]
        if option:
            names.add(option.replace("-", "_"))
    return names



def _validate_args(
    ns: argparse.Namespace,
    *,
    uses_subcommand: bool,
    parser: argparse.ArgumentParser,
    explicit_legacy_flags: set[str],
    explicit_long_options: set[str],
) -> None:
    usage = _usage(parser)

    if uses_subcommand and explicit_legacy_flags:
        raise CLIParseError("Cannot mix canonical subcommands with legacy flags in one invocation.", usage=usage)

    if "mcp_args" in explicit_long_options and not getattr(ns, "mcp_call", None):
        raise CLIParseError("`--mcp-args` requires `--mcp-call <tool>`.", usage=usage)

    if "workflow_max_cycles" in explicit_long_options and not getattr(ns, "workflow_goal", None):
        raise CLIParseError("`--workflow-max-cycles` requires `--workflow-goal <goal>`.", usage=usage)

    if not uses_subcommand and len(explicit_legacy_flags) > 1:
        if frozenset(explicit_legacy_flags) not in legacy_allowed_multi_primary_flag_sets():
            human_flags = ", ".join(f"--{name.replace('_', '-')}" for name in sorted(explicit_legacy_flags))
            raise CLIParseError(f"Conflicting legacy actions provided: {human_flags}.", usage=usage)



def _legacy_warning_for_action(action: str) -> str:
    path = action_default_canonical_path(action)
    if not path:
        return "Legacy flags are deprecated; use canonical subcommands."
    return f"Legacy flags are deprecated; use `aura {' '.join(path)}` instead."


def _legacy_warning_record(action: str, explicit_legacy_flags: set[str]) -> CLIWarningRecord:
    message = _legacy_warning_for_action(action)
    path = action_default_canonical_path(action)
    replacement_command = f"aura {' '.join(path)}" if path else None
    return CLIWarningRecord(
        code=CLI_WARNINGS_CODE_LEGACY_FLAGS_DEPRECATED,
        message=message,
        action=action,
        replacement_command=replacement_command,
        legacy_flags=tuple(sorted(f"--{name.replace('_', '-')}" for name in explicit_legacy_flags)),
    )



def _normalize_command_identity(*, ns: argparse.Namespace, uses_subcommand: bool, action: str) -> tuple[str | None, str | None]:
    if uses_subcommand:
        path = tuple(getattr(ns, "_cmd_path", ()) or ())
        if not path:
            return None, None
        return path[0], (path[1] if len(path) > 1 else None)
    return action_display_command(action), action_display_subcommand(action)



def parse_cli_args(argv: Sequence[str]) -> ParsedCLIArgs:
    raw_argv = list(argv)
    parser = build_parser()

    unknown = _first_token_unknown_top_level(raw_argv)
    if unknown is not None:
        token, suggestion = unknown
        message = f"Unknown command '{token}'."
        if suggestion:
            message += f" Did you mean '{suggestion}'?"
        raise CLIParseError(message, usage=_usage(parser))

    explicit_long_options = _explicit_long_option_names(raw_argv)
    explicit_legacy_flags = explicit_long_options & legacy_primary_flag_names()

    ns = parser.parse_args(raw_argv)
    cmd_path = tuple(getattr(ns, "_cmd_path", ()) or ())
    uses_subcommand = bool(cmd_path)

    _validate_args(
        ns,
        uses_subcommand=uses_subcommand,
        parser=parser,
        explicit_legacy_flags=explicit_legacy_flags,
        explicit_long_options=explicit_long_options,
    )

    command = cmd_path[0] if uses_subcommand else None
    subcommand = cmd_path[1] if uses_subcommand and len(cmd_path) > 1 else None
    action = match_action_from_rules(command=command, subcommand=subcommand, namespace=ns, uses_subcommand=uses_subcommand)

    command, subcommand = _normalize_command_identity(ns=ns, uses_subcommand=uses_subcommand, action=action)

    legacy_invocation_used = bool(explicit_legacy_flags and not uses_subcommand)
    warnings: list[str] = []
    warning_records: list[CLIWarningRecord] = []
    if legacy_invocation_used:
        record = _legacy_warning_record(action, explicit_legacy_flags)
        warning_records.append(record)
        warnings.append(record.message)

    return ParsedCLIArgs(
        namespace=ns,
        command=command,
        subcommand=subcommand,
        action=action,
        uses_subcommand=uses_subcommand,
        legacy_invocation_used=legacy_invocation_used,
        warnings=warnings,
        warning_records=warning_records,
    )


# Help rendering --------------------------------------------------------------

def _format_command_path(path: Sequence[str]) -> str:
    return " ".join(path)



def _render_text_help_for_path(path: tuple[str, ...]) -> str:
    spec = COMMAND_SPECS_BY_PATH.get(path)
    if spec is None:
        labels = [_format_command_path(spec.path) for spec in COMMAND_SPECS]
        target = _format_command_path(path)
        suggestion = difflib.get_close_matches(target, labels, n=1, cutoff=0.5)
        if suggestion:
            raise ValueError(f"Unknown command help topic '{target}'. Did you mean '{suggestion[0]}'?")
        raise ValueError(f"Unknown command help topic '{target}'.")

    lines = [f"aura {_format_command_path(spec.path)}", "", spec.description or spec.summary]
    if spec.legacy_flags:
        lines.extend(["", "Legacy flags:", f"  {' '.join(spec.legacy_flags)}"])
    if spec.examples:
        lines.extend(["", "Examples:"])
        lines.extend(f"  {example}" for example in spec.examples)

    children = [child for child in COMMAND_SPECS if child.path[:-1] == spec.path]
    if children:
        lines.extend(["", "Subcommands:"])
        for child in sorted(children, key=lambda s: s.path):
            lines.append(f"  {child.path[-1]:<12} {child.summary}")

    return "\n".join(lines).rstrip() + "\n"



def _render_top_level_help() -> str:
    top_level = [spec for spec in COMMAND_SPECS if len(spec.path) == 1]
    lines = ["AURA CLI", "", "Commands:"]
    for spec in top_level:
        lines.append(f"  {spec.path[0]:<12} {spec.summary}")
    lines.extend(
        [
            "",
            "Examples:",
            "  python3 main.py goal add \"Refactor queue\" --run",
            "  python3 main.py mcp tools",
            "  python3 main.py help goal add",
            "",
            "Legacy flat flags remain supported but are deprecated.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"



def render_help(topics: Sequence[str] | None = None, *, format: str = "text") -> str:
    if format == "json":
        return json.dumps(help_schema(), indent=2)
    if format != "text":
        raise ValueError(f"Unsupported help format: {format}")
    if topics:
        return _render_text_help_for_path(tuple(topics))
    return _render_top_level_help()


# Contract helpers ------------------------------------------------------------

def iter_parser_command_paths() -> list[tuple[str, ...]]:
    return [spec.path for spec in COMMAND_SPECS]


_CLI_CONTRACT_FAILURE_KEYS: tuple[str, ...] = (
    "missing_in_specs",
    "missing_in_parser",
    "unknown_spec_lookups",
    "missing_in_help_schema",
    "extra_in_help_schema",
    "duplicate_help_paths",
    "customizers_on_non_leaf_paths",
    "missing_required_parent_paths",
    "extra_required_parent_paths",
    "unknown_help_actions",
    "undocumented_actions",
    "runtime_requirement_mismatches",
    "help_actions_with_runtime_mismatches",
    "actions_missing_command_specs",
    "actions_with_positional_smoke_args_missing_customizers",
    "command_specs_with_unknown_legacy_flags",
    "actions_with_undocumented_legacy_flags",
    "smoke_invocation_failures",
    "missing_in_dispatch",
    "extra_in_dispatch",
    "smoke_dispatch_mismatches",
)


def _report_failures(report: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for key in _CLI_CONTRACT_FAILURE_KEYS:
        value = report.get(key)
        if isinstance(value, list) and value:
            failures.append(key)
    return failures


def _documented_default_actions() -> set[str]:
    documented: set[str] = set()
    for item in help_schema().get("commands", []):
        action = item.get("action")
        if action:
            documented.add(action)
    return documented


def _smoke_invocation_and_dispatch_failures(
    dispatch_registry: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    invocation_failures: list[dict[str, Any]] = []
    dispatch_mismatches: list[dict[str, Any]] = []

    for action in sorted(CLI_ACTION_SPECS_BY_ACTION):
        try:
            argv = list(action_smoke_argv(action))
            parsed = parse_cli_args(argv)
        except Exception as exc:
            invocation_failures.append({"action": action, "error": str(exc)})
            continue

        resolved = parsed.action
        if resolved != action:
            dispatch_mismatches.append({"action": action, "resolved_action": resolved, "argv": argv})
            continue

        if dispatch_registry is not None and resolved not in dispatch_registry:
            dispatch_mismatches.append({"action": action, "resolved_action": resolved, "argv": argv})

    return invocation_failures, dispatch_mismatches


def _legacy_flag_contracts() -> tuple[list[tuple[str, ...]], list[dict[str, Any]]]:
    known_flags = {
        *{f"--{name.replace('_', '-')}" for name in legacy_primary_flag_names()},
        *{f"--{name.replace('_', '-')}" for name in legacy_auxiliary_flag_names()},
    }
    unknown_legacy_flags: list[tuple[str, ...]] = []
    undocumented_action_flags: list[dict[str, Any]] = []

    for spec in COMMAND_SPECS:
        if any(flag not in known_flags for flag in spec.legacy_flags):
            unknown_legacy_flags.append(spec.path)

    for action, action_spec in CLI_ACTION_SPECS_BY_ACTION.items():
        if action_spec.canonical_path is None or not action_spec.legacy_primary_flags:
            continue

        command_spec = COMMAND_SPECS_BY_PATH.get(action_spec.canonical_path)
        if command_spec is None:
            continue

        documented_flags = set(command_spec.legacy_flags)
        expected_flags = {f"--{name.replace('_', '-')}" for name in action_spec.legacy_primary_flags}
        if not expected_flags.issubset(documented_flags):
            undocumented_action_flags.append(
                {"action": action, "missing_flags": sorted(expected_flags - documented_flags)}
            )

    return sorted(unknown_legacy_flags), undocumented_action_flags


def cli_contract_report(dispatch_registry: Mapping[str, Any] | None = None) -> dict[str, Any]:
    parser_paths = set(iter_parser_command_paths())
    spec_paths = set(spec.path for spec in COMMAND_SPECS)
    leaf_paths = parser_leaf_command_paths()
    parent_paths = parser_parent_command_paths()
    required_parent_paths = parser_required_subcommand_parent_paths()
    customizer_paths = parser_customizer_paths()

    help_items = help_schema().get("commands", [])
    help_paths_list = [tuple(item.get("path", ())) for item in help_items]
    help_paths = set(help_paths_list)
    duplicate_help_paths = sorted(path for path in set(help_paths_list) if help_paths_list.count(path) > 1)
    help_actions = sorted({item["action"] for item in help_items if item.get("action")})

    unknown_spec_lookups: list[tuple[str, ...]] = []
    for path in sorted(parser_paths):
        try:
            spec_argparse_kwargs(path)
        except Exception:
            unknown_spec_lookups.append(path)

    runtime_requirement_mismatches: list[dict[str, Any]] = []
    for item in help_items:
        action = item.get("action")
        if not action or action not in CLI_ACTION_SPECS_BY_ACTION:
            continue
        expected = CLI_ACTION_SPECS_BY_ACTION[action].requires_runtime
        actual = item.get("requires_runtime")
        if actual != expected:
            runtime_requirement_mismatches.append(
                {"action": action, "expected_requires_runtime": expected, "actual_requires_runtime": actual}
            )

    actions_missing_command_specs = sorted(
        action
        for action, spec in CLI_ACTION_SPECS_BY_ACTION.items()
        if spec.canonical_path is not None and spec.canonical_path not in COMMAND_SPECS_BY_PATH
    )

    actions_with_positional_smoke_args_missing_customizers = sorted(
        action
        for action, spec in CLI_ACTION_SPECS_BY_ACTION.items()
        if spec.canonical_path is not None
        and any(token and not token.startswith("-") for token in action_smoke_argv(action)[len(spec.canonical_path):])
        and spec.canonical_path not in customizer_paths
    )

    unknown_legacy_flags, undocumented_action_flags = _legacy_flag_contracts()
    action_spec_actions = sorted(CLI_ACTION_SPECS_BY_ACTION)
    unknown_help_actions = sorted(set(help_actions) - set(action_spec_actions))

    documented_default_actions = _documented_default_actions()
    undocumented_actions = sorted(
        action
        for action, spec in CLI_ACTION_SPECS_BY_ACTION.items()
        if spec.canonical_path is not None and action in documented_default_actions and action not in help_actions
    )

    smoke_invocation_failures, smoke_dispatch_mismatches = _smoke_invocation_and_dispatch_failures(dispatch_registry)

    report: dict[str, Any] = {
        "parser_paths": sorted(parser_paths),
        "spec_paths": sorted(spec_paths),
        "help_paths": sorted(help_paths),
        "leaf_paths": sorted(leaf_paths),
        "parent_paths": sorted(parent_paths),
        "required_parent_paths": sorted(required_parent_paths),
        "customizer_paths": sorted(customizer_paths),
        "unknown_spec_lookups": unknown_spec_lookups,
        "missing_in_specs": sorted(parser_paths - spec_paths),
        "missing_in_parser": sorted(spec_paths - parser_paths),
        "missing_in_help_schema": sorted(spec_paths - help_paths),
        "extra_in_help_schema": sorted(help_paths - spec_paths),
        "duplicate_help_paths": duplicate_help_paths,
        "customizers_on_non_leaf_paths": sorted(customizer_paths - leaf_paths),
        "missing_required_parent_paths": sorted(parent_paths - required_parent_paths),
        "extra_required_parent_paths": sorted(required_parent_paths - parent_paths),
        "action_spec_actions": action_spec_actions,
        "help_actions": help_actions,
        "unknown_help_actions": unknown_help_actions,
        "undocumented_actions": undocumented_actions,
        "runtime_requirement_mismatches": runtime_requirement_mismatches,
        "help_actions_with_runtime_mismatches": list(runtime_requirement_mismatches),
        "actions_missing_command_specs": actions_missing_command_specs,
        "actions_with_positional_smoke_args_missing_customizers": actions_with_positional_smoke_args_missing_customizers,
        "command_specs_with_unknown_legacy_flags": unknown_legacy_flags,
        "actions_with_undocumented_legacy_flags": undocumented_action_flags,
        "smoke_invocation_failures": smoke_invocation_failures,
    }

    if dispatch_registry is not None:
        dispatch_actions = sorted(dispatch_registry)
        report["dispatch_actions"] = dispatch_actions
        report["missing_in_dispatch"] = sorted(set(action_spec_actions) - set(dispatch_actions))
        report["extra_in_dispatch"] = sorted(set(dispatch_actions) - set(action_spec_actions))
        report["smoke_dispatch_mismatches"] = smoke_dispatch_mismatches

    report["failure_keys"] = _report_failures(report)
    report["ok"] = not report["failure_keys"]
    return report


__all__ = [
    "attach_cli_warnings",
    "cli_parse_error_payload",
    "CLI_PARSE_ERROR_CODE",
    "AuraArgumentParser",
    "CLIWarningRecord",
    "CLIParseError",
    "UNKNOWN_COMMAND_HELP_TOPIC_CODE",
    "ParsedCLIArgs",
    "build_parser",
    "cli_contract_report",
    "iter_parser_command_paths",
    "unknown_command_help_topic_payload",
    "parse_cli_args",
    "parser_customizer_paths",
    "parser_leaf_command_paths",
    "parser_parent_command_paths",
    "parser_required_subcommand_parent_paths",
    "render_help",
]
