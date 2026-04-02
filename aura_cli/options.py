from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class CommandSpec:
    path: tuple[str, ...]
    summary: str
    description: str = ""
    examples: tuple[str, ...] = ()
    legacy_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class CLIActionSpec:
    action: str
    requires_runtime: bool
    canonical_path: tuple[str, ...] | None
    legacy_primary_flags: tuple[str, ...] = ()
    display_command: str | None = None
    display_subcommand: str | None = None


COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec(
        path=("help",),
        summary="Show CLI help",
        description="Show top-level help or help for a specific command path.",
        examples=(
            "python3 main.py help",
            "python3 main.py help goal add",
        ),
    ),
    CommandSpec(
        path=("doctor",),
        summary="Run system diagnostics",
        description="Run the AURA doctor checks for local environment health.",
        examples=("python3 main.py doctor",),
    ),
    CommandSpec(
        path=("readiness",),
        summary="Check V2 runtime readiness",
        description="Validate async runtime and MCP registry health.",
        examples=("python3 main.py readiness",),
    ),
    CommandSpec(
        path=("bootstrap",),
        summary="Create default config",
        description="Bootstrap local configuration files for AURA.",
        examples=("python3 main.py bootstrap",),
        legacy_flags=("--bootstrap",),
    ),
    CommandSpec(
        path=("config",),
        summary="Show effective config",
        description="Print the resolved effective runtime configuration.",
        examples=("python3 main.py config",),
    ),
    CommandSpec(
        path=("contract-report",),
        summary="Print CLI contract report",
        description="Print aggregated parser/help/schema/dispatch contract checks as JSON.",
        examples=(
            "python3 main.py contract-report --check",
            "python3 main.py contract-report --compact",
        ),
    ),
    CommandSpec(
        path=("diag",),
        summary="MCP diagnostics snapshot",
        description="Fetch MCP health, metrics, limits, and recent logs via HTTP.",
        examples=("python3 main.py diag",),
        legacy_flags=("--diag",),
    ),
    CommandSpec(
        path=("logs",),
        summary="Stream AURA logs",
        description="Tail or follow logs from stdin or a file.",
        examples=(
            "python3 main.py logs --tail 50",
            "python3 main.py logs --file memory/aura.log --follow",
        ),
    ),
    CommandSpec(
        path=("watch",),
        summary="Launch TUI monitor",
        description="Launch the AuraStudio terminal UI. Use --autonomous to start the goal loop.",
        examples=(
            "python3 main.py watch",
            "python3 main.py watch --autonomous",
        ),
    ),
    CommandSpec(
        path=("studio",),
        summary="Launch AURA Studio",
        description="Launch the rich real-time dashboard. Use --autonomous to start the goal loop.",
        examples=(
            "python3 main.py studio",
            "python3 main.py studio --autonomous",
        ),
    ),
    CommandSpec(
        path=("goal",),
        summary="Goal queue operations",
        description="Manage queued goals and run one-off goals.",
        examples=(
            "python3 main.py goal status",
            'python3 main.py goal add "Refactor queue"',
        ),
    ),
    CommandSpec(
        path=("goal", "add"),
        summary="Add a goal to the queue",
        description="Add a goal, optionally running the queue immediately.",
        examples=(
            'python3 main.py goal add "Fix tests"',
            'python3 main.py goal add "Fix tests" --run',
        ),
        legacy_flags=("--add-goal", "--run-goals"),
    ),
    CommandSpec(
        path=("goal", "run"),
        summary="Run queued goals",
        description="Run the goal queue through the autonomous loop.",
        examples=("python3 main.py goal run --dry-run",),
        legacy_flags=("--run-goals",),
    ),
    CommandSpec(
        path=("goal", "status"),
        summary="Show queue status",
        description="Show queued and completed goals.",
        examples=(
            "python3 main.py goal status",
            "python3 main.py goal status --json",
        ),
        legacy_flags=("--status",),
    ),
    CommandSpec(
        path=("goal", "once"),
        summary="Run a one-off goal",
        description="Run a single goal directly without queueing it.",
        examples=(
            'python3 main.py goal once "Summarize repo"',
            'python3 main.py goal once "Refactor core" --max-cycles 3',
        ),
        legacy_flags=("--goal",),
    ),
    CommandSpec(
        path=("goal", "resume"),
        summary="Resume an interrupted goal",
        description=(
            "Re-queue a goal that was interrupted mid-execution due to a crash or "
            "process kill. Reads memory/in_flight_goal.json written by the goal run "
            "loop. Use --run to immediately execute the re-queued goal."
        ),
        examples=(
            "python3 main.py goal resume",
            "python3 main.py goal resume --run",
        ),
    ),
    CommandSpec(
        path=("workflow",),
        summary="Workflow operations",
        description="Run orchestrated workflow goals with explicit cycle limits.",
        examples=('python3 main.py workflow run "Summarize repo"',),
    ),
    CommandSpec(
        path=("workflow", "run"),
        summary="Run a workflow goal",
        description="Run the orchestrator loop for a single workflow goal.",
        examples=('python3 main.py workflow run "Summarize repo" --max-cycles 3',),
        legacy_flags=("--workflow-goal",),
    ),
    CommandSpec(
        path=("mcp",),
        summary="MCP HTTP client commands",
        description="Inspect and call MCP tools from the CLI.",
        examples=("python3 main.py mcp tools",),
    ),
    CommandSpec(
        path=("mcp", "tools"),
        summary="List MCP tools",
        description="List tools exposed by the MCP HTTP server.",
        examples=("python3 main.py mcp tools",),
        legacy_flags=("--mcp-tools",),
    ),
    CommandSpec(
        path=("mcp", "call"),
        summary="Call an MCP tool",
        description="Invoke an MCP tool with optional JSON args.",
        examples=(
            "python3 main.py mcp call limits",
            "python3 main.py mcp call tail_logs --args '{\"lines\": 10}'",
        ),
        legacy_flags=("--mcp-call",),
    ),
    CommandSpec(
        path=("scaffold",),
        summary="Scaffold a project",
        description="Run the scaffolder agent for a named project type.",
        examples=(
            'python3 main.py scaffold demo --desc "small demo app"',
            "python3 main.py scaffold demo --json",
        ),
        legacy_flags=("--scaffold",),
    ),
    CommandSpec(
        path=("evolve",),
        summary="Run innovation workflow",
        description="Run the innovation workflow and optional queue-backed implementation loop for AURA core.",
        examples=(
            "python3 main.py evolve",
            "python3 main.py evolve --json",
            "python3 main.py evolve --queue-only --proposal-limit 3 --focus research",
        ),
        legacy_flags=("--evolve",),
    ),
    CommandSpec(
        path=("queue",),
        summary="Goal queue management",
        description="List, add, or clear goals in the autonomous queue.",
        examples=(
            "python3 main.py queue list",
            "python3 main.py queue list --json",
        ),
    ),
    CommandSpec(
        path=("queue", "list"),
        summary="List queued goals",
        description="Show all pending and completed goals.",
        examples=(
            "python3 main.py queue list",
            "python3 main.py queue list --json",
        ),
    ),
    CommandSpec(
        path=("queue", "clear"),
        summary="Clear the goal queue",
        description="Remove all pending goals from the queue.",
        examples=(
            "python3 main.py queue clear",
            "python3 main.py queue clear --json",
        ),
    ),
    CommandSpec(
        path=("memory",),
        summary="Semantic memory operations",
        description="Search or browse through the AURA brain.",
        examples=(
            'python3 main.py memory search "workflow engine"',
            'python3 main.py memory search "workflow engine" --json',
        ),
    ),
    CommandSpec(
        path=("memory", "search"),
        summary="Search semantic memory",
        description="Perform a semantic search over brain entries.",
        examples=(
            'python3 main.py memory search "workflow engine"',
            'python3 main.py memory search "workflow engine" --json',
        ),
    ),
    CommandSpec(
        path=("memory", "reindex"),
        summary="Rebuild semantic memory embeddings",
        description="Rebuild semantic memory embeddings for the active model and force a project sync.",
        examples=(
            "python3 main.py memory reindex",
            "python3 main.py memory reindex --json",
        ),
    ),
    CommandSpec(
        path=("metrics",),
        summary="Show performance metrics",
        description="Display cycle success rates and timing stats.",
        examples=(
            "python3 main.py metrics",
            "python3 main.py metrics --json",
        ),
    ),
    CommandSpec(
        path=("sadd",),
        summary="Sub-Agent Driven Development",
        description="Decompose a design spec into parallel workstreams and execute via sub-agents.",
        examples=(
            "python3 main.py sadd run --spec design.md --dry-run",
            "python3 main.py sadd run --spec design.md --max-parallel 3",
        ),
    ),
    CommandSpec(
        path=("sadd", "run"),
        summary="Run a SADD session",
        description="Parse a design spec and execute workstreams. Use --dry-run to preview decomposition only.",
        examples=(
            "python3 main.py sadd run --spec design.md --dry-run",
            "python3 main.py sadd run --spec design.md --max-parallel 2 --max-cycles 3",
        ),
    ),
    CommandSpec(
        path=("sadd", "status"),
        summary="Show SADD session status",
        description="Show status of recent SADD sessions or a specific session.",
        examples=(
            "python3 main.py sadd status",
            "python3 main.py sadd status --session-id <id>",
        ),
    ),
    CommandSpec(
        path=("sadd", "resume"),
        summary="Resume a SADD session",
        description="Resume an interrupted SADD session from its last checkpoint.",
        examples=("python3 main.py sadd resume --session-id <id>",),
    ),
    # ── Innovation Catalyst Commands ────────────────────────────────────────────
    CommandSpec(
        path=("innovate",),
        summary="Innovation Catalyst session management",
        description="Start, list, and manage innovation sessions using brainstorming techniques.",
        examples=(
            'python3 main.py innovate start "How to improve X?"',
            "python3 main.py innovate list",
            "python3 main.py innovate show --session-id abc123",
        ),
    ),
    CommandSpec(
        path=("innovate", "start"),
        summary="Start a new innovation session",
        description="Start a new innovation session with the Innovation Catalyst framework.",
        examples=(
            'python3 main.py innovate start "How to improve code review?"',
            'python3 main.py innovate start "Reduce bugs" --techniques scamper,six_hats',
            'python3 main.py innovate start "Improve UX" --execute-phase divergence',
        ),
    ),
    CommandSpec(
        path=("innovate", "list"),
        summary="List innovation sessions",
        description="List all innovation sessions with their status and metrics.",
        examples=(
            "python3 main.py innovate list",
            "python3 main.py innovate list --status active",
            "python3 main.py innovate list --json",
        ),
    ),
    CommandSpec(
        path=("innovate", "show"),
        summary="Show session details",
        description="Show detailed information about a specific innovation session.",
        examples=(
            "python3 main.py innovate show --session-id abc123",
            "python3 main.py innovate show --session-id abc123 --show-ideas",
            "python3 main.py innovate show --session-id abc123 --json",
        ),
    ),
    CommandSpec(
        path=("innovate", "resume"),
        summary="Resume an innovation session",
        description="Resume an innovation session at a specific phase.",
        examples=(
            "python3 main.py innovate resume --session-id abc123",
            "python3 main.py innovate resume --session-id abc123 --phase convergence",
        ),
    ),
    CommandSpec(
        path=("innovate", "export"),
        summary="Export session results",
        description="Export innovation session results to markdown or JSON.",
        examples=(
            "python3 main.py innovate export --session-id abc123 --format markdown",
            "python3 main.py innovate export --session-id abc123 --output report.md",
        ),
    ),
    CommandSpec(
        path=("innovate", "techniques"),
        summary="List available brainstorming techniques",
        description="Show all available brainstorming techniques with descriptions.",
        examples=(
            "python3 main.py innovate techniques",
            "python3 main.py innovate techniques --json",
        ),
    ),
    CommandSpec(
        path=("innovate", "to-goals"),
        summary="Convert selected ideas to goals",
        description="Convert selected ideas from an innovation session to goals in the queue.",
        examples=(
            "python3 main.py innovate to-goals --session-id abc123",
            "python3 main.py innovate to-goals --session-id abc123 --preview",
        ),
    ),
    CommandSpec(
        path=("innovate", "insights"),
        summary="Show innovation analytics and insights",
        description="Display analytics about innovation sessions including trends, technique effectiveness, and idea quality metrics.",
        examples=(
            "python3 main.py innovate insights",
            "python3 main.py innovate insights --session-id abc123",
            "python3 main.py innovate insights --json",
        ),
    ),
    CommandSpec(
        path=("agent", "run"),
        summary="Run goal via Agent SDK meta-controller",
        description="Execute a development goal using Claude-as-brain orchestration with dynamic tool/skill/workflow selection.",
    ),
)

COMMAND_SPECS_BY_PATH: dict[tuple[str, ...], CommandSpec] = {spec.path: spec for spec in COMMAND_SPECS}


CLI_ACTION_SPECS: tuple[CLIActionSpec, ...] = (
    CLIActionSpec("json_help", False, None),
    CLIActionSpec("help", False, ("help",)),
    CLIActionSpec("doctor", False, ("doctor",)),
    CLIActionSpec("readiness", True, ("readiness",)),
    CLIActionSpec("bootstrap", False, ("bootstrap",), legacy_primary_flags=("bootstrap",)),
    CLIActionSpec("show_config", False, ("config",)),
    CLIActionSpec("contract_report", False, ("contract-report",)),
    CLIActionSpec("mcp_tools", False, ("mcp", "tools"), legacy_primary_flags=("mcp_tools",)),
    CLIActionSpec("mcp_call", False, ("mcp", "call"), legacy_primary_flags=("mcp_call",)),
    CLIActionSpec("diag", False, ("diag",), legacy_primary_flags=("diag",)),
    CLIActionSpec("logs", False, ("logs",)),
    CLIActionSpec("watch", True, ("watch",)),
    CLIActionSpec("studio", True, ("studio",)),
    CLIActionSpec("workflow_run", True, ("workflow", "run"), legacy_primary_flags=("workflow_goal",)),
    CLIActionSpec("scaffold", True, ("scaffold",), legacy_primary_flags=("scaffold",)),
    CLIActionSpec("evolve", True, ("evolve",), legacy_primary_flags=("evolve",)),
    CLIActionSpec("goal_status", True, ("goal", "status"), legacy_primary_flags=("status",)),
    CLIActionSpec("goal_add", True, ("goal", "add"), legacy_primary_flags=("add_goal",)),
    CLIActionSpec(
        "goal_add_run",
        True,
        ("goal", "add"),
        legacy_primary_flags=("add_goal", "run_goals"),
        display_subcommand="add+run",
    ),
    CLIActionSpec("goal_once", True, ("goal", "once"), legacy_primary_flags=("goal",)),
    CLIActionSpec("goal_run", True, ("goal", "run"), legacy_primary_flags=("run_goals",)),
    CLIActionSpec("goal_resume", False, ("goal", "resume")),
    CLIActionSpec("queue_list", True, ("queue", "list")),
    CLIActionSpec("queue_clear", True, ("queue", "clear")),
    CLIActionSpec("memory_search", True, ("memory", "search")),
    CLIActionSpec("memory_reindex", True, ("memory", "reindex")),
    CLIActionSpec("metrics_show", True, ("metrics",)),
    CLIActionSpec("sadd_run", True, ("sadd", "run")),
    CLIActionSpec("sadd_status", False, ("sadd", "status")),
    CLIActionSpec("sadd_resume", True, ("sadd", "resume")),
    # ── Innovation Catalyst Actions ────────────────────────────────────────────
    CLIActionSpec("innovate_start", True, ("innovate", "start")),
    CLIActionSpec("innovate_list", True, ("innovate", "list")),
    CLIActionSpec("innovate_show", True, ("innovate", "show")),
    CLIActionSpec("innovate_resume", True, ("innovate", "resume")),
    CLIActionSpec("innovate_export", True, ("innovate", "export")),
    CLIActionSpec("innovate_techniques", False, ("innovate", "techniques")),
    CLIActionSpec("innovate_to_goals", True, ("innovate", "to-goals")),
    CLIActionSpec("innovate_insights", True, ("innovate", "insights")),
    CLIActionSpec("interactive", True, None),
    CLIActionSpec("agent_run", True, ("agent", "run")),
)

CLI_ACTION_SPECS_BY_ACTION: dict[str, CLIActionSpec] = {spec.action: spec for spec in CLI_ACTION_SPECS}

_ACTION_SMOKE_OVERRIDES: dict[str, tuple[str, ...]] = {
    "interactive": (),
    "json_help": ("--json-help",),
    "goal_add_run": ("goal", "add", "example-goal", "--run"),
    "sadd_run": ("sadd", "run", "--spec", "example-spec.md", "--dry-run"),
    "sadd_resume": ("sadd", "resume", "--session-id", "example-id"),
}

_SMOKE_POSITIONAL_ARGS_BY_PATH: dict[tuple[str, ...], tuple[str, ...]] = {
    ("goal", "add"): ("example-goal",),
    ("goal", "once"): ("example-goal",),
    ("workflow", "run"): ("example-goal",),
    ("mcp", "call"): ("limits",),
    ("scaffold",): ("demo",),
    ("memory", "search"): ("example-query",),
    ("innovate", "start"): ("example-problem-statement",),
}

HELP_SCHEMA_VERSION = 3
HELP_SCHEMA_GENERATED_BY = "aura_cli.options.help_schema"
CLI_WARNINGS_JSON_CONTRACT_VERSION = 1
CLI_WARNINGS_JSON_FIELD = "cli_warnings"
CLI_WARNINGS_CODE_LEGACY_FLAGS_DEPRECATED = "legacy_cli_flags_deprecated"
CLI_WARNINGS_RECORD_FIELDS: tuple[str, ...] = (
    "code",
    "message",
    "category",
    "action",
    "replacement_command",
    "legacy_flags",
    "phase",
)
CLI_WARNINGS_RECORD_CODES: tuple[dict[str, str], ...] = (
    {
        "code": CLI_WARNINGS_CODE_LEGACY_FLAGS_DEPRECATED,
        "category": "deprecation",
        "phase": "compatibility",
        "description": "Legacy flat CLI flags were used and mapped to a canonical command.",
    },
)

CLI_PARSE_ERROR_CODE = "cli_parse_error"
UNKNOWN_COMMAND_HELP_TOPIC_CODE = "unknown_command_help_topic"
CLI_ERRORS_JSON_CONTRACT_NAME = "cli_errors"
CLI_ERRORS_JSON_CONTRACT_VERSION = 1
CLI_ERRORS_RECORD_FIELDS: tuple[str, ...] = (
    "status",
    "code",
    "message",
)
CLI_ERRORS_OPTIONAL_FIELDS: tuple[str, ...] = ("usage",)
CLI_ERRORS_RECORD_CODES: tuple[dict[str, str], ...] = (
    {
        "code": CLI_PARSE_ERROR_CODE,
        "status": "error",
        "description": "CLI argument parsing or validation failed.",
    },
    {
        "code": UNKNOWN_COMMAND_HELP_TOPIC_CODE,
        "status": "error",
        "description": "Help topic path was not recognized.",
    },
)


_CANONICAL_PATH_TO_ACTION: dict[tuple[str, ...], str] = {
    ("help",): "help",
    ("doctor",): "doctor",
    ("readiness",): "readiness",
    ("bootstrap",): "bootstrap",
    ("config",): "show_config",
    ("contract-report",): "contract_report",
    ("diag",): "diag",
    ("logs",): "logs",
    ("watch",): "watch",
    ("studio",): "studio",
    ("workflow", "run"): "workflow_run",
    ("mcp", "tools"): "mcp_tools",
    ("mcp", "call"): "mcp_call",
    ("scaffold",): "scaffold",
    ("evolve",): "evolve",
    ("goal", "status"): "goal_status",
    ("goal", "once"): "goal_once",
    ("goal", "run"): "goal_run",
    ("goal", "resume"): "goal_resume",
    ("queue", "list"): "queue_list",
    ("queue", "clear"): "queue_clear",
    ("memory", "search"): "memory_search",
    ("memory", "reindex"): "memory_reindex",
    ("metrics",): "metrics_show",
    ("sadd", "run"): "sadd_run",
    ("sadd", "status"): "sadd_status",
    ("sadd", "resume"): "sadd_resume",
    ("innovate", "start"): "innovate_start",
    ("innovate", "list"): "innovate_list",
    ("innovate", "show"): "innovate_show",
    ("innovate", "resume"): "innovate_resume",
    ("innovate", "export"): "innovate_export",
    ("innovate", "techniques"): "innovate_techniques",
    ("innovate", "to-goals"): "innovate_to_goals",
    ("innovate", "insights"): "innovate_insights",
    ("agent", "run"): "agent_run",
}

_LEGACY_PRIMARY_FLAGS: tuple[str, ...] = (
    "bootstrap",
    "mcp_tools",
    "mcp_call",
    "diag",
    "workflow_goal",
    "scaffold",
    "evolve",
    "status",
    "add_goal",
    "goal",
    "run_goals",
)

_LEGACY_AUX_FLAGS: tuple[str, ...] = (
    "mcp_args",
    "workflow_max_cycles",
    "scaffold_desc",
)

_ALLOWED_MULTI_PRIMARY_FLAG_SETS: tuple[frozenset[str], ...] = (frozenset({"add_goal", "run_goals"}),)


def iter_cli_action_specs() -> tuple[CLIActionSpec, ...]:
    return CLI_ACTION_SPECS


def action_runtime_required(action: str) -> bool:
    spec = CLI_ACTION_SPECS_BY_ACTION.get(action)
    if spec is None:
        raise KeyError(f"Unknown CLI action: {action}")
    return spec.requires_runtime


def action_smoke_argv(action: str) -> tuple[str, ...]:
    override = _ACTION_SMOKE_OVERRIDES.get(action)
    if override is not None:
        return override

    spec = CLI_ACTION_SPECS_BY_ACTION.get(action)
    if spec is None:
        raise KeyError(f"Unknown CLI action: {action}")
    if spec.canonical_path is None:
        raise KeyError(f"No canonical smoke argv for action '{action}'.")

    argv: list[str] = list(spec.canonical_path)
    argv.extend(_SMOKE_POSITIONAL_ARGS_BY_PATH.get(spec.canonical_path, ()))
    return tuple(argv)


def action_default_canonical_path(action: str) -> tuple[str, ...] | None:
    spec = CLI_ACTION_SPECS_BY_ACTION.get(action)
    return spec.canonical_path if spec else None


def action_display_command(action: str) -> str | None:
    spec = CLI_ACTION_SPECS_BY_ACTION.get(action)
    if spec is None:
        return None
    if spec.display_command is not None:
        return spec.display_command
    if spec.canonical_path:
        return spec.canonical_path[0]
    return None


def action_display_subcommand(action: str) -> str | None:
    spec = CLI_ACTION_SPECS_BY_ACTION.get(action)
    if spec is None:
        return None
    if spec.display_subcommand is not None:
        return spec.display_subcommand
    if spec.canonical_path and len(spec.canonical_path) > 1:
        return spec.canonical_path[1]
    return None


def legacy_primary_flag_names() -> set[str]:
    return set(_LEGACY_PRIMARY_FLAGS)


def legacy_auxiliary_flag_names() -> set[str]:
    return set(_LEGACY_AUX_FLAGS)


def legacy_allowed_multi_primary_flag_sets() -> set[frozenset[str]]:
    return set(_ALLOWED_MULTI_PRIMARY_FLAG_SETS)


def _ns_get(namespace: Any, name: str, default: Any = None) -> Any:
    return getattr(namespace, name, default)


def match_canonical_action(command: str | None, subcommand: str | None, namespace: Any) -> str:
    if _ns_get(namespace, "json_help", False):
        return "json_help"
    cmd_path = tuple(getattr(namespace, "_cmd_path", ()) or ())
    path = cmd_path or tuple(part for part in (command, subcommand) if part)
    if path == ("goal", "add"):
        return "goal_add_run" if _ns_get(namespace, "run_goals", False) else "goal_add"
    return _CANONICAL_PATH_TO_ACTION.get(path, "interactive")


def match_legacy_action(namespace: Any) -> str:
    if _ns_get(namespace, "json_help", False):
        return "json_help"
    if _ns_get(namespace, "bootstrap", False):
        return "bootstrap"
    if _ns_get(namespace, "mcp_tools", False):
        return "mcp_tools"
    if _ns_get(namespace, "mcp_call", None):
        return "mcp_call"
    if _ns_get(namespace, "diag", False):
        return "diag"
    if _ns_get(namespace, "workflow_goal", None):
        return "workflow_run"
    if _ns_get(namespace, "scaffold", None):
        return "scaffold"
    if _ns_get(namespace, "evolve", False):
        return "evolve"
    if _ns_get(namespace, "status", False):
        return "goal_status"
    if _ns_get(namespace, "add_goal", None) and _ns_get(namespace, "run_goals", False):
        return "goal_add_run"
    if _ns_get(namespace, "add_goal", None):
        return "goal_add"
    if _ns_get(namespace, "goal", None):
        return "goal_once"
    if _ns_get(namespace, "run_goals", False):
        return "goal_run"
    return "interactive"


def match_action_from_rules(*, command: str | None, subcommand: str | None, namespace: Any, uses_subcommand: bool) -> str:
    if uses_subcommand:
        return match_canonical_action(command, subcommand, namespace)
    return match_legacy_action(namespace)


def render_examples(spec: CommandSpec) -> str:
    if not spec.examples:
        return ""
    return "\n".join(["Examples:", *[f"  {example}" for example in spec.examples]])


def spec_argparse_kwargs(path: Sequence[str]) -> dict[str, Any]:
    spec = COMMAND_SPECS_BY_PATH[tuple(path)]
    kwargs: dict[str, Any] = {"help": spec.summary}
    if spec.description:
        kwargs["description"] = spec.description
    if spec.examples:
        kwargs["epilog"] = render_examples(spec)
    return kwargs


def help_schema() -> dict[str, Any]:
    commands: list[dict[str, Any]] = []
    for spec in COMMAND_SPECS:
        default_action = _CANONICAL_PATH_TO_ACTION.get(spec.path)
        if spec.path == ("goal", "add"):
            default_action = "goal_add"
        action_spec = CLI_ACTION_SPECS_BY_ACTION.get(default_action) if default_action else None

        flags = []
        if default_action and default_action not in {"logs", "watch", "studio", "interactive"}:
            flags.append({"name": "--json", "summary": "Output JSON instead of text"})

        commands.append(
            {
                "path": list(spec.path),
                "summary": spec.summary,
                "description": spec.description,
                "examples": list(spec.examples),
                "legacy_flags": list(spec.legacy_flags),
                "action": default_action,
                "requires_runtime": action_spec.requires_runtime if action_spec else None,
                "flags": flags,
            }
        )
    return {
        "version": HELP_SCHEMA_VERSION,
        "generated_by": HELP_SCHEMA_GENERATED_BY,
        "deterministic": True,
        "json_contracts": {
            CLI_WARNINGS_JSON_FIELD: {
                "field": CLI_WARNINGS_JSON_FIELD,
                "version": CLI_WARNINGS_JSON_CONTRACT_VERSION,
                "description": "Structured CLI warnings attached to JSON outputs when legacy flags are used.",
                "inclusion_rule": ("Present when a parsed invocation emits structured warning records and the command outputs JSON."),
                "record_fields": list(CLI_WARNINGS_RECORD_FIELDS),
                "record_codes": [dict(item) for item in CLI_WARNINGS_RECORD_CODES],
            },
            CLI_ERRORS_JSON_CONTRACT_NAME: {
                "version": CLI_ERRORS_JSON_CONTRACT_VERSION,
                "description": "Structured CLI JSON error payloads for parse and help-topic failures.",
                "inclusion_rule": "Returned as the top-level JSON payload when `--json` is used and an error occurs.",
                "record_fields": list(CLI_ERRORS_RECORD_FIELDS),
                "optional_fields": list(CLI_ERRORS_OPTIONAL_FIELDS),
                "record_codes": [dict(item) for item in CLI_ERRORS_RECORD_CODES],
            },
        },
        "commands": commands,
    }


__all__ = [
    "CLI_ERRORS_JSON_CONTRACT_NAME",
    "CLI_ERRORS_JSON_CONTRACT_VERSION",
    "CLI_ERRORS_OPTIONAL_FIELDS",
    "CLI_ERRORS_RECORD_CODES",
    "CLI_ERRORS_RECORD_FIELDS",
    "CLI_PARSE_ERROR_CODE",
    "CLI_WARNINGS_JSON_CONTRACT_VERSION",
    "CLI_WARNINGS_CODE_LEGACY_FLAGS_DEPRECATED",
    "CLI_WARNINGS_JSON_FIELD",
    "CLI_WARNINGS_RECORD_CODES",
    "CLI_WARNINGS_RECORD_FIELDS",
    "CLIActionSpec",
    "CLI_ACTION_SPECS",
    "CLI_ACTION_SPECS_BY_ACTION",
    "CommandSpec",
    "COMMAND_SPECS",
    "COMMAND_SPECS_BY_PATH",
    "action_default_canonical_path",
    "action_display_command",
    "action_smoke_argv",
    "action_display_subcommand",
    "action_runtime_required",
    "help_schema",
    "HELP_SCHEMA_GENERATED_BY",
    "HELP_SCHEMA_VERSION",
    "iter_cli_action_specs",
    "legacy_allowed_multi_primary_flag_sets",
    "legacy_auxiliary_flag_names",
    "legacy_primary_flag_names",
    "match_action_from_rules",
    "match_canonical_action",
    "match_legacy_action",
    "render_examples",
    "spec_argparse_kwargs",
    "UNKNOWN_COMMAND_HELP_TOPIC_CODE",
]
