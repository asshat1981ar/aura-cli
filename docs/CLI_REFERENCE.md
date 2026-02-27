# CLI Reference

Generated from `aura_cli.options.help_schema`.

Schema version: `3`
Deterministic output: `true`

## Contributor Notes

This file is generated. Update CLI docs and snapshots together when CLI behavior changes.

Recommended update steps:
- `python3 scripts/generate_cli_reference.py`
- `python3 -m pytest -q tests/test_cli_docs_generator.py tests/test_cli_help_snapshots.py tests/test_cli_error_snapshots.py tests/test_cli_main_dispatch.py -k snapshot`
- If output changes intentionally, update `tests/snapshots/*` in the same change.

## JSON Output Contracts

### `cli_errors`

Structured CLI JSON error payloads for parse and help-topic failures.

Field name: `cli_errors`
Version: `1`
Inclusion rule: Returned as the top-level JSON payload when `--json` is used and an error occurs.

Record fields:
- `status`
- `code`
- `message`

Optional fields:
- `usage`

Known record codes:
- `cli_parse_error`: CLI argument parsing or validation failed.
- `unknown_command_help_topic`: Help topic path was not recognized.

### `cli_warnings`

Structured CLI warnings attached to JSON outputs when legacy flags are used.

Field name: `cli_warnings`
Version: `1`
Inclusion rule: Present when a parsed invocation emits structured warning records and the command outputs JSON.

Record fields:
- `code`
- `message`
- `category`
- `action`
- `replacement_command`
- `legacy_flags`
- `phase`

Known record codes:
- `legacy_cli_flags_deprecated` category=`deprecation` phase=`compatibility`: Legacy flat CLI flags were used and mapped to a canonical command.

## Table of Contents

- [`help`](#help)
- [`doctor`](#doctor)
- [`bootstrap`](#bootstrap)
- [`config`](#config)
- [`diag`](#diag)
- [`logs`](#logs)
- [`watch`](#watch)
- [`studio`](#studio)
- [`goal`](#goal)
- [`aura goal add`](#aura-goal-add)
- [`aura goal run`](#aura-goal-run)
- [`aura goal status`](#aura-goal-status)
- [`aura goal once`](#aura-goal-once)
- [`workflow`](#workflow)
- [`aura workflow run`](#aura-workflow-run)
- [`mcp`](#mcp)
- [`aura mcp tools`](#aura-mcp-tools)
- [`aura mcp call`](#aura-mcp-call)
- [`scaffold`](#scaffold)
- [`evolve`](#evolve)

## `help`

### `aura help`

Show CLI help

Show top-level help or help for a specific command path.

`action`: `help` `requires_runtime`: `false`

Examples:
- `python3 main.py help`
- `python3 main.py help goal add`

## `doctor`

### `aura doctor`

Run system diagnostics

Run the AURA doctor checks for local environment health.

`action`: `doctor` `requires_runtime`: `false`

Examples:
- `python3 main.py doctor`

## `bootstrap`

### `aura bootstrap`

Create default config

Bootstrap local configuration files for AURA.

`action`: `bootstrap` `requires_runtime`: `false`

Legacy flags:
- `--bootstrap`

Examples:
- `python3 main.py bootstrap`

## `config`

### `aura config`

Show effective config

Print the resolved effective runtime configuration.

`action`: `show_config` `requires_runtime`: `false`

Examples:
- `python3 main.py config`

## `diag`

### `aura diag`

MCP diagnostics snapshot

Fetch MCP health, metrics, limits, and recent logs via HTTP.

`action`: `diag` `requires_runtime`: `false`

Legacy flags:
- `--diag`

Examples:
- `python3 main.py diag`

## `logs`

### `aura logs`

Stream AURA logs

Tail or follow logs from stdin or a file.

`action`: `logs` `requires_runtime`: `false`

Examples:
- `python3 main.py logs --tail 50`
- `python3 main.py logs --file memory/aura.log --follow`

## `watch`

### `aura watch`

Launch TUI monitor

Launch the AuraStudio terminal UI. Use --autonomous to start the goal loop.

`action`: `watch` `requires_runtime`: `true`

Examples:
- `python3 main.py watch`
- `python3 main.py watch --autonomous`

## `studio`

### `aura studio`

Launch AURA Studio

Launch the rich real-time dashboard. Use --autonomous to start the goal loop.

`action`: `studio` `requires_runtime`: `true`

Examples:
- `python3 main.py studio`
- `python3 main.py studio --autonomous`

## `goal`

### `aura goal`

Goal queue operations

Manage queued goals and run one-off goals.

Examples:
- `python3 main.py goal status`
- `python3 main.py goal add "Refactor queue"`

### `aura goal add`

Add a goal to the queue

Add a goal, optionally running the queue immediately.

`action`: `goal_add` `requires_runtime`: `true`

Legacy flags:
- `--add-goal`
- `--run-goals`

Examples:
- `python3 main.py goal add "Fix tests"`
- `python3 main.py goal add "Fix tests" --run`

### `aura goal run`

Run queued goals

Run the goal queue through the autonomous loop.

`action`: `goal_run` `requires_runtime`: `true`

Legacy flags:
- `--run-goals`

Examples:
- `python3 main.py goal run --dry-run`

### `aura goal status`

Show queue status

Show queued and completed goals.

`action`: `goal_status` `requires_runtime`: `true`

Legacy flags:
- `--status`

Examples:
- `python3 main.py goal status --json`

### `aura goal once`

Run a one-off goal

Run a single goal directly without queueing it.

`action`: `goal_once` `requires_runtime`: `true`

Legacy flags:
- `--goal`

Examples:
- `python3 main.py goal once "Summarize repo" --dry-run`

## `workflow`

### `aura workflow`

Workflow operations

Run orchestrated workflow goals with explicit cycle limits.

Examples:
- `python3 main.py workflow run "Summarize repo"`

### `aura workflow run`

Run a workflow goal

Run the orchestrator loop for a single workflow goal.

`action`: `workflow_run` `requires_runtime`: `true`

Legacy flags:
- `--workflow-goal`

Examples:
- `python3 main.py workflow run "Summarize repo" --max-cycles 3`

## `mcp`

### `aura mcp`

MCP HTTP client commands

Inspect and call MCP tools from the CLI.

Examples:
- `python3 main.py mcp tools`

### `aura mcp tools`

List MCP tools

List tools exposed by the MCP HTTP server.

`action`: `mcp_tools` `requires_runtime`: `false`

Legacy flags:
- `--mcp-tools`

Examples:
- `python3 main.py mcp tools`

### `aura mcp call`

Call an MCP tool

Invoke an MCP tool with optional JSON args.

`action`: `mcp_call` `requires_runtime`: `false`

Legacy flags:
- `--mcp-call`

Examples:
- `python3 main.py mcp call limits`
- `python3 main.py mcp call tail_logs --args '{"lines": 10}'`

## `scaffold`

### `aura scaffold`

Scaffold a project

Run the scaffolder agent for a named project type.

`action`: `scaffold` `requires_runtime`: `true`

Legacy flags:
- `--scaffold`

Examples:
- `python3 main.py scaffold demo --desc "small demo app"`

## `evolve`

### `aura evolve`

Run evolution loop

Run the evolution loop to mutate and improve the system.

`action`: `evolve` `requires_runtime`: `true`

Legacy flags:
- `--evolve`

Examples:
- `python3 main.py evolve`
