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

Pre-merge validation (run before opening a PR):
- `python3 scripts/pre_merge_check.py` — validates CLI reference, help snapshots, and sweep artifact tests are all current. Exits non-zero with coloured ✓/✗ output if any check fails.

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
- [`readiness`](#readiness)
- [`bootstrap`](#bootstrap)
- [`config`](#config)
  - [`aura config set`](#aura-config-set)
- [`contract-report`](#contract-report)
- [`diag`](#diag)
- [`logs`](#logs)
- [`history`](#history)
- [`watch`](#watch)
- [`studio`](#studio)
- [`goal`](#goal)
  - [`aura goal add`](#aura-goal-add)
  - [`aura goal run`](#aura-goal-run)
  - [`aura goal status`](#aura-goal-status)
  - [`aura goal once`](#aura-goal-once)
  - [`aura goal resume`](#aura-goal-resume)
- [`workflow`](#workflow)
  - [`aura workflow run`](#aura-workflow-run)
- [`mcp`](#mcp)
  - [`aura mcp tools`](#aura-mcp-tools)
  - [`aura mcp call`](#aura-mcp-call)
  - [`aura mcp status`](#aura-mcp-status)
  - [`aura mcp restart`](#aura-mcp-restart)
- [`beads`](#beads)
  - [`aura beads schemas`](#aura-beads-schemas)
- [`scaffold`](#scaffold)
- [`evolve`](#evolve)
- [`queue`](#queue)
  - [`aura queue list`](#aura-queue-list)
  - [`aura queue clear`](#aura-queue-clear)
- [`memory`](#memory)
  - [`aura memory search`](#aura-memory-search)
  - [`aura memory reindex`](#aura-memory-reindex)
- [`metrics`](#metrics)
- [`sadd`](#sadd)
  - [`aura sadd run`](#aura-sadd-run)
  - [`aura sadd status`](#aura-sadd-status)
  - [`aura sadd resume`](#aura-sadd-resume)
- [`innovate`](#innovate)
  - [`aura innovate start`](#aura-innovate-start)
  - [`aura innovate list`](#aura-innovate-list)
  - [`aura innovate show`](#aura-innovate-show)
  - [`aura innovate resume`](#aura-innovate-resume)
  - [`aura innovate export`](#aura-innovate-export)
  - [`aura innovate techniques`](#aura-innovate-techniques)
  - [`aura innovate to-goals`](#aura-innovate-to-goals)
  - [`aura innovate insights`](#aura-innovate-insights)
- [`agent`](#agent)
  - [`aura agent run`](#aura-agent-run)
  - [`aura agent list`](#aura-agent-list)
- [`credentials`](#credentials)
  - [`aura credentials migrate`](#aura-credentials-migrate)
  - [`aura credentials store`](#aura-credentials-store)
  - [`aura credentials delete`](#aura-credentials-delete)
  - [`aura credentials status`](#aura-credentials-status)
- [`cancel`](#cancel)

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

## `readiness`

### `aura readiness`

Check V2 runtime readiness

Validate async runtime and MCP registry health.

`action`: `readiness` `requires_runtime`: `true`

Examples:
- `python3 main.py readiness`

## `bootstrap`

### `aura bootstrap`

[EXPERIMENTAL] Create default config

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

### `aura config set`

Set a config value

Persist a configuration key-value pair to aura.config.json.
Use dotted model paths like 'model.code_generation' to set model routing.

`action`: `config_set` `requires_runtime`: `false`

Examples:
- `python3 main.py config set model.code_generation google/gemini-2.5-pro`
- `python3 main.py config set dry_run true`

## `contract-report`

### `aura contract-report`

[EXPERIMENTAL] Print CLI contract report

[EXPERIMENTAL] Print aggregated parser/help/schema/dispatch contract checks as JSON.

`action`: `contract_report` `requires_runtime`: `false`

Examples:
- `python3 main.py contract-report --check`
- `python3 main.py contract-report --compact`

## `diag`

### `aura diag`

[EXPERIMENTAL] MCP diagnostics snapshot

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

## `history`

### `aura history`

Show completed goal history

List the last N completed goals with their scores and timestamps from the goal archive.

`action`: `history` `requires_runtime`: `true`

Examples:
- `python3 main.py history`
- `python3 main.py history --limit 20`
- `python3 main.py history --json`

## `watch`

### `aura watch`

[EXPERIMENTAL] Launch TUI monitor

[EXPERIMENTAL] Launch the AuraStudio terminal UI. Use --autonomous to start the goal loop.

`action`: `watch` `requires_runtime`: `true`

Examples:
- `python3 main.py watch`
- `python3 main.py watch --autonomous`

## `studio`

### `aura studio`

[EXPERIMENTAL] Launch AURA Studio

[EXPERIMENTAL] Launch the rich real-time dashboard. Use --autonomous to start the goal loop.

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
- `python3 main.py goal status`
- `python3 main.py goal status --json`

### `aura goal once`

Run a one-off goal

Run a single goal directly without queueing it.

`action`: `goal_once` `requires_runtime`: `true`

Legacy flags:
- `--goal`

Examples:
- `python3 main.py goal once "Summarize repo"`
- `python3 main.py goal once "Refactor core" --max-cycles 3`

### `aura goal resume`

Resume an interrupted goal

Re-queue a goal that was interrupted mid-execution due to a crash or process kill. Reads memory/in_flight_goal.json written by the goal run loop. Use --run to immediately execute the re-queued goal.

`action`: `goal_resume` `requires_runtime`: `false`

Examples:
- `python3 main.py goal resume`
- `python3 main.py goal resume --run`

## `workflow`

### `aura workflow`

Workflow operations

Run orchestrated workflow goals with explicit cycle limits.

Examples:
- `python3 main.py workflow run "Summarize repo"`

### `aura workflow run`

[EXPERIMENTAL] Run a workflow goal

Run the orchestrator loop for a single workflow goal.

`action`: `workflow_run` `requires_runtime`: `true`

Legacy flags:
- `--workflow-goal`

Examples:
- `python3 main.py workflow run "Summarize repo" --max-cycles 3`

## `mcp`

### `aura mcp`

Repo-local MCP commands

Inspect and call MCP servers configured in the repo-local MCP config.

Examples:
- `python3 main.py mcp tools`

### `aura mcp tools`

[EXPERIMENTAL] List MCP tools

List servers and tools exposed by the repo-local MCP config.

`action`: `mcp_tools` `requires_runtime`: `false`

Legacy flags:
- `--mcp-tools`

Examples:
- `python3 main.py mcp tools`

### `aura mcp call`

[EXPERIMENTAL] Call an MCP tool

Invoke a repo-local MCP server/tool target with optional JSON args.

`action`: `mcp_call` `requires_runtime`: `false`

Legacy flags:
- `--mcp-call`

Examples:
- `python3 main.py mcp call filesystem/read_file --args '{"path":"README.md"}'`
- `python3 main.py mcp call sqlite/query --args '{"sql":"select 1"}'`

### `aura mcp status`

Show MCP server health dashboard

Query all registered MCP servers and render a Rich table showing server name, endpoint URL, health status, last heartbeat, average response latency, and tool count.

`action`: `mcp_status` `requires_runtime`: `false`

Examples:
- `python3 main.py mcp status`

### `aura mcp restart`

Restart / validate an MCP server

Trigger a manual health validation pass for a named MCP server. Pass the server config-name (e.g. 'dev_tools', 'skills').

`action`: `mcp_restart` `requires_runtime`: `false`

Examples:
- `python3 main.py mcp restart dev_tools`
- `python3 main.py mcp restart skills`

## `beads`

### `aura beads`

BEADS contract commands

Inspect and validate BEADS schema contracts.

Examples:
- `python3 main.py beads schemas`

### `aura beads schemas`

List registered BEADS schemas

Print a table of all BEADS schema contracts registered in .beads/. Includes schema version, TypedDict names, and interaction count.

`action`: `beads_schemas` `requires_runtime`: `false`

Examples:
- `python3 main.py beads schemas`
- `python3 main.py beads schemas --json`

## `scaffold`

### `aura scaffold`

Scaffold a project

Run the scaffolder agent for a named project type.

`action`: `scaffold` `requires_runtime`: `true`

Legacy flags:
- `--scaffold`

Examples:
- `python3 main.py scaffold demo --desc "small demo app"`
- `python3 main.py scaffold demo --json`

## `evolve`

### `aura evolve`

Run innovation workflow

Run the innovation workflow and optional queue-backed implementation loop for AURA core.

`action`: `evolve` `requires_runtime`: `true`

Legacy flags:
- `--evolve`

Examples:
- `python3 main.py evolve`
- `python3 main.py evolve --json`
- `python3 main.py evolve --queue-only --proposal-limit 3 --focus research`

## `queue`

### `aura queue`

Goal queue management

List, add, or clear goals in the autonomous queue.

Examples:
- `python3 main.py queue list`
- `python3 main.py queue list --json`

### `aura queue list`

List queued goals

Show all pending and completed goals.

`action`: `queue_list` `requires_runtime`: `true`

Examples:
- `python3 main.py queue list`
- `python3 main.py queue list --json`

### `aura queue clear`

Clear the goal queue

Remove all pending goals from the queue.

`action`: `queue_clear` `requires_runtime`: `true`

Examples:
- `python3 main.py queue clear`
- `python3 main.py queue clear --json`

## `memory`

### `aura memory`

Semantic memory operations

Search or browse through the AURA brain.

Examples:
- `python3 main.py memory search "workflow engine"`
- `python3 main.py memory search "workflow engine" --json`

### `aura memory search`

[EXPERIMENTAL] Search semantic memory

Perform a semantic search over brain entries.

`action`: `memory_search` `requires_runtime`: `true`

Examples:
- `python3 main.py memory search "workflow engine"`
- `python3 main.py memory search "workflow engine" --json`

### `aura memory reindex`

[EXPERIMENTAL] Rebuild semantic memory embeddings

Rebuild semantic memory embeddings for the active model and force a project sync.

`action`: `memory_reindex` `requires_runtime`: `true`

Examples:
- `python3 main.py memory reindex`
- `python3 main.py memory reindex --json`

## `metrics`

### `aura metrics`

Show performance metrics

Display cycle success rates and timing stats.

`action`: `metrics_show` `requires_runtime`: `true`

Examples:
- `python3 main.py metrics`
- `python3 main.py metrics --json`

## `sadd`

### `aura sadd`

[EXPERIMENTAL] Sub-Agent Driven Development

Decompose a design spec into parallel workstreams and execute via sub-agents.

Examples:
- `python3 main.py sadd run --spec design.md --dry-run`
- `python3 main.py sadd run --spec design.md --max-parallel 3`

### `aura sadd run`

Run a SADD session

Parse a design spec and execute workstreams. Use --dry-run to preview decomposition only.

`action`: `sadd_run` `requires_runtime`: `true`

Examples:
- `python3 main.py sadd run --spec design.md --dry-run`
- `python3 main.py sadd run --spec design.md --max-parallel 2 --max-cycles 3`

### `aura sadd status`

Show SADD session status

Show status of recent SADD sessions or a specific session.

`action`: `sadd_status` `requires_runtime`: `false`

Examples:
- `python3 main.py sadd status`
- `python3 main.py sadd status --session-id <id>`

### `aura sadd resume`

Resume a SADD session

Resume an interrupted SADD session from its last checkpoint.

`action`: `sadd_resume` `requires_runtime`: `true`

Examples:
- `python3 main.py sadd resume --session-id <id>`

## `innovate`

### `aura innovate`

Innovation Catalyst session management

Start, list, and manage innovation sessions using brainstorming techniques.

Examples:
- `python3 main.py innovate start "How to improve X?"`
- `python3 main.py innovate list`
- `python3 main.py innovate show --session-id abc123`

### `aura innovate start`

Start a new innovation session

Start a new innovation session with the Innovation Catalyst framework.

`action`: `innovate_start` `requires_runtime`: `true`

Examples:
- `python3 main.py innovate start "How to improve code review?"`
- `python3 main.py innovate start "Reduce bugs" --techniques scamper,six_hats`
- `python3 main.py innovate start "Improve UX" --execute-phase divergence`

### `aura innovate list`

List innovation sessions

List all innovation sessions with their status and metrics.

`action`: `innovate_list` `requires_runtime`: `true`

Examples:
- `python3 main.py innovate list`
- `python3 main.py innovate list --limit 10`
- `python3 main.py innovate list --output json`

### `aura innovate show`

Show session details

Show detailed information about a specific innovation session.

`action`: `innovate_show` `requires_runtime`: `true`

Examples:
- `python3 main.py innovate show --session-id abc123`
- `python3 main.py innovate show --session-id abc123 --output json`

### `aura innovate resume`

Resume an innovation session

Resume an innovation session at a specific phase.

`action`: `innovate_resume` `requires_runtime`: `true`

Examples:
- `python3 main.py innovate resume --session-id abc123`
- `python3 main.py innovate resume --session-id abc123 --phase convergence`

### `aura innovate export`

Export session results

Export innovation session results to markdown or JSON.

`action`: `innovate_export` `requires_runtime`: `true`

Examples:
- `python3 main.py innovate export --session-id abc123 --format markdown`
- `python3 main.py innovate export --session-id abc123 --output report.md`

### `aura innovate techniques`

List available brainstorming techniques

Show all available brainstorming techniques with descriptions.

`action`: `innovate_techniques` `requires_runtime`: `false`

Examples:
- `python3 main.py innovate techniques`
- `python3 main.py innovate techniques --json`

### `aura innovate to-goals`

Convert selected ideas to goals

Convert selected ideas from an innovation session to goals in the queue.

`action`: `innovate_to_goals` `requires_runtime`: `true`

Examples:
- `python3 main.py innovate to-goals --session-id abc123`
- `python3 main.py innovate to-goals --session-id abc123 --preview`

### `aura innovate insights`

Show innovation analytics and insights

Display analytics about innovation sessions including trends, technique effectiveness, and idea quality metrics.

`action`: `innovate_insights` `requires_runtime`: `true`

Examples:
- `python3 main.py innovate insights`
- `python3 main.py innovate insights --session-id abc123`
- `python3 main.py innovate insights --json`

## `agent`

### `aura agent`

Agent SDK commands

Commands for the Agent SDK meta-controller.

### `aura agent run`

Run goal via Agent SDK meta-controller

Execute a development goal using Claude-as-brain orchestration with dynamic tool/skill/workflow selection.

`action`: `agent_run` `requires_runtime`: `true`

### `aura agent list`

List registered agents

Display all registered AURA agents, their type, and status.

`action`: `agent_list` `requires_runtime`: `false`

Examples:
- `python3 main.py agent list`

## `credentials`

### `aura credentials`

Secure credential management

Manage API keys and credentials in secure storage (OS keyring or encrypted file).

Examples:
- `python3 main.py credentials migrate`
- `python3 main.py credentials migrate --yes`
- `python3 main.py credentials store --key api_key`
- `python3 main.py credentials delete --key api_key --yes`

### `aura credentials migrate`

Migrate credentials to secure storage

Migrate API keys from plaintext aura.config.json to secure credential store (OS keyring).

`action`: `credentials_migrate` `requires_runtime`: `false`

Examples:
- `python3 main.py credentials migrate`
- `python3 main.py credentials migrate --yes`

### `aura credentials store`

Store a credential securely

Store an API key or credential in the secure credential store.

`action`: `credentials_store` `requires_runtime`: `false`

Examples:
- `python3 main.py credentials store --key api_key`
- `python3 main.py credentials store --key api_key --value sk-xxx`

### `aura credentials delete`

Delete a stored credential

Remove a credential from the secure credential store.

`action`: `credentials_delete` `requires_runtime`: `false`

Examples:
- `python3 main.py credentials delete --key api_key`
- `python3 main.py credentials delete --key api_key --yes`

### `aura credentials status`

Show credential storage status

Display information about the credential store configuration and stored credentials.

`action`: `credentials_status` `requires_runtime`: `false`

Examples:
- `python3 main.py credentials status`
- `python3 main.py credentials status --json`

## `cancel`

### `aura cancel`

Cancel an active pipeline run

Send a cancellation signal to a running AURA pipeline and verify that no partial filesystem changes remain on disk.

Exit codes:
  0 — run cancelled and filesystem restored
  1 — run_id not found in the active-run registry
  2 — cancellation signal sent but rollback verification failed

`action`: `cancel` `requires_runtime`: `false`

Examples:
- `python3 main.py cancel <run-id>`
