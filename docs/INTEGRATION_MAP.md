# AURA Integration Map

This document describes the current AURA runtime architecture as it exists in
the codebase today. It is intended as a practical orientation guide for
developers working on the CLI, orchestrator, operator surfaces, local models,
and BEADS integration.

## Canonical Entrypoints

- [main.py](/data/data/com.termux/files/home/aura_cli/aura-cli/main.py)
  Lightweight shim. It handles early CLI parsing, `help`, `--json-help`, and
  error formatting, then delegates to the real CLI runtime.
- [aura_cli/cli_main.py](/data/data/com.termux/files/home/aura_cli/aura-cli/aura_cli/cli_main.py)
  Canonical entrypoint for runtime creation, dispatch, watch/studio flows,
  queue operations, memory tools, doctor/config commands, and workflow runs.
- [run_aura.sh](/data/data/com.termux/files/home/aura_cli/aura-cli/run_aura.sh)
  Shell wrapper used for local startup and Android local-model health gating.

## Runtime Assembly

Runtime creation is centralized in
[aura_cli/cli_main.py](/data/data/com.termux/files/home/aura_cli/aura-cli/aura_cli/cli_main.py)
via `create_runtime(project_root, overrides=None)`.

### Runtime modes

- `queue`
  Queue-only runtime used for lightweight status/add/interactive flows.
  It avoids full model, brain, vector-store, and orchestrator initialization.
- `lean`
  Minimal execution runtime used for some single-goal and dry-run flows.
- `full`
  Full runtime with brain, model adapter, vector store, memory store,
  orchestrator, background sync, improvement loops, and optional BEADS bridge.

### Core runtime objects

`create_runtime()` is responsible for wiring:

- [core/goal_queue.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/goal_queue.py)
- [core/goal_archive.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/goal_archive.py)
- [memory/brain.py](/data/data/com.termux/files/home/aura_cli/aura-cli/memory/brain.py)
- [core/model_adapter.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/model_adapter.py)
- [core/vector_store.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/vector_store.py)
- [memory/store.py](/data/data/com.termux/files/home/aura_cli/aura-cli/memory/store.py)
- [core/orchestrator.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/orchestrator.py)
- [core/policy.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/policy.py)
- [core/beads_bridge.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/beads_bridge.py)

## Command Dispatch

The dispatch layer also lives in
[aura_cli/cli_main.py](/data/data/com.termux/files/home/aura_cli/aura-cli/aura_cli/cli_main.py).

### Main dispatch flow

1. Parse args in [main.py](/data/data/com.termux/files/home/aura_cli/aura-cli/main.py).
2. Call `aura_cli.cli_main.main(...)`.
3. Resolve the action with `_resolve_dispatch_action(...)`.
4. Build runtime only when required by the selected command.
5. Execute the handler registered in `COMMAND_DISPATCH_REGISTRY`.

### Major command families

- Help and reporting:
  - `help`
  - `json_help`
  - `show_config`
  - `contract_report`
- Diagnostics:
  - `doctor`
  - `diag`
  - `logs`
- Goal and queue control:
  - `goal_add`
  - `goal_run`
  - `goal_once`
  - `goal_status`
  - `queue_list`
  - `queue_clear`
- Runtime/operator surfaces:
  - `watch`
  - `studio`
  - `workflow_run`
- Memory tools:
  - `memory_search`
  - `memory_reindex`
- Self-improvement:
  - `evolve`

Human-readable status rendering still lives in
[aura_cli/commands.py](/data/data/com.termux/files/home/aura_cli/aura-cli/aura_cli/commands.py),
but command selection and runtime setup are controlled by
[aura_cli/cli_main.py](/data/data/com.termux/files/home/aura_cli/aura-cli/aura_cli/cli_main.py).

## Orchestrator Pipeline

The main execution engine is
[core/orchestrator.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/orchestrator.py),
implemented by `LoopOrchestrator`.

### Single-cycle flow

`run_cycle(goal, dry_run=False)` executes the canonical phase pipeline:

1. ingest
2. skill dispatch
3. optional BEADS gate
4. planning loop
5. critique
6. synthesize
7. act
8. sandbox
9. apply
10. verify
11. reflect
12. record cycle outcome

### Multi-cycle flow

`run_loop(goal, max_cycles=5, dry_run=False)` repeats `run_cycle(...)` until:

- the cycle itself sets a stop reason
- the policy stops execution
- max cycles are reached

### Improvement loops

Optional loops are attached after construction and run from
`_record_cycle_outcome(...)`. These include reflection, health monitoring,
adaptive skill weighting, convergence escape, compaction, autonomous discovery,
evolution, and periodic BEADS sync.

The legacy [core/hybrid_loop.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/hybrid_loop.py)
still exists, but the active runtime path for CLI/server/operator flows is the
`LoopOrchestrator` in [core/orchestrator.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/orchestrator.py).

## Operator Surfaces

Shared operator-facing summaries are normalized in
[core/operator_runtime.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/operator_runtime.py).

### Shared summary helpers

- `build_queue_summary(...)`
- `build_cycle_summary(...)`
- `build_beads_runtime_metadata(...)`
- `build_operator_runtime_snapshot(...)`

These helpers feed:

- [aura_cli/commands.py](/data/data/com.termux/files/home/aura_cli/aura-cli/aura_cli/commands.py)
  for `goal status`
- [aura_cli/server.py](/data/data/com.termux/files/home/aura_cli/aura-cli/aura_cli/server.py)
  for SSE event payloads
- [aura_cli/tui/app.py](/data/data/com.termux/files/home/aura_cli/aura-cli/aura_cli/tui/app.py)
  and [aura_cli/tui/panels/cycle_panel.py](/data/data/com.termux/files/home/aura_cli/aura-cli/aura_cli/tui/panels/cycle_panel.py)
  for watch/studio rendering

## BEADS Integration

BEADS is integrated as a control-plane decision layer, not as the primary code
executor.

### Key files

- [core/beads_contract.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/beads_contract.py)
  defines the Python-side schema.
- [core/beads_bridge.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/beads_bridge.py)
  builds runtime input and invokes the Node bridge.
- [scripts/beads_bridge.mjs](/data/data/com.termux/files/home/aura_cli/aura-cli/scripts/beads_bridge.mjs)
  is the Node adapter around `@beads/bd`.
- [core/orchestrator.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/orchestrator.py)
  applies the BEADS gate before planning and records BEADS decisions into cycle
  outputs and summaries.

### BEADS responsibilities

- build queue + context payloads
- load PRD context from `plans/beads-orchestrator-prd.md`
- load active conductor track metadata from `conductor/tracks/*/metadata.json`
- return structured allow/block/revise decisions
- expose runtime metadata for CLI, TUI, and SSE

## Local Models And Android Path

Local model routing and Android workflows are now first-class parts of the
runtime.

### Key files

- [core/runtime_auth.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/runtime_auth.py)
  resolves provider readiness and local embedding/chat availability.
- [core/model_adapter.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/model_adapter.py)
  handles provider calls, role routing, local model profiles, and embeddings.
- [aura.config.json](/data/data/com.termux/files/home/aura_cli/aura-cli/aura.config.json)
  contains real project config.
- [aura.config.android.example.json](/data/data/com.termux/files/home/aura_cli/aura-cli/aura.config.android.example.json)
  documents the Android local-model layout.
- [scripts/setup_llama_cpp_termux.sh](/data/data/com.termux/files/home/aura_cli/aura-cli/scripts/setup_llama_cpp_termux.sh)
- [scripts/run_android_local_models.sh](/data/data/com.termux/files/home/aura_cli/aura-cli/scripts/run_android_local_models.sh)
- [scripts/check_android_local_models.py](/data/data/com.termux/files/home/aura_cli/aura-cli/scripts/check_android_local_models.py)
- [scripts/run_android_aura.sh](/data/data/com.termux/files/home/aura_cli/aura-cli/scripts/run_android_aura.sh)
- [docs/LOCAL_MODELS_ANDROID.md](/data/data/com.termux/files/home/aura_cli/aura-cli/docs/LOCAL_MODELS_ANDROID.md)

## Persistence

There is no single persistence mechanism; AURA uses several:

- queue/archive JSON:
  - [memory/goal_queue.json](/data/data/com.termux/files/home/aura_cli/aura-cli/memory/goal_queue.json)
  - [memory/goal_archive_v2.json](/data/data/com.termux/files/home/aura_cli/aura-cli/memory/goal_archive_v2.json)
- brain SQLite:
  - [memory/brain_v2.db](/data/data/com.termux/files/home/aura_cli/aura-cli/memory/brain_v2.db)
- memory store log and project memory:
  - [memory/store](/data/data/com.termux/files/home/aura_cli/aura-cli/memory/store)
- semantic memory / project sync support:
  - [core/vector_store.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/vector_store.py)
  - [core/project_syncer.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/project_syncer.py)
- task hierarchy persistence:
  - [memory/task_hierarchy_v2.json](/data/data/com.termux/files/home/aura_cli/aura-cli/memory/task_hierarchy_v2.json)

## Configuration

The unified config layer is
[core/config_manager.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/config_manager.py).

Precedence is:

1. runtime overrides
2. environment variables
3. config file (`settings.json` or `aura.config.json`)
4. defaults

Notable nested config groups:

- `beads`
- `model_routing`
- `local_model_profiles`
- `local_model_routing`
- `semantic_memory`

## Testing

Tests live primarily under [tests/](/data/data/com.termux/files/home/aura_cli/aura-cli/tests).

Notable coverage areas:

- CLI dispatch and snapshots:
  - [tests/test_cli_main_dispatch.py](/data/data/com.termux/files/home/aura_cli/aura-cli/tests/test_cli_main_dispatch.py)
  - [tests/test_cli_help_snapshots.py](/data/data/com.termux/files/home/aura_cli/aura-cli/tests/test_cli_help_snapshots.py)
- orchestrator/runtime:
  - [tests/test_orchestrator_phases.py](/data/data/com.termux/files/home/aura_cli/aura-cli/tests/test_orchestrator_phases.py)
  - [tests/test_operator_runtime.py](/data/data/com.termux/files/home/aura_cli/aura-cli/tests/test_operator_runtime.py)
- BEADS:
  - [tests/test_beads_bridge.py](/data/data/com.termux/files/home/aura_cli/aura-cli/tests/test_beads_bridge.py)
  - [tests/test_beads_skill.py](/data/data/com.termux/files/home/aura_cli/aura-cli/tests/test_beads_skill.py)
- local models / Android:
  - [tests/test_model_adapter_runtime.py](/data/data/com.termux/files/home/aura_cli/aura-cli/tests/test_model_adapter_runtime.py)
  - [tests/test_android_local_models_healthcheck.py](/data/data/com.termux/files/home/aura_cli/aura-cli/tests/test_android_local_models_healthcheck.py)

## Practical Debugging Order

When debugging runtime behavior, start here:

1. [main.py](/data/data/com.termux/files/home/aura_cli/aura-cli/main.py)
2. [aura_cli/cli_main.py](/data/data/com.termux/files/home/aura_cli/aura-cli/aura_cli/cli_main.py)
3. [core/orchestrator.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/orchestrator.py)
4. [core/operator_runtime.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/operator_runtime.py)
5. the specific subsystem:
   - BEADS: [core/beads_bridge.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/beads_bridge.py)
   - models: [core/model_adapter.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/model_adapter.py)
   - memory: [core/vector_store.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/vector_store.py)
   - queue/task flow: [core/task_handler.py](/data/data/com.termux/files/home/aura_cli/aura-cli/core/task_handler.py)
