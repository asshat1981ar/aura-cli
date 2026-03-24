# Aura-CLI Canary Rollback Playbook

This document provides exact steps to disable the new async orchestrator and typed registry in case of failure during the Milestone 5 rollout.

## Immediate Rollback (Standard)

If you observe unexpected errors, latency spikes, or invalid outputs during a run, perform these steps immediately:

1.  **Disable the New Orchestrator:**
    ```bash
    export AURA_ENABLE_NEW_ORCHESTRATOR=false
    ```
    This stops the canary routing in `_run_phase` and forces all tasks through the legacy sync path.

2.  **Disable the Typed Registry:**
    ```bash
    export AURA_ENABLE_MCP_REGISTRY=false
    ```
    This disables modern agent resolution and MCP discovery.

3.  **Disable Shadow Mode:**
    ```bash
    export AURA_NEW_ORCHESTRATOR_SHADOW_MODE=false
    ```
    This stops the parallel async execution and comparison logging.

## Full Emergency Reversion

If the standard rollback does not resolve the issue (e.g., due to a breaking change in core models):

1.  **Force Legacy Mode:**
    ```bash
    export AURA_FORCE_LEGACY_ORCHESTRATOR=true
    ```
    (Note: This flag should be implemented in `core/orchestrator.py` to completely bypass new logic).

2.  **Unset all experimental flags:**
    ```bash
    unset AURA_ENABLE_MCP_REGISTRY
    unset AURA_ENABLE_NEW_ORCHESTRATOR
    unset AURA_NEW_ORCHESTRATOR_SHADOW_MODE
    ```

## Verification after Rollback

Run the core test suite to ensure the system has returned to a stable legacy state:

```bash
pytest tests/test_agents_unit.py tests/test_cli_main_dispatch.py
```

## Troubleshooting Rollback Issues

- **IndentationErrors or SyntaxErrors:** If the code fails to load, ensure no surgical edits to `core/orchestrator.py` were left in a partial state.
- **Connection Leaks:** If you suspect HTTP client leaks, restart the AURA process to clear the `MCPAsyncClient._client_pool`.
