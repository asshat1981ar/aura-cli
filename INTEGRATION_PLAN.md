# Integration Plan — Connecting Sub-Agent Deliverables

## Objective
Wire up all new components into a cohesive system.

## Integration Tasks

### Task 1: Integrate Error Presenter into CLI
**File**: `aura_cli/cli_main.py`
- Import `error_presenter`
- Wrap main() with error boundary
- Add --json and --verbose flags

### Task 2: Initialize DI Container
**File**: `aura_cli/cli_main.py`
- Import `Container` from `core.container`
- Register core services on startup
- Ensure singleton behavior

### Task 3: Add Retry to Network Calls
**File**: `core/model_adapter.py` or network modules
- Import `with_retry` from `core.retry`
- Decorate API call methods
- Configure retryable error codes

### Task 4: Use Pydantic Config
**File**: `core/config_manager.py` or config loading
- Import `AuraConfig` from `core.config_schema`
- Replace dict-based config with Pydantic models
- Add validation on load

### Task 5: Test Integration
- Run full test suite
- Verify CLI commands work
- Check error handling flows
