# Code Style & Conventions

## Python Style
- **Indentation:** 4 spaces
- **Naming:** snake_case (functions/variables/modules), PascalCase (classes), UPPER_SNAKE_CASE (constants)
- **Private members:** Leading underscore (`_my_private_method`)
- **Type hints:** Required throughout (Python 3.10+ syntax)
- **Path handling:** Always use `pathlib.Path`
- **No formatter config** — match surrounding file style

## Key Patterns
- **Factory pattern:** `default_agents()`, `cache_adapter_factory()`
- **Adapter pattern:** Pipeline adapters (`PlannerAdapter`, `CriticAdapter`) in `agents/registry.py`
- **Lazy imports:** Heavy deps (TextBlob, NetworkX) loaded only when needed
- **Atomic file ops:** `tempfile` + `move` for write safety
- **Structured logging:** `log_json(level, event, details)` — automatic secret masking
- **Custom exceptions:** `CLIParseError`, `FileToolsError`, `OldCodeNotFoundError`

## Error Handling
- Distinguish retry-able vs skip-able failures
- Never log raw API keys (automatic masking in `core/logging_utils.py`)
- Use existing exception hierarchy, not bare `Exception`

## Testing Conventions
- File naming: `test_*.py`, classes: `Test*`, methods: `test_*`
- Snapshot tests in `tests/snapshots/` — compare with `_assert_json_snapshot()`
- Mock factories with `MagicMock` for isolation
- Integration tests in `tests/integration/`
