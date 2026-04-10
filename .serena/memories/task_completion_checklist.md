# Task Completion Checklist

When a task is completed, verify:

1. **Tests pass:** `python3 -m pytest tests/ --no-cov -q --timeout=120`
2. **CLI docs current:** If CLI commands changed, run `python3 scripts/generate_cli_reference.py --check`
3. **Snapshot tests:** If CLI output changed, update `tests/snapshots/` and run `python3 -m pytest -k snapshot -q`
4. **No secrets committed:** Check `git diff --cached` for API keys or `.env` content
5. **Type hints:** New/modified functions should have parameter + return type annotations
6. **Imports clean:** No unused imports in modified files
7. **Commit message:** Imperative, sentence-case (e.g., "Add goal-status snapshot tests")
