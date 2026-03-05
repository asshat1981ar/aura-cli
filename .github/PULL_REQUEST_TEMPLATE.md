## Summary

<!-- Briefly describe what this PR does and why. -->

## Changes

<!-- List the files changed and what was changed in each. -->

## Testing

<!-- Describe how you tested these changes (e.g., `python3 -m pytest`, manual steps). -->

## Related Issues / PRs

<!-- Link to any related issues or PRs (e.g., Fixes #123, Closes #456). -->

## Checklist

- [ ] Code follows the project style (4-space indent, `snake_case` for functions/modules, `PascalCase` for classes)
- [ ] Tests added or updated for new functionality
- [ ] CLI reference regenerated if CLI commands, help text, or JSON output changed:
  ```
  python3 scripts/generate_cli_reference.py
  ```
- [ ] Snapshot files updated if test output changed (`tests/snapshots/`)
- [ ] No secrets committed (`aura.config.json` uses placeholder `api_key` only; real key comes from `AURA_API_KEY` env var)
- [ ] `docs/INTEGRATION_MAP.md` updated if architecture or runtime wiring changed
- [ ] Documentation links use repo-relative paths (not absolute or platform-specific paths)
