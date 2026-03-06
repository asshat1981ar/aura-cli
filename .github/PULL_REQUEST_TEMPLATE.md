## Summary

<!-- What does this PR do? Provide a concise, one-paragraph description. -->

## Changes

<!-- List every file modified and what was changed in each one. -->

- **`path/to/file.py`**: Description of change.

## Motivation

<!-- Why is this change needed? What problem does it solve or what improvement does it make? -->

## Testing

<!-- How was this tested? List the commands you ran and their results. -->

```bash
python3 -m pytest
```

- [ ] All existing tests pass.
- [ ] New tests added (if applicable).
- [ ] Manual testing performed: <!-- describe steps -->

## CI / Checklist

- [ ] `python3 -m pytest` passes.
- [ ] CLI docs are current: `python3 scripts/generate_cli_reference.py --check` (run if CLI commands changed).
- [ ] No secrets or credentials committed.
- [ ] `aura.config.json` API key left as placeholder (real value comes from `AURA_API_KEY` env var).
- [ ] Documentation updated (if applicable).
- [ ] Breaking changes documented (if applicable).
