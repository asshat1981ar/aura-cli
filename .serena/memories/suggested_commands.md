# Suggested Commands

## Testing
```bash
python3 -m pytest                              # Run all tests
python3 -m pytest tests/test_foo.py -v         # Run specific test file
python3 -m pytest -k "snapshot" -q             # Run tests matching keyword
python3 -m pytest tests/agents/ --no-cov -q    # Agent tests without coverage
```

## CLI Usage
```bash
python3 main.py --help                         # Show CLI help
python3 main.py doctor                         # System diagnostics
python3 main.py goal once "task" --max-cycles 3  # Run a single goal
python3 main.py sadd run --spec spec.md        # Run SADD session
python3 main.py config                         # Show effective config
```

## Development
```bash
AURA_SKIP_CHDIR=1 python3 main.py ...         # Keep CWD (recommended for dev)
python3 scripts/generate_cli_reference.py      # Regenerate CLI docs
python3 scripts/generate_cli_reference.py --check  # Verify docs are current
```

## Git
```bash
git status                                     # Check changes
git diff                                       # See unstaged changes
git log --oneline -10                          # Recent commits
```
