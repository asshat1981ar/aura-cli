# goal Command

The `goal` command is the primary interface for autonomous development in AURA. It manages the goal queue and executes goals through the 10-phase multi-agent pipeline.

## Overview

Goals are natural language descriptions of development tasks. AURA processes them through:
1. **Ingestion** - Gather context from memory and codebase
2. **Planning** - Generate implementation steps
3. **Critique** - Review plan for issues
4. **Synthesis** - Bundle tasks for execution
5. **Act** - Generate code changes
6. **Sandbox** - Execute safely
7. **Verify** - Run tests and validation
8. **Reflect** - Analyze results
9. **Adapt** - Adjust strategy
10. **Archive** - Store results

## Subcommands

### `goal once`

Execute a single goal immediately without adding it to the queue.

```bash
# Basic usage
aura goal once "Add input validation to the login endpoint"

# With dry-run (simulate without changes)
aura goal once "Refactor auth module" --dry-run

# With verbose output
aura goal once "Fix bug in payment flow" --verbose

# With specific model
aura goal once "Optimize database queries" --model gpt-4
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--dry-run` | Simulate execution without making changes | `false` |
| `--verbose` | Enable verbose logging | `false` |
| `--model` | Specify LLM model to use | From config |
| `--context` | Additional context file | None |
| `--priority` | Priority level (1-5) | `3` |

### `goal add`

Add a goal to the queue for later processing.

```bash
# Add to queue
aura goal add "Implement OAuth2 authentication"

# Add and immediately run the queue
aura goal add "Update documentation" --run

# Add with high priority
aura goal add "Fix critical security bug" --priority 1

# Add with context
aura goal add "Refactor user service" --context ./context.md
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--run` | Start processing queue after adding | `false` |
| `--priority` | Priority level (1=highest, 5=lowest) | `3` |
| `--context` | Path to context file | None |
| `--tags` | Comma-separated tags | None |

### `goal run`

Process all goals in the queue.

```bash
# Run all queued goals
aura goal run

# Dry run mode
aura goal run --dry-run

# Run with specific concurrency
aura goal run --max-workers 3

# Run until queue is empty (continuous)
aura goal run --continuous
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--dry-run` | Simulate without changes | `false` |
| `--max-workers` | Parallel goal processing | `1` |
| `--continuous` | Keep running until queue empty | `false` |
| `--timeout` | Max seconds per goal | `300` |

### `goal status`

Show the current state of the goal queue.

```bash
# Human-readable status
aura goal status

# JSON output for scripting
aura goal status --json

# Show detailed info
aura goal status --verbose

# Filter by status
aura goal status --filter pending
```

**Output Example:**

```
┌─────────────────────────────────────────────────────────────┐
│ Goal Queue Status                                           │
├─────────────────────────────────────────────────────────────┤
│ Pending:   3                                                │
│ Running:   1                                                │
│ Completed: 12                                               │
│ Failed:    1                                                │
├─────────────────────────────────────────────────────────────┤
│ Pending Goals:                                              │
│ #14 P2 "Update API documentation"            [docs]         │
│ #15 P3 "Refactor error handling"             [refactor]     │
│ #16 P1 "Fix race condition in cache"         [bugfix]       │
└─────────────────────────────────────────────────────────────┘
```

### `goal list`

List all goals with filtering options.

```bash
# List all goals
aura goal list

# List completed goals
aura goal list --status completed

# List with pagination
aura goal list --limit 10 --offset 20

# Search goals
aura goal list --search "authentication"
```

### `goal remove`

Remove a goal from the queue.

```bash
# Remove by ID
aura goal remove 14

# Remove multiple
aura goal remove 14 15 16

# Remove all completed
aura goal remove --status completed
```

### `goal clear`

Clear the goal queue.

```bash
# Clear all pending goals
aura goal clear

# Clear with confirmation
aura goal clear --force
```

## Goal Lifecycle

```
    ┌──────────┐
    │  QUEUED  │
    └────┬─────┘
         │ goal run
         ▼
    ┌──────────┐
    │ INGEST   │
    └────┬─────┘
         │
         ▼
    ┌──────────┐     ┌──────────┐
    │  PLAN    │────▶│ CRITIQUE │◀──┐
    └────┬─────┘     └────┬─────┘   │ (feedback loop)
         │                └─────────┘
         ▼
    ┌──────────┐
    │  ACT     │
    └────┬─────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌────────┐
│SUCCESS│  │ FAILED │
└───────┘  └────────┘
```

## Best Practices

### Writing Effective Goals

**Good goals:**
```bash
# Specific and actionable
aura goal once "Add email validation regex to User.email field"

# Includes context
aura goal once "Refactor auth middleware to use JWT instead of sessions"

# Has clear success criteria
aura goal once "Update all /api/v1/ endpoints to return 404 for missing resources"
```

**Avoid vague goals:**
```bash
# Too broad
aura goal once "Improve the codebase"

# Unclear success criteria
aura goal once "Make things better"
```

### Queue Management

1. **Prioritize critical fixes** - Use `--priority 1` for bugs
2. **Batch related changes** - Group similar goals
3. **Use tags** - For filtering and organization
4. **Review before running** - Use `--dry-run` for complex goals

## Examples

### Example 1: Bug Fix Workflow

```bash
# Add critical bug fix
aura goal add "Fix SQL injection in search endpoint" --priority 1 --tags security

# Run immediately
aura goal run
```

### Example 2: Feature Development

```bash
# Add feature goals
aura goal add "Design user profile API" --tags feature,api --context ./design.md
aura goal add "Implement GET /api/users/{id}" --tags feature,api
aura goal add "Implement PUT /api/users/{id}" --tags feature,api
aura goal add "Add tests for user profile endpoints" --tags feature,test

# Run all
aura goal run
```

### Example 3: Refactoring Sprint

```bash
# Queue refactoring tasks
aura goal add "Extract validation logic to validators.py" --tags refactor
aura goal add "Consolidate error handling middleware" --tags refactor
aura goal add "Update imports after refactoring" --tags refactor

# Check status
aura goal status

# Run with dry-run first
aura goal run --dry-run
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Goal stuck in "running" | Check `aura doctor` for system health |
| Goals failing repeatedly | Review logs with `aura logs --tail 100` |
| Queue not processing | Ensure `AURA_JWT_SECRET` is set |
| Out of API credits | Check LLM provider billing |

## Related Commands

- [`sadd`](sadd.md) - Sub-Agent Driven Development for complex tasks
- [`doctor`](doctor.md) - Diagnose system issues
- [`logs`](logs.md) - View execution logs
