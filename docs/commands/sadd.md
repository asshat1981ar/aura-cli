# sadd Command

**Sub-Agent Driven Development (SADD)** is AURA's advanced workflow for decomposing complex development tasks into parallel workstreams executed by specialized sub-agents.

## Overview

When a goal is too complex for single-agent processing, SADD:
1. **Decomposes** the goal into independent sub-tasks
2. **Dispatches** sub-agents for parallel execution
3. **Coordinates** workstreams with dependency management
4. **Synthesizes** results into cohesive changes
5. **Validates** the integrated solution

```
┌─────────────────────────────────────────────────────────────────┐
│                    SADD Workflow                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   Complex    │─────▶│Decomposition │─────▶│ Workstream   │  │
│  │    Goal      │      │   Engine     │      │ Scheduler    │  │
│  └──────────────┘      └──────────────┘      └──────┬───────┘  │
│                                                     │           │
│                           ┌─────────────────────────┼────────┐  │
│                           ▼                         ▼        │  │
│                     ┌──────────┐              ┌──────────┐   │  │
│                     │ Agent A  │              │ Agent B  │   │  │
│                     │(Backend) │              │(Frontend)│   │  │
│                     └────┬─────┘              └────┬─────┘   │  │
│                          │                         │          │  │
│                          └───────────┬─────────────┘          │  │
│                                      ▼                        │  │
│                              ┌──────────────┐                 │  │
│                              │ Synthesizer  │                 │  │
│                              └──────┬───────┘                 │  │
│                                     ▼                         │  │
│                              ┌──────────────┐                 │  │
│                              │   Unified    │                 │  │
│                              │   Output     │                 │  │
│                              └──────────────┘                 │  │
└─────────────────────────────────────────────────────────────────┘
```

## Subcommands

### `sadd run`

Execute a SADD workflow from a specification file.

```bash
# Run with spec file
aura sadd run --spec ./design.md

# Run inline goal
aura sadd run --goal "Implement user authentication system"

# Run with custom max depth
aura sadd run --spec design.md --max-depth 3
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--spec` | Path to specification file | Required* |
| `--goal` | Inline goal description | Required* |
| `--max-depth` | Maximum decomposition depth | `3` |
| `--parallel` | Max parallel sub-agents | `5` |
| `--timeout` | Timeout per workstream (seconds) | `600` |
| `--output` | Output directory for artifacts | `./sadd-output` |

*One of `--spec` or `--goal` is required.

### `sadd decompose`

Analyze and decompose a goal without executing.

```bash
# Decompose and show structure
aura sadd decompose --goal "Build e-commerce API" --show-tree

# Export decomposition to JSON
aura sadd decompose --spec design.md --output decomposition.json
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--goal` | Goal to decompose | Required |
| `--show-tree` | Display workstream tree | `false` |
| `--output` | Export to file | None |
| `--format` | Output format (json/yaml) | `json` |

### `sadd status`

Check status of running SADD workflows.

```bash
# Show all workflows
aura sadd status

# Show specific workflow
aura sadd status --id sadd-2024-001

# JSON output
aura sadd status --json
```

### `sadd fleet`

Manage the sub-agent fleet (advanced).

```bash
# List active agents
aura sadd fleet list

# Scale fleet size
aura sadd fleet scale --size 10

# Check fleet health
aura sadd fleet health
```

## Specification Format

SADD specifications use Markdown with YAML frontmatter:

```markdown
---
title: User Authentication System
description: Implement complete auth flow
type: feature
priority: high
domain: backend
---

# Overview

Implement a user authentication system with:
- JWT token generation
- Password hashing with bcrypt
- Login/logout endpoints
- Token refresh mechanism

## Constraints

- Use existing User model
- Follow REST conventions
- Include comprehensive tests

## Sub-Tasks

### 1. Database Schema
- Add password_hash field to users table
- Create refresh_tokens table

### 2. Authentication Service
- Implement password verification
- Implement token generation
- Implement token validation

### 3. API Endpoints
- POST /api/auth/login
- POST /api/auth/logout
- POST /api/auth/refresh

### 4. Middleware
- Add JWT verification middleware
- Add authentication guards

## Dependencies

```yaml
graph:
  - task: "Database Schema"
    before: ["Authentication Service"]
  
  - task: "Authentication Service"
    before: ["API Endpoints", "Middleware"]
```

## Acceptance Criteria

- [ ] All endpoints return correct status codes
- [ ] Tokens expire after configured time
- [ ] Refresh tokens are single-use
- [ ] 100% test coverage for auth logic
```

## Workstream Types

SADD supports different workstream types based on task nature:

| Type | Description | Use Case |
|------|-------------|----------|
| `code` | Code generation and modification | Feature implementation |
| `test` | Test writing and validation | Test coverage |
| `docs` | Documentation generation | API docs, README |
| `review` | Code review and analysis | PR review, audits |
| `research` | Investigation and analysis | Tech research |
| `refactor` | Code restructuring | Legacy modernization |

## Parallel Execution

Control parallelism with the `--parallel` flag:

```bash
# Conservative (safe for resource-intensive tasks)
aura sadd run --spec design.md --parallel 2

# Aggressive (for many independent tasks)
aura sadd run --spec design.md --parallel 10

# Dynamic (auto-scale based on system load)
aura sadd run --spec design.md --parallel auto
```

## Dependency Management

Define task dependencies to ensure correct execution order:

```yaml
# Linear chain
tasks:
  - name: "Database migration"
  - name: "Model updates"
    depends_on: ["Database migration"]
  - name: "API endpoints"
    depends_on: ["Model updates"]

# Parallel branches
tasks:
  - name: "Design API"
  - name: "Implement backend"
    depends_on: ["Design API"]
  - name: "Implement frontend"
    depends_on: ["Design API"]
  - name: "Integration tests"
    depends_on: ["Implement backend", "Implement frontend"]
```

## Output Structure

SADD produces organized output:

```
sadd-output/
├── index.json              # Workflow summary
├── workstreams/
│   ├── 01-database-schema/
│   │   ├── changes.json    # File modifications
│   │   ├── log.txt         # Execution log
│   │   └── artifacts/      # Generated files
│   ├── 02-auth-service/
│   └── ...
├── synthesis/
│   ├── merged-changes.json # Combined changes
│   └── conflicts.md        # Any merge conflicts
└── report.md               # Final report
```

## Examples

### Example 1: Feature Implementation

```bash
# Create spec file
cat > feature.md << 'EOF'
---
title: Add Email Notifications
type: feature
---

# Email Notification System

Implement email notifications for:
- User registration welcome email
- Password reset emails
- Weekly digest emails

## Sub-Tasks

1. Email service abstraction
2. Template system
3. Queue integration
4. SendGrid/SMTP provider
EOF

# Run SADD
aura sadd run --spec feature.md --parallel 3
```

### Example 2: Large Refactoring

```bash
# Decompose legacy migration
aura sadd decompose \
  --goal "Migrate from REST to GraphQL" \
  --show-tree \
  --max-depth 4

# Run with conservative parallelism
aura sadd run \
  --goal "Migrate from REST to GraphQL" \
  --parallel 2 \
  --timeout 1800
```

### Example 3: Code Review

```bash
# Review PR with multiple agents
aura sadd run \
  --goal "Review PR #123 for security and best practices" \
  --spec review-config.yaml \
  --output ./review-results
```

## Best Practices

### 1. Task Granularity

- **Too small**: Overhead outweighs benefit
- **Too large**: Defeats parallelization purpose
- **Sweet spot**: 15-60 minutes per workstream

### 2. Dependency Minimization

- Design tasks to be independent where possible
- Use interfaces/contracts between components
- Consider mock implementations for parallel development

### 3. Specification Quality

- Include clear acceptance criteria
- Document constraints and assumptions
- Provide examples where helpful

### 4. Monitoring

```bash
# Watch progress in real-time
watch -n 5 aura sadd status

# Tail logs
aura logs --follow --filter sadd
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Deadlock | Circular dependencies | Check dependency graph |
| Slow execution | Too many dependencies | Increase `--parallel` |
| Memory issues | Too many agents | Reduce `--parallel` |
| Merge conflicts | Overlapping changes | Use smaller workstreams |

## Advanced Configuration

### Fleet Dispatcher Settings

```json
{
  "sadd": {
    "fleet": {
      "min_agents": 2,
      "max_agents": 20,
      "scale_up_threshold": 0.8,
      "scale_down_threshold": 0.3,
      "health_check_interval": 30
    },
    "scheduler": {
      "algorithm": "dependency_aware",
      "retry_failed": true,
      "max_retries": 3
    }
  }
}
```

## Related Commands

- [`goal`](goal.md) - Standard goal processing
- [`agent`](agent.md) - Manage individual agents
- [`doctor`](doctor.md) - Diagnose system issues
