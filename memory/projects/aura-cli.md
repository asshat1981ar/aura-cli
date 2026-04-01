# AURA CLI Project

## Overview
Autonomous software development platform with Sub-Agent Driven Development (SADD)

## Stats
- **Language**: Python
- **Files**: 414 Python files
- **LoC**: 76K+
- **Test Coverage**: 40% (target: 70%)

## Architecture
```
agents/          - Agent implementations
cli/             - Command line interface
core/            - Core orchestration
skills/          - Modular capabilities
plans/           - SADD specifications
n8n-workflows/   - Automation workflows
```

## Key Components

### SADD System
- SQLite persistence for goals
- 8 parallel workstreams
- Wave-based dependency ordering
- 10-phase adaptive pipeline

### Agent Swarm
- Feature flag: `AURA_ENABLE_SWARM=1`
- 20+ registered agent types
- Multi-agent coordination
- Capability-based routing

### n8n Integration
- 7 fleet dispatcher workflows (WF-0 to WF-6)
- Webhook triggers
- MCP proxy for Claude Code

### MCP Connections
- GitHub (connected)
- Slack (pending auth)
- Sentry (pending auth)
- Supabase (pending auth)

## Active Workstreams
1. Exception Handling Hardening
2. Remove Hardcoded Secrets
3. Agent Swarm Orchestration
4. File Filtering Deduplication
5. Implement Streaming for Large Files
6. Test Coverage Enforcement
7. Fix Skipped Tests
8. Swarm Integration with SADD n8n

## Ports
- Web server: 8001
- MCP skills: 8002
- n8n: 5678

## Current Session
- **ID**: b00f7213-c107-4735-a16b-33498f0f3e1c
- **Status**: Active (8 workstreams)
- **Goals in Queue**: 206+
