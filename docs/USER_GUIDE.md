# AURA User Guide

Complete guide to using AURA for automated development workflows.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Goals and Tasks](#goals-and-tasks)
3. [GitHub Integration](#github-integration)
4. [Web UI](#web-ui)
5. [Sub-Agent Driven Development](#sub-agent-driven-development)
6. [Notifications](#notifications)
7. [Performance](#performance)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/asshat1981ar/aura-cli.git
cd aura-cli

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### Configuration

Create a `.env` file:

```bash
# GitHub Integration (optional)
GITHUB_APP_ID=your_app_id
GITHUB_PRIVATE_KEY=path/to/private-key.pem
GITHUB_WEBHOOK_SECRET=your_webhook_secret

# Slack Notifications (optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Discord Notifications (optional)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# API Configuration
AURA_API_HOST=0.0.0.0
AURA_API_PORT=8000
```

### Starting the Server

```bash
# Start the API server
python3 main.py api-server

# Or with the convenience script
./run_aura.sh api-server
```

Access the Web UI at `http://localhost:8000`.

## Goals and Tasks

### Adding a Goal

Using the CLI:
```bash
python3 main.py goal add "Refactor authentication module"
```

Using the Web UI:
1. Navigate to the Dashboard
2. Click "Add Goal"
3. Enter goal description
4. Set priority (1-10)
5. Submit

### Running Goals

```bash
# Run all pending goals
python3 main.py goal run

# Run a single goal
python3 main.py goal once "Specific task description"

# Run with auto-resume
python3 main.py goal run --resume
```

### Goal Priority

Priority levels (1 = highest):
- **1-3**: Critical - Urgent tasks requiring immediate attention
- **4-6**: Normal - Standard development tasks
- **7-9**: Low - Nice-to-have improvements
- **10**: Backlog - Future considerations

### Monitoring Goals

```bash
# Check goal status
python3 main.py goal status

# View goal queue
python3 main.py queue list
```

## GitHub Integration

### Setup

1. Create a GitHub App at Settings > Developer Settings > GitHub Apps
2. Set webhook URL to `https://your-server.com/api/github/webhook`
3. Download the private key
4. Configure environment variables

### Pull Request Reviews

AURA automatically reviews PRs when:
- A PR is opened
- The `/aura review` command is commented

Review checks include:
- Print statements (should use logging)
- TODOs without ticket references
- Debug breakpoints left in code
- Bare except clauses
- Hardcoded secrets
- Long lines (>100 chars)

### Slash Commands

Comment these commands on PRs:

- `/aura review` - Request code review
- `/aura fix` - Request automatic fixes
- `/aura help` - Show available commands

### Automatic Fixes

When you comment `/aura fix`, AURA will:
1. Analyze the PR for fixable issues
2. Generate fixes for safe transformations
3. Commit changes to the PR branch
4. Comment with a summary

Fixable issues:
- ✅ Print → log_json
- ✅ TODO → TODO(TICKET-XXX)
- ✅ breakpoint() → removed
- ✅ bare except → except Exception
- ❌ Long lines → manual review required

## Web UI

### Dashboard

The main dashboard provides:
- Goal queue overview
- Active agents status
- Recent activity feed
- Quick actions

### GitHub PR Dashboard

Access via `/prs` or the PR menu item:

Features:
- List all open/closed PRs
- Filter by state, author, search
- Real-time updates via WebSocket
- View PR details and reviews
- Track review comments

### Chat Interface

The AI Chat interface allows:
- Direct conversation with agents
- Natural language goal creation
- Querying system status
- Getting help and suggestions

### Terminal

The integrated terminal:
- Multi-session support
- Command history
- Security filtering (blocks dangerous commands)
- Real-time output streaming

## Sub-Agent Driven Development

### SADD Overview

SADD (Sub-Agent Driven Development) enables complex tasks to be broken down into parallel workstreams.

### Creating a SADD Session

Using the CLI:
```bash
python3 main.py sadd run --spec design_spec.yaml
```

Using the API:
```bash
curl -X POST http://localhost:8000/api/sadd/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "design_spec": "path/to/spec.yaml",
    "workstreams": ["ws1", "ws2", "ws3"]
  }'
```

### Design Specifications

Create a YAML design spec:

```yaml
session_id: my-feature
workstreams:
  - id: ws1
    title: "Backend API"
    description: "Implement REST endpoints"
    priority: high
    
  - id: ws2
    title: "Frontend UI"
    description: "Create React components"
    priority: high
    depends_on: [ws1]
    
  - id: ws3
    title: "Tests"
    description: "Write integration tests"
    priority: medium
    depends_on: [ws1, ws2]
```

### Monitoring Sessions

```bash
# List active sessions
python3 main.py sadd status

# View session details
python3 main.py sadd status --session my-feature

# Resume interrupted session
python3 main.py sadd resume --session my-feature --run
```

## Notifications

### Slack Setup

1. Create an incoming webhook in Slack
2. Set `SLACK_WEBHOOK_URL` environment variable
3. Configure notification rules

### Discord Setup

1. Create a webhook in Discord server settings
2. Set `DISCORD_WEBHOOK_URL` environment variable
3. Configure bot username and avatar (optional)

### Notification Rules

Configure in `settings.json`:

```json
{
  "notifications": {
    "rules": [
      {
        "event": "pr_merged",
        "channels": ["slack", "discord"]
      },
      {
        "event": "goal_completed",
        "channels": ["slack"]
      },
      {
        "event": "error",
        "channels": ["slack", "discord"],
        "min_priority": "high"
      }
    ]
  }
}
```

### Testing Notifications

```bash
# Test Slack
python3 main.py notify test --channel slack

# Test Discord
python3 main.py notify test --channel discord
```

## Performance

### Cache Management

View cache statistics:
```bash
curl http://localhost:8000/api/performance/stats
```

Clear cache:
```bash
curl -X POST http://localhost:8000/api/performance/cache/clear
```

### Memory Profiling

Enable memory tracing:
```python
from core.memory_profiler import profile_memory

with profile_memory("operation_name"):
    # Your code here
    result = expensive_operation()
```

View memory stats:
```bash
python3 main.py metrics memory
```

### Optimization Tips

1. **Use caching**: Enable result caching for expensive operations
2. **Batch operations**: Process items in batches
3. **Async execution**: Use async patterns for I/O operations
4. **Lazy loading**: Load heavy resources only when needed

## Troubleshooting

### Common Issues

#### Goals not running

Check:
```bash
# Goal queue status
python3 main.py queue list

# Check for in-flight goals
python3 main.py goal status
```

Resume if needed:
```bash
python3 main.py goal run --resume
```

#### GitHub webhook not working

1. Verify webhook URL is accessible
2. Check signature validation
3. Review logs:
   ```bash
   python3 main.py logs --filter github
   ```

#### WebSocket connection issues

1. Check browser console for errors
2. Verify server is running
3. Check firewall settings
4. Test with curl:
   ```bash
   curl -i -N \
     -H "Connection: Upgrade" \
     -H "Upgrade: websocket" \
     -H "Host: localhost:8000" \
     -H "Origin: http://localhost:8000" \
     http://localhost:8000/ws
   ```

### Debug Mode

Enable debug logging:
```bash
export AURA_LOG_LEVEL=DEBUG
python3 main.py goal run
```

### Getting Help

- CLI help: `python3 main.py --help`
- Command help: `python3 main.py goal --help`
- GitHub Issues: https://github.com/asshat1981ar/aura-cli/issues
- Documentation: https://github.com/asshat1981ar/aura-cli/docs

## Advanced Topics

### Custom Agents

Create custom agents by extending the base agent class:

```python
from agents.base import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self):
        super().__init__("my_agent")
    
    async def run(self, context: dict) -> dict:
        # Your logic here
        return {"result": "success"}
```

### MCP Tool Integration

Register custom MCP tools:

```python
from core.mcp_agent_registry import register_tool

@register_tool("my_tool")
def my_tool(param1: str, param2: int) -> dict:
    """Tool description."""
    return {"result": param1 * param2}
```

### Webhook Customization

Create custom webhook handlers:

```python
from aura_cli.github_integration import GitHubApp

class CustomGitHubApp(GitHubApp):
    def _handle_custom_event(self, payload: dict) -> dict:
        # Your custom logic
        return {"status": "processed"}
```

## Best Practices

1. **Goal Granularity**: Keep goals specific and actionable
2. **Priority Management**: Use priority levels effectively
3. **Review Regularly**: Check PR reviews promptly
4. **Monitor Performance**: Keep an eye on cache hit rates
5. **Backup**: Regularly backup memory/ directory
6. **Security**: Rotate webhook secrets periodically

## Glossary

- **AURA**: Autonomous Unified Reasoning Agent
- **SADD**: Sub-Agent Driven Development
- **MCP**: Model Context Protocol
- **Workstream**: Parallel task in SADD
- **Goal**: High-level objective for AURA
- **Agent**: Specialized AI component
- **Orchestrator**: Central coordination system
