# doctor Command

Diagnose system health, verify dependencies, and troubleshoot issues with AURA CLI.

## Overview

The `doctor` command runs a comprehensive health check of your AURA installation, verifying:
- Environment configuration
- Required dependencies
- MCP server status
- LLM provider connectivity
- File permissions
- Network connectivity

## Subcommands

### `doctor` (default)

Run full system diagnostics.

```bash
# Complete system check
aura doctor

# JSON output for automation
aura doctor --json

# Verbose output with recommendations
aura doctor --verbose
```

**Output Example:**

```
┌─────────────────────────────────────────────────────────────┐
│ AURA System Diagnostics                                     │
├─────────────────────────────────────────────────────────────┤
│ Environment: development                                    │
│ Timestamp: 2026-04-10T14:32:01Z                            │
├─────────────────────────────────────────────────────────────┤
│ ✓ Python Version                                            │
│   Version: 3.11.4 (required: >=3.10)                        │
│                                                             │
│ ✓ Dependencies                                              │
│   All required packages installed                           │
│   FastAPI: 0.135.1                                          │
│   Pydantic: 2.12.5                                          │
│   Typer: 0.12.0                                             │
│                                                             │
│ ✓ Configuration                                             │
│   Config file: Found                                        │
│   AURA_JWT_SECRET: Set (56 chars)                           │
│   OPENAI_API_KEY: Set                                       │
│                                                             │
│ ✓ LLM Providers                                             │
│   OpenAI: Connected (latency: 245ms)                        │
│   Anthropic: Not configured                                 │
│                                                             │
│ ✓ MCP Servers                                               │
│   dev_tools: Running on port 8001                           │
│   skills: Running on port 8002                              │
│                                                             │
│ ⚠ Memory/Cache                                              │
│   Redis: Not configured (caching disabled)                  │
│   → Recommendation: Set REDIS_URL for better performance    │
│                                                             │
│ ✓ Git Repository                                            │
│   Git: Installed (2.42.0)                                   │
│   Repository: Initialzed                                    │
│   Working tree: Clean                                       │
└─────────────────────────────────────────────────────────────┘
│ Status: HEALTHY (1 warning)                                 │
└─────────────────────────────────────────────────────────────┘
```

### `doctor --fix`

Attempt to automatically fix detected issues.

```bash
# Try to fix common issues
aura doctor --fix

# Fix with confirmation prompts
aura doctor --fix --interactive

# Fix specific category
aura doctor --fix --category permissions
```

**Auto-fix capabilities:**
- Create missing config files
- Fix file permissions
- Restart stopped MCP servers
- Clear stale lock files

### `doctor check`

Run specific diagnostic checks.

```bash
# Check specific component
aura doctor check config
aura doctor check mcp
aura doctor check llm
aura doctor check git
aura doctor check network
aura doctor check permissions
```

## Diagnostic Categories

### Python Environment

Checks:
- Python version (≥3.10 required)
- Virtual environment status
- Required package availability
- Package version compatibility

**Common Issues:**

| Issue | Cause | Fix |
|-------|-------|-----|
| `Python 3.9 detected` | Old Python version | Upgrade to Python 3.10+ |
| `Package not found` | Missing dependency | `pip install -e ".[dev]"` |
| `Version mismatch` | Outdated package | `pip install --upgrade <package>` |

### Configuration

Checks:
- Config file exists and is valid JSON
- Required environment variables set
- JWT secret length (≥43 characters)
- API key format validity

**Common Issues:**

| Issue | Cause | Fix |
|-------|-------|-----|
| `AURA_JWT_SECRET not set` | Missing env var | `export AURA_JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(43))")` |
| `JWT secret too short` | Weak secret | Generate longer secret |
| `Invalid OpenAI key` | Wrong format | Key should start with `sk-` |

### LLM Providers

Checks:
- API key validity
- Network connectivity
- Rate limit status
- Account quota/billing

**Common Issues:**

| Issue | Cause | Fix |
|-------|-------|-----|
| `Connection timeout` | Network issue | Check internet connection |
| `401 Unauthorized` | Invalid API key | Verify key in `.env` |
| `429 Rate limited` | Too many requests | Wait or upgrade plan |
| `Insufficient quota` | Billing issue | Check account billing |

### MCP Servers

Checks:
- Server processes running
- Port availability
- Health endpoint responses
- Tool discovery

**Common Issues:**

| Issue | Cause | Fix |
|-------|-------|-----|
| `Port 8001 in use` | Another process | `kill $(lsof -t -i:8001)` or change port |
| `Server not responding` | Process crashed | Restart with `aura mcp start` |
| `Tool discovery failed` | Schema error | Check server logs |

### Git Repository

Checks:
- Git installation
- Repository initialization
- Working tree status
- Remote connectivity

### File System

Checks:
- Required directories exist
- Read/write permissions
- Disk space available
- Lock files stale

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | All checks passed |
| `1` | One or more warnings |
| `2` | One or more errors |
| `3` | Critical system failure |

Useful for CI/CD pipelines:

```bash
# Fail CI on doctor errors
aura doctor --json || exit 1
```

## JSON Output Format

```json
{
  "timestamp": "2026-04-10T14:32:01Z",
  "environment": "development",
  "status": "healthy",
  "checks": {
    "python": {
      "status": "pass",
      "version": "3.11.4",
      "required": ">=3.10"
    },
    "dependencies": {
      "status": "pass",
      "packages": {
        "fastapi": "0.135.1",
        "pydantic": "2.12.5"
      }
    },
    "config": {
      "status": "pass",
      "file_found": true,
      "jwt_secret_set": true,
      "api_keys": ["openai"]
    },
    "llm_providers": {
      "status": "pass",
      "providers": {
        "openai": {"connected": true, "latency_ms": 245}
      }
    },
    "mcp_servers": {
      "status": "pass",
      "servers": {
        "dev_tools": {"running": true, "port": 8001},
        "skills": {"running": true, "port": 8002}
      }
    },
    "redis": {
      "status": "warning",
      "configured": false,
      "message": "Caching disabled"
    }
  },
  "recommendations": [
    "Set REDIS_URL for better caching performance"
  ]
}
```

## Health Status Levels

### 🟢 HEALTHY
All checks passed. System ready for use.

### 🟡 DEGRADED
Non-critical issues detected. System functional but may have reduced performance.

**Common causes:**
- Redis not configured (caching disabled)
- Fallback model in use
- Debug mode enabled in production

### 🔴 UNHEALTHY
Critical issues detected. System may not function correctly.

**Common causes:**
- Missing required API keys
- MCP servers not running
- Configuration errors

### ⚫ CRITICAL
System failure. Immediate attention required.

**Common causes:**
- Database corruption
- Permission denied on critical paths
- Network completely unavailable

## Continuous Monitoring

### Watch Mode

```bash
# Continuous health monitoring
watch -n 30 aura doctor

# With notifications
while true; do
  aura doctor --json | jq -e '.status == "healthy"' || notify-send "AURA Unhealthy"
  sleep 60
done
```

### Integration with Monitoring

```bash
# Prometheus exporter format
aura doctor --format prometheus

# Health endpoint for load balancers
aura doctor --format simple
```

## Troubleshooting Workflows

### First-Time Setup Issues

```bash
# Step 1: Check Python
aura doctor check python

# Step 2: Install dependencies
pip install -e ".[dev]"

# Step 3: Configure environment
aura config init
aura doctor check config

# Step 4: Verify LLM connectivity
aura doctor check llm

# Step 5: Start MCP servers
aura mcp start
aura doctor check mcp
```

### Performance Issues

```bash
# Full diagnostic
aura doctor --verbose

# Check specific components
aura doctor check mcp  # Slow responses?
aura doctor check llm  # High latency?

# View resource usage
aura doctor --stats
```

### Network Issues

```bash
# Test connectivity
aura doctor check network

# Check proxy settings
env | grep -i proxy

# Test specific endpoints
curl -v https://api.openai.com/v1/models
```

## Best Practices

1. **Run before first use**: `aura doctor` after installation
2. **Run after updates**: Check for breaking changes
3. **CI/CD integration**: Fail builds on critical issues
4. **Regular monitoring**: Schedule periodic health checks
5. **Document fixes**: Keep notes on resolved issues

## Related Commands

- [`config`](config.md) - View and modify configuration
- [`mcp status`](mcp.md) - Check MCP server status
- [`logs`](logs.md) - View detailed logs
