# config Command

Manage AURA CLI configuration settings, view current configuration, and validate setup.

## Overview

AURA uses a hierarchical configuration system:

```
Priority (high → low):
1. Environment variables / .env file
2. aura.config.json (or AURA_CONFIG_PATH)
3. settings.json
4. Built-in defaults
```

## Subcommands

### `config list`

Display all configuration values.

```bash
# Show all config
aura config list

# Show specific section
aura config list --section pipeline

# JSON output for scripting
aura config list --json

# Show only user-modified values
aura config list --modified
```

**Output Example:**

```
┌─────────────────────────────────────────────────────────────┐
│ AURA Configuration                                          │
├─────────────────────────────────────────────────────────────┤
│ Environment: development                                    │
│ Config File: /home/user/project/aura.config.json           │
├─────────────────────────────────────────────────────────────┤
│ Core Settings                                               │
│   AURA_ENV              = development                       │
│   AURA_LOG_LEVEL        = info                              │
│   AURA_API_HOST         = 0.0.0.0                           │
│   AURA_API_PORT         = 8001                              │
├─────────────────────────────────────────────────────────────┤
│ Pipeline Settings                                           │
│   pipeline.max_cycles   = 5                                 │
│   pipeline.enable_critique = true                           │
│   pipeline.enable_sandbox = true                            │
├─────────────────────────────────────────────────────────────┤
│ Model Settings                                              │
│   default_model         = gpt-4                             │
│   fallback_model        = gpt-3.5-turbo                     │
└─────────────────────────────────────────────────────────────┘
```

### `config get`

Get a specific configuration value.

```bash
# Get single value
aura config get AURA_LOG_LEVEL
# Output: info

# Get nested value
aura config get pipeline.max_cycles
# Output: 5

# With default if not set
aura config get CUSTOM_VAR --default "fallback"
```

### `config set`

Set a configuration value.

```bash
# Set environment variable (recommended for secrets)
export OPENAI_API_KEY="sk-..."

# Update aura.config.json
aura config set pipeline.max_cycles 10

# Set nested value
aura config set models.gpt4.temperature 0.7

# Set with type
aura config set pipeline.enable_critique false --type bool
```

**Warning:** Use environment variables for sensitive data (API keys, secrets). Never commit secrets to `aura.config.json`.

### `config validate`

Validate configuration files and environment.

```bash
# Validate current config
aura config validate

# Validate specific file
aura config validate --file ./custom-config.json

# Strict validation (fail on warnings)
aura config validate --strict
```

**Output Example:**

```
✓ Configuration file syntax valid
✓ Required environment variables set
✓ API keys have correct format
✓ Model configuration valid
✓ Pipeline settings in valid range
⚠ Warning: REDIS_URL not set (caching disabled)
✓ JWT secret meets minimum length
```

### `config init`

Initialize configuration in current directory.

```bash
# Interactive setup
aura config init

# Quick init with defaults
aura config init --quick

# Init with specific template
aura config init --template production
```

### `config edit`

Open configuration file in default editor.

```bash
# Edit main config
aura config edit

# Edit specific file
aura config edit --file settings.json
```

## Configuration Files

### `.env`

Environment-specific secrets and overrides:

```bash
# Required
AURA_JWT_SECRET=your-secret-key-min-43-chars-long
OPENAI_API_KEY=sk-...
# Or: ANTHROPIC_API_KEY=sk-ant-...

# Optional
AURA_ENV=development
AURA_LOG_LEVEL=debug
REDIS_URL=redis://localhost:6379
AURA_API_PORT=8001
```

### `aura.config.json`

Main configuration file:

```json
{
  "environment": "development",
  "logging": {
    "level": "info",
    "format": "json"
  },
  "pipeline": {
    "max_cycles": 5,
    "enable_critique": true,
    "enable_sandbox": true,
    "sandbox_timeout": 30,
    "verification_required": true
  },
  "models": {
    "default": "gpt-4",
    "fallback": "gpt-3.5-turbo",
    "routing": {
      "planning": "gpt-4",
      "coding": "gpt-4",
      "review": "gpt-3.5-turbo"
    }
  },
  "memory": {
    "tiers": {
      "working": {"max_items": 100},
      "short_term": {"max_items": 1000},
      "long_term": {"max_items": 10000}
    }
  },
  "mcp": {
    "servers": {
      "dev_tools": {
        "enabled": true,
        "port": 8001
      },
      "skills": {
        "enabled": true,
        "port": 8002
      }
    }
  }
}
```

### `settings.json`

Model and provider-specific settings:

```json
{
  "providers": {
    "openai": {
      "api_key": "${OPENAI_API_KEY}",
      "organization": null,
      "base_url": null
    },
    "anthropic": {
      "api_key": "${ANTHROPIC_API_KEY}",
      "base_url": null
    }
  },
  "models": {
    "gpt-4": {
      "provider": "openai",
      "max_tokens": 8192,
      "temperature": 0.1,
      "top_p": 1.0
    },
    "claude-3-opus": {
      "provider": "anthropic",
      "max_tokens": 4096,
      "temperature": 0.1
    }
  },
  "routing": {
    "default": "gpt-4",
    "by_capability": {
      "planning": "gpt-4",
      "code_generation": "gpt-4",
      "review": "claude-3-opus"
    }
  }
}
```

### `.mcp.json`

MCP server registry:

```json
{
  "mcpServers": {
    "dev_tools": {
      "command": "python",
      "args": ["tools/mcp_server.py"],
      "env": {
        "PORT": "8001"
      }
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@github/mcp-server"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

## Configuration Schema

### Core Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `AURA_ENV` | string | `development` | Runtime environment |
| `AURA_LOG_LEVEL` | string | `info` | Log verbosity |
| `AURA_API_HOST` | string | `0.0.0.0` | Server bind address |
| `AURA_API_PORT` | int | `8001` | Server bind port |
| `AURA_JWT_SECRET` | string | — | JWT signing key (required) |
| `AURA_SKIP_CHDIR` | bool | `false` | Skip directory changes |

### Pipeline Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `pipeline.max_cycles` | int | `5` | Max pipeline iterations |
| `pipeline.enable_critique` | bool | `true` | Enable critique phase |
| `pipeline.enable_sandbox` | bool | `true` | Enable sandbox execution |
| `pipeline.sandbox_timeout` | int | `30` | Sandbox timeout (seconds) |
| `pipeline.verification_required` | bool | `true` | Require verification |

### Model Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `models.default` | string | `gpt-4` | Default LLM model |
| `models.fallback` | string | `gpt-3.5-turbo` | Fallback model |
| `models.routing.*` | string | — | Per-capability routing |

### Memory Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `memory.tiers.*.max_items` | int | varies | Tier capacity limits |

### Security Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `security.rate_limit.enabled` | bool | `true` | Enable rate limiting |
| `security.rate_limit.requests` | int | `100` | Requests per window |
| `security.rate_limit.window` | int | `3600` | Window in seconds |

## Environment Variables Reference

| Variable | Required | Secret | Description |
|----------|----------|--------|-------------|
| `AURA_JWT_SECRET` | ✓ | ✓ | JWT signing key |
| `OPENAI_API_KEY` | ✓* | ✓ | OpenAI API key |
| `ANTHROPIC_API_KEY` | ✓* | ✓ | Anthropic API key |
| `REDIS_URL` | — | — | Redis connection |
| `GITHUB_TOKEN` | — | ✓ | GitHub integration |
| `AURA_ENV` | — | — | Environment name |
| `AURA_LOG_LEVEL` | — | — | Log level |

*At least one LLM provider key required.

## Configuration Templates

### Development Template

```bash
aura config init --template development
```

Optimizes for:
- Debug logging
- Fast iteration
- Local MCP servers
- Disabled rate limiting

### Production Template

```bash
aura config init --template production
```

Optimizes for:
- Info logging
- Full security
- Redis caching
- Rate limiting enabled

### CI/CD Template

```bash
aura config init --template ci
```

Optimizes for:
- Minimal logging
- Fast execution
- No interactive prompts
- Strict validation

## Best Practices

### 1. Security

```bash
# Good: Use environment variables for secrets
export OPENAI_API_KEY="sk-..."

# Bad: Never commit secrets
echo '{"api_key": "sk-..."}' > aura.config.json
```

### 2. Version Control

```bash
# Add to .gitignore
echo ".env" >> .gitignore
echo "aura_auth.db" >> .gitignore
echo "*.local.json" >> .gitignore

# Commit safe templates
git add aura.config.json.template
```

### 3. Environment-Specific Configs

```bash
# Development
AURA_ENV=development AURA_LOG_LEVEL=debug aura goal run

# Staging
AURA_ENV=staging aura goal run

# Production
AURA_ENV=production aura goal run
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Config file not found` | Run `aura config init` |
| `JWT secret too short` | Generate: `python3 -c "import secrets; print(secrets.token_urlsafe(43))"` |
| `Invalid API key format` | Check key prefix (OpenAI: `sk-...`) |
| `Missing required variable` | Run `aura config validate` |
| `Config merge conflict` | Check both `aura.config.json` and `.env` |

## Related Commands

- [`doctor`](doctor.md) - System health check
- [`env`](env.md) - Environment management
