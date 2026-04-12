# ADR-009: Pydantic for Configuration Management

**Date:** 2026-04-10  
**Status:** Accepted  
**Deciders:** AURA Core Team  

## Context

AURA CLI requires robust configuration management to handle:

1. Multiple configuration sources (env vars, JSON files, defaults)
2. Complex nested configuration structures
3. Type validation and coercion
4. Environment-specific overrides
5. Secret management and redaction
6. Configuration migration between versions

The main contenders were:
- **dataclasses** — Standard library, lightweight, but no validation
- **Pydantic v1** — Battle-tested, excellent validation
- **Pydantic v2** — Performance improvements, stricter validation
- **attrs + cattrs** — Flexible, but more verbose
- **Custom solution** — Full control but high maintenance

## Decision

We chose **Pydantic v2** for configuration management.

### Rationale

1. **Validation**: Automatic type checking and validation
   ```python
   from pydantic import BaseModel, Field, validator
   
   class MCPServerConfig(BaseModel):
       dev_tools: int = Field(default=8001, ge=1024, le=65535)
       skills: int = Field(default=8002, ge=1024, le=65535)
   ```

2. **Nested Models**: Clean representation of complex structures
   ```python
   class AuraConfig(BaseModel):
       model_routing: ModelRoutingConfig = Field(default_factory=ModelRoutingConfig)
       mcp_servers: MCPServerConfig = Field(default_factory=MCPServerConfig)
       log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = "INFO"
   ```

3. **Environment Integration**: Native support for env vars via `Settings` (optional)

4. **JSON Schema**: Automatic schema generation for documentation

5. **Performance**: Pydantic v2 is significantly faster than v1

6. **Error Messages**: Clear, actionable validation errors

## Configuration Hierarchy

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: Environment Variables (.env file)         │
│  Highest priority, overrides all others             │
├─────────────────────────────────────────────────────┤
│  Layer 2: aura.config.json                          │
│  Project-specific configuration                     │
├─────────────────────────────────────────────────────┤
│  Layer 3: settings.json                             │
│  Model routing and provider config                  │
├─────────────────────────────────────────────────────┤
│  Layer 4: Built-in Defaults                         │
│  Lowest priority, fallback values                   │
└─────────────────────────────────────────────────────┘
```

## Implementation

```python
# core/config_schema.py
from pydantic import BaseModel, Field, field_validator
from typing import Literal

class ModelRoutingConfig(BaseModel):
    """Model selection for different task types."""
    code_generation: str = "anthropic/claude-3.5-sonnet"
    planning: str = "deepseek/deepseek-chat"
    analysis: str = "google/gemini-2.5-flash"
    critique: str = "deepseek/deepseek-r1-0528"

class AuraConfig(BaseModel):
    """Root configuration model."""
    model_name: str = "google/gemini-2.0-flash-exp:free"
    model_routing: ModelRoutingConfig = Field(default_factory=ModelRoutingConfig)
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = "INFO"
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        return v.upper()
```

## Consequences

### Positive

- Type-safe configuration with runtime validation
- Clear error messages for invalid configuration
- IDE support through type hints
- Automatic serialization/deserialization
- Schema generation for documentation
- Migration path for config versions

### Negative

- Dependency on external library (though Pydantic is widely adopted)
- Some complexity in validator definitions
- Need to handle Pydantic v1 to v2 migration (completed in ADR-004)

## Migration History

- **v1.0.0**: Initial Pydantic v1 implementation
- **v1.1.0**: Migrated to Pydantic v2 (see ADR-004)
- All schemas now use Pydantic v2 syntax (`field_validator`, `model_validator`)

## References

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Configuration Schema](https://github.com/asshat1981ar/aura-cli/tree/main/core/config_schema.py)
- [ADR-004: Pydantic v2 Migration](ADR-004-pydantic-v2-migration.md)
