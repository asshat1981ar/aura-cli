# ADR-012: MCP-Based Plugin System

**Date:** 2026-04-10  
**Status:** Accepted  
**Deciders:** AURA Core Team  

## Context

AURA needed an extensible plugin system that would:

1. Allow third-party extensions without core code changes
2. Provide a standardized interface for tools
3. Support both local and remote tool execution
4. Enable secure, sandboxed tool operations
5. Allow dynamic discovery of available capabilities
6. Integrate with LLM function calling

We evaluated several approaches:
- **Direct Python imports** — Simple but unsafe, tight coupling
- **Webhooks** — Flexible but requires infrastructure
- **gRPC plugins** — High performance but complex
- **MCP (Model Context Protocol)** — Purpose-built for LLM tool integration

## Decision

We adopted the **Model Context Protocol (MCP)** as our plugin architecture.

## What is MCP?

MCP is an open protocol that standardizes how applications provide context and tools to LLMs. Think of it as a "USB-C for AI applications" — a universal interface for connecting AI systems to data sources and tools.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AURA CLI                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              MCP Client (aura_cli/mcp_client.py)        │   │
│  │  • Tool discovery        • Request routing              │   │
│  │  • Schema validation     • Error handling               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              MCP Transport Layer                        │   │
│  │  • HTTP/WebSocket        • stdio (local)                │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │  Dev Tools   │   │   Skills     │   │  External    │
    │   (:8001)    │   │   (:8002)    │   │   Servers    │
    ├──────────────┤   ├──────────────┤   ├──────────────┤
    │ • file_read  │   │ • analyze    │   │ • github     │
    │ • file_write │   │ • lint       │   │ • slack      │
    │ • shell_exec │   │ • test       │   │ • jira       │
    │ • git_ops    │   │ • format     │   │ • custom     │
    └──────────────┘   └──────────────┘   └──────────────┘
```

## MCP Server Types

### 1. Built-in Servers

| Server | Port | Tools | Purpose |
|--------|------|-------|---------|
| `dev_tools` | 8001 | file operations, shell, git | Core development |
| `skills` | 8002 | analyze, lint, test | Code quality |
| `control` | 8003 | orchestration, monitoring | System control |
| `sadd` | 8020 | decomposition, workstreams | SADD workflow |

### 2. Configuration

```json
// .mcp.json
{
  "mcpServers": {
    "dev_tools": {
      "command": "python",
      "args": ["tools/mcp_server.py"],
      "env": {"PORT": "8001"}
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@github/mcp-server"],
      "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}
    }
  }
}
```

## Tool Interface

```python
# Standard MCP tool definition
@mcp.tool()
def read_file(path: str, offset: int = 0, limit: int = None) -> str:
    """Read contents of a file.
    
    Args:
        path: Absolute path to the file
        offset: Line number to start reading from
        limit: Maximum number of lines to read
        
    Returns:
        File contents as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file cannot be read
    """
    ...
```

## Benefits of MCP

### 1. Standardization

All tools use a consistent interface:
```python
# Call any MCP tool uniformly
result = await mcp_client.call(
    server="dev_tools",
    tool="read_file",
    args={"path": "/tmp/test.py"}
)
```

### 2. Discoverability

```python
# List all available tools
tools = await mcp_client.list_tools()
for tool in tools:
    print(f"{tool.server}/{tool.name}: {tool.description}")
```

### 3. Type Safety

Tools declare their schemas:
```json
{
  "name": "read_file",
  "description": "Read file contents",
  "inputSchema": {
    "type": "object",
    "properties": {
      "path": {"type": "string"},
      "offset": {"type": "integer", "default": 0}
    },
    "required": ["path"]
  }
}
```

### 4. Security

- Sandboxed execution for untrusted tools
- Fine-grained permissions per server
- Audit logging of all tool calls

## Integration with Agents

Agents resolve capabilities to MCP tools:

```python
# Agent uses MCP tools transparently
class CodeSearchAgent(Agent):
    capabilities = ["code_search", "semantic_query"]
    
    async def run(self, input_data: dict) -> dict:
        # Automatically routes to appropriate MCP tool
        results = await self.mcp.call(
            tool="semantic_search",
            args={"query": input_data["goal"]}
        )
        return {"matches": results}
```

## Consequences

### Positive

- Clean separation between core and extensions
- Easy to add new capabilities without code changes
- Standard interface enables ecosystem growth
- Remote tool execution support
- Strong typing and validation
- Built-in observability

### Negative

- Additional infrastructure (MCP servers to manage)
- Network overhead for remote tools
- Learning curve for plugin developers
- Version compatibility between client and servers

## Best Practices

1. **Version MCP servers** independently of core
2. **Use local sockets** for local tools to reduce latency
3. **Implement health checks** for all servers
4. **Cache tool schemas** to reduce discovery overhead
5. **Log all tool calls** for debugging and audit

## References

- [MCP Specification](https://modelcontextprotocol.io/)
- [MCP Servers](https://github.com/asshat1981ar/aura-cli/tree/main/tools)
- [MCP Client](https://github.com/asshat1981ar/aura-cli/tree/main/aura_cli/mcp_client.py)
