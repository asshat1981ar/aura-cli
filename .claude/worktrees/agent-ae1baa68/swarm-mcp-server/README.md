# swarm-mcp-server

A Model Context Protocol (MCP) server for multi-agent swarm orchestration. Initialize, manage, and coordinate teams of specialized AI agents with configurable topologies, task delegation, inter-agent messaging, and per-agent persistent memory.

## Features

- **5 Topologies**: hierarchical, flat, mesh, ring, star
- **4 Strategies**: specialized (role-based routing), generalist, hybrid, round-robin
- **12 Agent Roles**: coordinator, code_generator, code_reviewer, tester, architect, documenter, debugger, security_auditor, devops, data_analyst, ux_designer, custom
- **Task Decomposition**: Parent/subtask trees with auto-completion detection
- **Inter-Agent Messaging**: Point-to-point and broadcast with full audit log
- **Per-Agent Memory**: Key-value persistent memory slots for context across tasks
- **Auto-Assignment**: Intelligent agent selection based on role, status, and workload

## Installation

```bash
# Clone or copy the project
cd swarm-mcp-server

# Install dependencies
npm install

# Build
npm run build
```

## Usage with Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "swarm": {
      "command": "node",
      "args": ["/absolute/path/to/swarm-mcp-server/dist/index.js"]
    }
  }
}
```

## Usage with Claude Code

```bash
claude mcp add swarm node /absolute/path/to/swarm-mcp-server/dist/index.js
```

## Available Tools

| Tool | Description |
|------|-------------|
| `swarm_init` | Create a new swarm with topology, strategy, and agent limits |
| `swarm_add_agent` | Add a specialized agent to the swarm |
| `swarm_remove_agent` | Remove an agent (with optional task reassignment) |
| `swarm_create_task` | Create and optionally auto-assign a task |
| `swarm_delegate_task` | Reassign a task to a different agent |
| `swarm_complete_task` | Mark a task complete and store its result |
| `swarm_send_message` | Send a message between agents |
| `swarm_status` | Get full swarm status (agents, tasks, messages) |
| `swarm_list` | List all swarm instances |
| `swarm_terminate` | Terminate a swarm and all agents |
| `swarm_agent_memory` | Read/write/delete per-agent memory slots |

## Example Workflow

```
1. swarm_init({ topology: "hierarchical", maxAgents: 8, strategy: "specialized" })
   → Creates swarm with auto-generated Coordinator agent

2. swarm_add_agent({ swarmId, name: "Architect", role: "architect", parentId: coordinatorId })
   → Adds architect under coordinator

3. swarm_add_agent({ swarmId, name: "Coder", role: "code_generator", parentId: coordinatorId })
4. swarm_add_agent({ swarmId, name: "Reviewer", role: "code_reviewer", parentId: coordinatorId })
5. swarm_add_agent({ swarmId, name: "Tester", role: "tester", parentId: coordinatorId })

6. swarm_create_task({ swarmId, title: "Design API schema", requiredRole: "architect" })
   → Auto-assigned to the Architect agent

7. swarm_complete_task({ swarmId, taskId, result: "OpenAPI 3.1 schema for /users endpoint..." })
   → Architect freed, result stored

8. swarm_create_task({ swarmId, title: "Implement /users endpoint", requiredRole: "code_generator" })
   → Auto-assigned to Coder

9. swarm_status({ swarmId })
   → Full dashboard of agents, tasks, messages
```

## Topologies

- **hierarchical**: Tree structure with coordinator at root. Best for structured delegation.
- **flat**: All agents are peers. Good for collaborative work.
- **mesh**: All-to-all connectivity. Maximum communication flexibility.
- **ring**: Each agent connects to next. Good for pipeline workflows.
- **star**: Central hub with spokes. Similar to hierarchical but single level.

## License

MIT
