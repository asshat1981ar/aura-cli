#!/usr/bin/env node

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

import {
  SwarmInitSchema, AddAgentSchema, RemoveAgentSchema,
  CreateTaskSchema, DelegateTaskSchema, CompleteTaskSchema,
  SendMessageSchema, SwarmStatusSchema, SwarmListSchema,
  SwarmTerminateSchema, AgentMemorySchema,
} from "./schemas/index.js";

import {
  swarmInit, addAgent, removeAgent,
  createTask, delegateTask, completeTask,
  sendMessage, swarmStatus, swarmList,
  swarmTerminate, agentMemory,
} from "./services/swarm-engine.js";

// ─── Server ────────────────────────────────────────────────────
const server = new McpServer({
  name: "swarm-mcp-server",
  version: "1.0.0",
});

// ─── Helper: format tool result ────────────────────────────────
function ok(data: Record<string, unknown>) {
  return {
    content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }],
    structuredContent: data,
  };
}

function fail(error: unknown) {
  const msg = error instanceof Error ? error.message : String(error);
  return {
    isError: true,
    content: [{ type: "text" as const, text: `Error: ${msg}` }],
  };
}

// ═══════════════════════════════════════════════════════════════
// TOOL REGISTRATIONS
// ═══════════════════════════════════════════════════════════════

server.registerTool("swarm_init", {
  title: "Initialize Swarm",
  description: "Create and initialize a new multi-agent swarm instance.",
  inputSchema: SwarmInitSchema,
  annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
}, async (params) => {
  try {
    return ok(swarmInit({
      name: params.name ?? "default-swarm",
      topology: params.topology ?? "hierarchical",
      maxAgents: params.maxAgents ?? 8,
      strategy: params.strategy ?? "specialized",
    }));
  } catch (e) { return fail(e); }
});

server.registerTool("swarm_add_agent", {
  title: "Add Agent to Swarm",
  description: "Add a new specialized agent to an existing swarm.",
  inputSchema: AddAgentSchema,
  annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
}, async (params) => {
  try {
    return ok(addAgent(params.swarmId, params.name, params.role, params.parentId, params.capabilities));
  } catch (e) { return fail(e); }
});

server.registerTool("swarm_remove_agent", {
  title: "Remove Agent from Swarm",
  description: "Remove an agent from the swarm. Optionally reassigns its tasks.",
  inputSchema: RemoveAgentSchema,
  annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: false, openWorldHint: false },
}, async (params) => {
  try {
    return ok(removeAgent(params.swarmId, params.agentId, params.reassignTasks));
  } catch (e) { return fail(e); }
});

server.registerTool("swarm_create_task", {
  title: "Create Task",
  description: "Create a new task in the swarm's task queue.",
  inputSchema: CreateTaskSchema,
  annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
}, async (params) => {
  try {
    return ok(createTask(
      params.swarmId, params.title, params.description,
      params.priority, params.requiredRole, params.parentTaskId, params.assignToAgentId,
    ));
  } catch (e) { return fail(e); }
});

server.registerTool("swarm_delegate_task", {
  title: "Delegate Task",
  description: "Reassign an existing task to a different agent.",
  inputSchema: DelegateTaskSchema,
  annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
}, async (params) => {
  try {
    return ok(delegateTask(params.swarmId, params.taskId, params.agentId));
  } catch (e) { return fail(e); }
});

server.registerTool("swarm_complete_task", {
  title: "Complete Task",
  description: "Mark a task as completed and store its result.",
  inputSchema: CompleteTaskSchema,
  annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
}, async (params) => {
  try {
    return ok(completeTask(params.swarmId, params.taskId, params.result));
  } catch (e) { return fail(e); }
});

server.registerTool("swarm_send_message", {
  title: "Send Inter-Agent Message",
  description: "Send a message between agents in the swarm.",
  inputSchema: SendMessageSchema,
  annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
}, async (params) => {
  try {
    return ok(sendMessage(
      params.swarmId, params.fromAgentId, params.toAgentId,
      params.type, params.payload,
    ));
  } catch (e) { return fail(e); }
});

server.registerTool("swarm_status", {
  title: "Get Swarm Status",
  description: "Get a comprehensive status report for a swarm.",
  inputSchema: SwarmStatusSchema,
  annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
}, async (params) => {
  try {
    return ok(swarmStatus(params.swarmId));
  } catch (e) { return fail(e); }
});

server.registerTool("swarm_list", {
  title: "List All Swarms",
  description: "List all swarm instances (active and terminated).",
  inputSchema: SwarmListSchema,
  annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
}, async () => {
  try {
    return ok(swarmList());
  } catch (e) { return fail(e); }
});

server.registerTool("swarm_terminate", {
  title: "Terminate Swarm",
  description: "Terminate a swarm and all its agents.",
  inputSchema: SwarmTerminateSchema,
  annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
}, async (params) => {
  try {
    return ok(swarmTerminate(params.swarmId));
  } catch (e) { return fail(e); }
});

server.registerTool("swarm_agent_memory", {
  title: "Agent Memory Operations",
  description: "Read, write, delete, or list persistent memory slots for an agent.",
  inputSchema: AgentMemorySchema,
  annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
}, async (params) => {
  try {
    return ok(agentMemory(
      params.swarmId, params.agentId, params.action,
      params.key, params.value,
    ));
  } catch (e) { return fail(e); }
});

// ═══════════════════════════════════════════════════════════════
// STARTUP
// ═══════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("swarm-mcp-server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
