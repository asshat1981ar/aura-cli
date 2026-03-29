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

server.registerTool(
  "swarm_init",
  {
    title: "Initialize Swarm",
    description: `Create and initialize a new multi-agent swarm instance.

Configures the agent topology, capacity, and task assignment strategy.
In hierarchical mode a coordinator agent is auto-created.

Args:
  - name (string): Human-readable swarm name (default: "default-swarm")
  - topology ("hierarchical"|"flat"|"mesh"|"ring"|"star"): Network topology (default: "hierarchical")
  - maxAgents (number 1-32): Maximum agents allowed (default: 8)
  - strategy ("specialized"|"generalist"|"hybrid"|"round_robin"): Assignment strategy (default: "specialized")

Returns:
  { swarmId, status, topology, strategy, maxAgents, agentCount, agents[], message }`,
    inputSchema: SwarmInitSchema,
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
  },
  async (params) => {
    try {
      return ok(swarmInit({
        name: params.name ?? "default-swarm",
        topology: params.topology ?? "hierarchical",
        maxAgents: params.maxAgents ?? 8,
        strategy: params.strategy ?? "specialized",
      }));
    } catch (e) { return fail(e); }
  },
);

server.registerTool(
  "swarm_add_agent",
  {
    title: "Add Agent to Swarm",
    description: `Add a new specialized agent to an existing swarm.

Each agent has a role, optional parent (for hierarchical topologies), and capability tags.
Fails if the swarm has reached its maxAgents limit.

Args:
  - swarmId (string): Target swarm ID
  - name (string): Agent display name
  - role: One of coordinator, code_generator, code_reviewer, tester, architect, documenter, debugger, security_auditor, devops, data_analyst, ux_designer, custom
  - parentId (string, optional): Parent agent ID for tree topologies
  - capabilities (string[]): Capability tags

Returns:
  { agentId, name, role, parentId, capabilities, swarmAgentCount, maxAgents }`,
    inputSchema: AddAgentSchema,
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
  },
  async (params) => {
    try {
      return ok(addAgent(params.swarmId, params.name, params.role, params.parentId, params.capabilities));
    } catch (e) { return fail(e); }
  },
);

server.registerTool(
  "swarm_remove_agent",
  {
    title: "Remove Agent from Swarm",
    description: `Remove an agent from the swarm. Optionally reassigns its tasks.

Children are reparented to the removed agent's parent. Current tasks can be
automatically reassigned to the next best available agent.

Args:
  - swarmId (string): Target swarm ID
  - agentId (string): Agent to remove
  - reassignTasks (boolean): Reassign in-progress tasks (default: true)

Returns:
  { removedAgentId, reassignedTaskIds[], remainingAgents }`,
    inputSchema: RemoveAgentSchema,
    annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: false, openWorldHint: false },
  },
  async (params) => {
    try {
      return ok(removeAgent(params.swarmId, params.agentId, params.reassignTasks));
    } catch (e) { return fail(e); }
  },
);

server.registerTool(
  "swarm_create_task",
  {
    title: "Create Task",
    description: `Create a new task in the swarm's task queue.

Tasks can be auto-assigned based on the swarm strategy, directly assigned to
a specific agent, or restricted to a required role. Supports subtask decomposition
via parentTaskId.

Args:
  - swarmId (string): Target swarm ID
  - title (string): Task title
  - description (string): Detailed requirements
  - priority ("critical"|"high"|"normal"|"low"): Priority (default: "normal")
  - requiredRole (string, optional): Restrict to agents with this role
  - parentTaskId (string, optional): Parent task for decomposition
  - assignToAgentId (string, optional): Direct assignment

Returns:
  { taskId, title, priority, status, assignedAgentId, parentTaskId }`,
    inputSchema: CreateTaskSchema,
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
  },
  async (params) => {
    try {
      return ok(createTask(
        params.swarmId, params.title, params.description,
        params.priority, params.requiredRole, params.parentTaskId, params.assignToAgentId,
      ));
    } catch (e) { return fail(e); }
  },
);

server.registerTool(
  "swarm_delegate_task",
  {
    title: "Delegate Task",
    description: `Reassign an existing task to a different agent.

Unassigns the previous agent (sets it idle) and assigns the task to the
specified agent. Logs a task_assignment message in the swarm message log.

Args:
  - swarmId (string): Target swarm ID
  - taskId (string): Task to delegate
  - agentId (string): Agent to assign to

Returns:
  { taskId, assignedAgentId, agentName, status }`,
    inputSchema: DelegateTaskSchema,
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  },
  async (params) => {
    try {
      return ok(delegateTask(params.swarmId, params.taskId, params.agentId));
    } catch (e) { return fail(e); }
  },
);

server.registerTool(
  "swarm_complete_task",
  {
    title: "Complete Task",
    description: `Mark a task as completed and store its result.

Frees the assigned agent (sets to idle, increments completedTasks).
If the task is a subtask, checks whether all sibling subtasks are also complete.

Args:
  - swarmId (string): Target swarm ID
  - taskId (string): Task to complete
  - result (string): Task output/result

Returns:
  { taskId, status, result, parentTaskAllSubtasksComplete }`,
    inputSchema: CompleteTaskSchema,
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  },
  async (params) => {
    try {
      return ok(completeTask(params.swarmId, params.taskId, params.result));
    } catch (e) { return fail(e); }
  },
);

server.registerTool(
  "swarm_send_message",
  {
    title: "Send Inter-Agent Message",
    description: `Send a message between agents in the swarm.

Supports point-to-point and broadcast messaging. Messages are logged
in the swarm's message history for audit and replay.

Args:
  - swarmId (string): Target swarm ID
  - fromAgentId (string): Sender agent ID
  - toAgentId (string): Recipient agent ID or "broadcast"
  - type: "task_assignment"|"result"|"status_update"|"query"|"directive"
  - payload (any): Message content

Returns:
  { messageId, from, to, type, timestamp }`,
    inputSchema: SendMessageSchema,
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
  },
  async (params) => {
    try {
      return ok(sendMessage(
        params.swarmId, params.fromAgentId, params.toAgentId,
        params.type, params.payload,
      ));
    } catch (e) { return fail(e); }
  },
);

server.registerTool(
  "swarm_status",
  {
    title: "Get Swarm Status",
    description: `Get a comprehensive status report for a swarm.

Returns all agents, all tasks, status breakdowns, and the 10 most recent messages.

Args:
  - swarmId (string): Target swarm ID

Returns:
  { swarmId, name, status, topology, strategy, maxAgents,
    agentCount, taskCount, messageCount, agentsByStatus, tasksByStatus,
    agents[], tasks[], recentMessages[] }`,
    inputSchema: SwarmStatusSchema,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  },
  async (params) => {
    try {
      return ok(swarmStatus(params.swarmId));
    } catch (e) { return fail(e); }
  },
);

server.registerTool(
  "swarm_list",
  {
    title: "List All Swarms",
    description: `List all swarm instances (active and terminated).

Returns summary info for each swarm including agent/task counts.

Returns:
  { swarms[{ swarmId, name, status, topology, strategy, agentCount, taskCount, createdAt }], total }`,
    inputSchema: SwarmListSchema,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  },
  async () => {
    try {
      return ok(swarmList());
    } catch (e) { return fail(e); }
  },
);

server.registerTool(
  "swarm_terminate",
  {
    title: "Terminate Swarm",
    description: `Terminate a swarm and all its agents.

All agents are set to terminated status. In-progress tasks remain in their current state.

Args:
  - swarmId (string): Target swarm ID

Returns:
  { swarmId, status, agentsTerminated, tasksInProgress }`,
    inputSchema: SwarmTerminateSchema,
    annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
  },
  async (params) => {
    try {
      return ok(swarmTerminate(params.swarmId));
    } catch (e) { return fail(e); }
  },
);

server.registerTool(
  "swarm_agent_memory",
  {
    title: "Agent Memory Operations",
    description: `Read, write, delete, or list persistent memory slots for an agent.

Each agent has a key-value memory store that persists across tasks within a session.

Args:
  - swarmId (string): Target swarm ID
  - agentId (string): Target agent ID
  - action ("get"|"set"|"delete"|"list"): Operation to perform
  - key (string, optional): Memory slot key (required for get/set/delete)
  - value (any, optional): Value to store (required for set)

Returns:
  Varies by action — see individual action docs.`,
    inputSchema: AgentMemorySchema,
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
  },
  async (params) => {
    try {
      return ok(agentMemory(
        params.swarmId, params.agentId, params.action,
        params.key, params.value,
      ));
    } catch (e) { return fail(e); }
  },
);

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
