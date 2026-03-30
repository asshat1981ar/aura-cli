import { z } from "zod";

// ─── Enums ─────────────────────────────────────────────────────
export const TopologyEnum = z.enum(["hierarchical", "flat", "mesh", "ring", "star"]);
export const StrategyEnum = z.enum(["specialized", "generalist", "hybrid", "round_robin"]);
export const AgentRoleEnum = z.enum([
  "coordinator", "code_generator", "code_reviewer", "tester",
  "architect", "documenter", "debugger", "security_auditor",
  "devops", "data_analyst", "ux_designer", "custom"
]);
export const TaskPriorityEnum = z.enum(["critical", "high", "normal", "low"]);

// ─── swarm_init ────────────────────────────────────────────────
export const SwarmInitSchema = z.object({
  name: z.string()
    .min(1).max(64)
    .default("default-swarm")
    .describe("Human-readable name for the swarm instance"),
  topology: TopologyEnum
    .default("hierarchical")
    .describe("Agent network topology: hierarchical (tree), flat (peer), mesh (all-to-all), ring, or star"),
  maxAgents: z.number()
    .int().min(1).max(32)
    .default(8)
    .describe("Maximum number of agents allowed in this swarm (1-32)"),
  strategy: StrategyEnum
    .default("specialized")
    .describe("Agent assignment strategy: specialized (role-based), generalist (any agent), hybrid, or round_robin"),
}).strict();

// ─── swarm_add_agent ───────────────────────────────────────────
export const AddAgentSchema = z.object({
  swarmId: z.string()
    .describe("ID of the swarm to add the agent to"),
  name: z.string()
    .min(1).max(64)
    .describe("Display name for the agent (e.g., 'Lead Architect')"),
  role: AgentRoleEnum
    .describe("Functional role determining what tasks this agent handles"),
  parentId: z.string()
    .optional()
    .describe("ID of the parent agent in a hierarchical topology. Omit for top-level agents."),
  capabilities: z.array(z.string())
    .default([])
    .describe("List of capability tags (e.g., ['typescript', 'testing', 'react'])"),
}).strict();

// ─── swarm_remove_agent ────────────────────────────────────────
export const RemoveAgentSchema = z.object({
  swarmId: z.string().describe("ID of the swarm"),
  agentId: z.string().describe("ID of the agent to remove"),
  reassignTasks: z.boolean()
    .default(true)
    .describe("Whether to reassign the agent's current tasks to other agents"),
}).strict();

// ─── swarm_create_task ─────────────────────────────────────────
export const CreateTaskSchema = z.object({
  swarmId: z.string().describe("ID of the swarm"),
  title: z.string().min(1).max(256).describe("Short title for the task"),
  description: z.string().max(4096).describe("Detailed task description and requirements"),
  priority: TaskPriorityEnum.default("normal").describe("Task priority level"),
  requiredRole: AgentRoleEnum.optional().describe("Restrict assignment to agents with this role"),
  parentTaskId: z.string().optional().describe("ID of a parent task (for subtask decomposition)"),
  assignToAgentId: z.string().optional().describe("Directly assign to a specific agent by ID"),
}).strict();

// ─── swarm_delegate_task ───────────────────────────────────────
export const DelegateTaskSchema = z.object({
  swarmId: z.string().describe("ID of the swarm"),
  taskId: z.string().describe("ID of the task to delegate"),
  agentId: z.string().describe("ID of the agent to assign the task to"),
}).strict();

// ─── swarm_complete_task ───────────────────────────────────────
export const CompleteTaskSchema = z.object({
  swarmId: z.string().describe("ID of the swarm"),
  taskId: z.string().describe("ID of the task to mark complete"),
  result: z.string().max(8192).describe("The output/result of the completed task"),
}).strict();

// ─── swarm_send_message ────────────────────────────────────────
export const SendMessageSchema = z.object({
  swarmId: z.string().describe("ID of the swarm"),
  fromAgentId: z.string().describe("ID of the sending agent"),
  toAgentId: z.string().describe("ID of the receiving agent, or 'broadcast' for all agents"),
  type: z.enum(["task_assignment", "result", "status_update", "query", "directive"])
    .describe("Message type classification"),
  payload: z.unknown().describe("Message content (structured data or text)"),
}).strict();

// ─── swarm_status ──────────────────────────────────────────────
export const SwarmStatusSchema = z.object({
  swarmId: z.string().describe("ID of the swarm to query"),
}).strict();

// ─── swarm_list ────────────────────────────────────────────────
export const SwarmListSchema = z.object({}).strict();

// ─── swarm_terminate ───────────────────────────────────────────
export const SwarmTerminateSchema = z.object({
  swarmId: z.string().describe("ID of the swarm to terminate"),
}).strict();

// ─── swarm_agent_memory ────────────────────────────────────────
const AgentMemoryBaseSchema = z.object({
  swarmId: z.string().describe("ID of the swarm"),
  agentId: z.string().describe("ID of the agent"),
});

export const AgentMemorySchema = z.discriminatedUnion("action", [
  AgentMemoryBaseSchema.extend({
    action: z.literal("get"),
    key: z.string().describe("Memory slot key to read"),
    value: z.unknown().optional(),
  }),
  AgentMemoryBaseSchema.extend({
    action: z.literal("set"),
    key: z.string().describe("Memory slot key to write"),
    value: z.unknown().describe("Value to store"),
  }),
  AgentMemoryBaseSchema.extend({
    action: z.literal("delete"),
    key: z.string().describe("Memory slot key to delete"),
    value: z.unknown().optional(),
  }),
  AgentMemoryBaseSchema.extend({
    action: z.literal("list"),
    key: z.string().optional(),
    value: z.unknown().optional(),
  }),
]);
export const AgentMemorySchema = z.object({
  swarmId: z.string().describe("ID of the swarm"),
  agentId: z.string().describe("ID of the agent"),
  action: z.enum(["get", "set", "delete", "list"])
    .describe("Memory operation to perform"),
  key: z.string().optional().describe("Memory slot key (required for get/set/delete)"),
  value: z.unknown().optional().describe("Value to store (required for set)"),
}).strict();
