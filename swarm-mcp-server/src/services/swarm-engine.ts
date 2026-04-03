import { v4 as uuidv4 } from "uuid";
import type {
  Swarm, SwarmConfig, Agent, Task, AgentMessage,
  AgentRole, AgentStatus, TaskPriority, TaskStatus,
} from "../types.js";

// ─── In-Memory Swarm Store ─────────────────────────────────────
const swarms = new Map<string, Swarm>();

// ─── Helpers ───────────────────────────────────────────────────
function now(): string {
  return new Date().toISOString();
}

function getSwarm(swarmId: string): Swarm {
  const swarm = swarms.get(swarmId);
  if (!swarm) throw new Error(`Swarm '${swarmId}' not found. Use swarm_list to see active swarms.`);
  if (swarm.status === "terminated") throw new Error(`Swarm '${swarmId}' has been terminated.`);
  return swarm;
}

function getAgent(swarm: Swarm, agentId: string): Agent {
  const agent = swarm.agents.get(agentId);
  if (!agent) throw new Error(`Agent '${agentId}' not found in swarm '${swarm.id}'. Use swarm_status to see agents.`);
  return agent;
}

function getTask(swarm: Swarm, taskId: string): Task {
  const task = swarm.tasks.get(taskId);
  if (!task) throw new Error(`Task '${taskId}' not found in swarm '${swarm.id}'.`);
  return task;
}

function findBestAgent(swarm: Swarm, role?: AgentRole | null): Agent | null {
  const candidates = Array.from(swarm.agents.values()).filter((a) => {
    if (a.status === "terminated") return false;
    if (role && a.role !== role && a.role !== "coordinator") return false;
    return true;
  });

  // Prefer idle agents, then those with fewest completed (least loaded)
  candidates.sort((a, b) => {
    const statusOrder: Record<AgentStatus, number> = {
      idle: 0, waiting: 1, completed: 2, busy: 3, error: 4, terminated: 5,
    };
    const diff = statusOrder[a.status] - statusOrder[b.status];
    if (diff !== 0) return diff;
    return a.completedTasks - b.completedTasks;
  });

  return candidates[0] ?? null;
}

function serializeAgent(agent: Agent): Record<string, unknown> {
  return { ...agent };
}

function serializeTask(task: Task): Record<string, unknown> {
  return { ...task };
}

// ─── Service Functions ─────────────────────────────────────────

export function swarmInit(config: SwarmConfig): Record<string, unknown> {
  const id = uuidv4();
  const swarm: Swarm = {
    id,
    config,
    agents: new Map(),
    tasks: new Map(),
    messageLog: [],
    status: "active",
    createdAt: now(),
  };

  // In hierarchical mode, auto-create a coordinator
  if (config.topology === "hierarchical") {
    const coordId = uuidv4();
    const coordinator: Agent = {
      id: coordId,
      name: "Coordinator",
      role: "coordinator",
      status: "idle",
      parentId: null,
      childIds: [],
      capabilities: ["orchestration", "delegation", "monitoring"],
      currentTaskId: null,
      completedTasks: 0,
      memorySlots: {},
      createdAt: now(),
      lastActiveAt: now(),
    };
    swarm.agents.set(coordId, coordinator);
  }

  swarms.set(id, swarm);

  return {
    swarmId: id,
    status: swarm.status,
    topology: config.topology,
    strategy: config.strategy,
    maxAgents: config.maxAgents,
    agentCount: swarm.agents.size,
    agents: Array.from(swarm.agents.values()).map(serializeAgent),
    message: `Swarm '${config.name}' initialized with ${config.topology} topology and ${config.strategy} strategy.` +
      (config.topology === "hierarchical" ? " A coordinator agent was auto-created." : ""),
  };
}

export function addAgent(
  swarmId: string,
  name: string,
  role: AgentRole,
  parentId?: string,
  capabilities: string[] = [],
): Record<string, unknown> {
  const swarm = getSwarm(swarmId);

  if (swarm.agents.size >= swarm.config.maxAgents) {
    throw new Error(
      `Swarm has reached its maximum of ${swarm.config.maxAgents} agents. ` +
      `Remove an agent first or initialize a new swarm with a higher maxAgents.`
    );
  }

  // Validate parent in hierarchical mode
  if (swarm.config.topology === "hierarchical" && parentId) {
    const parent = getAgent(swarm, parentId);
    parent.childIds.push(uuidv4()); // will fix below
  }

  const agentId = uuidv4();
  const agent: Agent = {
    id: agentId,
    name,
    role,
    status: "idle",
    parentId: parentId ?? null,
    childIds: [],
    capabilities,
    currentTaskId: null,
    completedTasks: 0,
    memorySlots: {},
    createdAt: now(),
    lastActiveAt: now(),
  };

  // Fix parent's childIds to use actual ID
  if (parentId) {
    const parent = swarm.agents.get(parentId);
    if (parent) {
      parent.childIds.pop(); // remove placeholder
      parent.childIds.push(agentId);
    }
  }

  swarm.agents.set(agentId, agent);

  return {
    agentId,
    name,
    role,
    parentId: parentId ?? null,
    capabilities,
    swarmAgentCount: swarm.agents.size,
    maxAgents: swarm.config.maxAgents,
  };
}

export function removeAgent(
  swarmId: string,
  agentId: string,
  reassignTasks: boolean = true,
): Record<string, unknown> {
  const swarm = getSwarm(swarmId);
  const agent = getAgent(swarm, agentId);

  const reassigned: string[] = [];

  // Reassign current task if any
  if (agent.currentTaskId && reassignTasks) {
    const task = swarm.tasks.get(agent.currentTaskId);
    if (task && task.status !== "completed" && task.status !== "failed") {
      const replacement = findBestAgent(swarm, task.requiredRole);
      if (replacement && replacement.id !== agentId) {
        task.assignedAgentId = replacement.id;
        replacement.currentTaskId = task.id;
        replacement.status = "busy";
        replacement.lastActiveAt = now();
        reassigned.push(task.id);
      } else {
        task.assignedAgentId = null;
        task.status = "queued";
      }
    }
  }

  // Reparent children
  for (const childId of agent.childIds) {
    const child = swarm.agents.get(childId);
    if (child) {
      child.parentId = agent.parentId;
      if (agent.parentId) {
        const grandparent = swarm.agents.get(agent.parentId);
        grandparent?.childIds.push(childId);
      }
    }
  }

  // Remove from parent's childIds
  if (agent.parentId) {
    const parent = swarm.agents.get(agent.parentId);
    if (parent) {
      parent.childIds = parent.childIds.filter((id) => id !== agentId);
    }
  }

  swarm.agents.delete(agentId);

  return {
    removedAgentId: agentId,
    reassignedTaskIds: reassigned,
    remainingAgents: swarm.agents.size,
  };
}

export function createTask(
  swarmId: string,
  title: string,
  description: string,
  priority: TaskPriority = "normal",
  requiredRole?: AgentRole,
  parentTaskId?: string,
  assignToAgentId?: string,
): Record<string, unknown> {
  const swarm = getSwarm(swarmId);
  const taskId = uuidv4();

  const task: Task = {
    id: taskId,
    title,
    description,
    priority,
    status: "queued",
    assignedAgentId: null,
    requiredRole: requiredRole ?? null,
    parentTaskId: parentTaskId ?? null,
    subtaskIds: [],
    result: null,
    createdAt: now(),
    updatedAt: now(),
  };

  // Link to parent task
  if (parentTaskId) {
    const parent = getTask(swarm, parentTaskId);
    parent.subtaskIds.push(taskId);
  }

  // Auto-assign
  if (assignToAgentId) {
    const agent = getAgent(swarm, assignToAgentId);
    task.assignedAgentId = agent.id;
    task.status = "assigned";
    agent.currentTaskId = taskId;
    agent.status = "busy";
    agent.lastActiveAt = now();
  } else if (swarm.config.strategy !== "generalist") {
    const agent = findBestAgent(swarm, requiredRole);
    if (agent && agent.status === "idle") {
      task.assignedAgentId = agent.id;
      task.status = "assigned";
      agent.currentTaskId = taskId;
      agent.status = "busy";
      agent.lastActiveAt = now();
    }
  }

  swarm.tasks.set(taskId, task);

  return {
    taskId,
    title,
    priority,
    status: task.status,
    assignedAgentId: task.assignedAgentId,
    parentTaskId: task.parentTaskId,
  };
}

export function delegateTask(
  swarmId: string,
  taskId: string,
  agentId: string,
): Record<string, unknown> {
  const swarm = getSwarm(swarmId);
  const task = getTask(swarm, taskId);
  const agent = getAgent(swarm, agentId);

  // Unassign previous agent
  if (task.assignedAgentId) {
    const prev = swarm.agents.get(task.assignedAgentId);
    if (prev) {
      prev.currentTaskId = null;
      prev.status = "idle";
      prev.lastActiveAt = now();
    }
  }

  task.assignedAgentId = agentId;
  task.status = "assigned";
  task.updatedAt = now();
  agent.currentTaskId = taskId;
  agent.status = "busy";
  agent.lastActiveAt = now();

  // Log message
  const msg: AgentMessage = {
    id: uuidv4(),
    fromAgentId: "system",
    toAgentId: agentId,
    type: "task_assignment",
    payload: { taskId, title: task.title },
    timestamp: now(),
  };
  swarm.messageLog.push(msg);

  return {
    taskId,
    assignedAgentId: agentId,
    agentName: agent.name,
    status: task.status,
  };
}

export function completeTask(
  swarmId: string,
  taskId: string,
  result: string,
): Record<string, unknown> {
  const swarm = getSwarm(swarmId);
  const task = getTask(swarm, taskId);

  task.status = "completed";
  task.result = result;
  task.updatedAt = now();

  if (task.assignedAgentId) {
    const agent = swarm.agents.get(task.assignedAgentId);
    if (agent) {
      agent.currentTaskId = null;
      agent.status = "idle";
      agent.completedTasks += 1;
      agent.lastActiveAt = now();
    }
  }

  // Check if parent task's subtasks are all complete
  let parentComplete = false;
  if (task.parentTaskId) {
    const parent = swarm.tasks.get(task.parentTaskId);
    if (parent) {
      const allDone = parent.subtaskIds.every((sid) => {
        const sub = swarm.tasks.get(sid);
        return sub?.status === "completed";
      });
      if (allDone) {
        parentComplete = true;
      }
    }
  }

  return {
    taskId,
    status: "completed",
    result,
    parentTaskAllSubtasksComplete: parentComplete,
  };
}

export function sendMessage(
  swarmId: string,
  fromAgentId: string,
  toAgentId: string,
  type: AgentMessage["type"],
  payload: unknown,
): Record<string, unknown> {
  const swarm = getSwarm(swarmId);
  getAgent(swarm, fromAgentId); // validate sender
  if (toAgentId !== "broadcast") getAgent(swarm, toAgentId); // validate recipient

  const msg: AgentMessage = {
    id: uuidv4(),
    fromAgentId,
    toAgentId,
    type,
    payload,
    timestamp: now(),
  };
  swarm.messageLog.push(msg);

  return {
    messageId: msg.id,
    from: fromAgentId,
    to: toAgentId,
    type,
    timestamp: msg.timestamp,
  };
}

export function swarmStatus(swarmId: string): Record<string, unknown> {
  const swarm = getSwarm(swarmId);

  const agents = Array.from(swarm.agents.values()).map(serializeAgent);
  const tasks = Array.from(swarm.tasks.values()).map(serializeTask);

  const tasksByStatus: Record<string, number> = {};
  for (const t of swarm.tasks.values()) {
    tasksByStatus[t.status] = (tasksByStatus[t.status] ?? 0) + 1;
  }

  const agentsByStatus: Record<string, number> = {};
  for (const a of swarm.agents.values()) {
    agentsByStatus[a.status] = (agentsByStatus[a.status] ?? 0) + 1;
  }

  return {
    swarmId: swarm.id,
    name: swarm.config.name,
    status: swarm.status,
    topology: swarm.config.topology,
    strategy: swarm.config.strategy,
    maxAgents: swarm.config.maxAgents,
    agentCount: swarm.agents.size,
    taskCount: swarm.tasks.size,
    messageCount: swarm.messageLog.length,
    agentsByStatus,
    tasksByStatus,
    agents,
    tasks,
    recentMessages: swarm.messageLog.slice(-10),
  };
}

export function swarmList(): Record<string, unknown> {
  const list = Array.from(swarms.values()).map((s) => ({
    swarmId: s.id,
    name: s.config.name,
    status: s.status,
    topology: s.config.topology,
    strategy: s.config.strategy,
    agentCount: s.agents.size,
    taskCount: s.tasks.size,
    createdAt: s.createdAt,
  }));

  return { swarms: list, total: list.length };
}

export function swarmTerminate(swarmId: string): Record<string, unknown> {
  const swarm = getSwarm(swarmId);
  swarm.status = "terminated";

  for (const agent of swarm.agents.values()) {
    agent.status = "terminated";
  }

  return {
    swarmId,
    status: "terminated",
    agentsTerminated: swarm.agents.size,
    tasksInProgress: Array.from(swarm.tasks.values()).filter(
      (t) => t.status !== "completed" && t.status !== "failed"
    ).length,
  };
}

export function agentMemory(
  swarmId: string,
  agentId: string,
  action: "get" | "set" | "delete" | "list",
  key?: string,
  value?: unknown,
): Record<string, unknown> {
  const swarm = getSwarm(swarmId);
  const agent = getAgent(swarm, agentId);

  switch (action) {
    case "list":
      return {
        agentId,
        memorySlots: Object.keys(agent.memorySlots),
        count: Object.keys(agent.memorySlots).length,
      };
    case "get":
      if (!key) throw new Error("'key' is required for get action.");
      return {
        agentId,
        key,
        value: agent.memorySlots[key] ?? null,
        exists: key in agent.memorySlots,
      };
    case "set":
      if (!key) throw new Error("'key' is required for set action.");
      agent.memorySlots[key] = value;
      agent.lastActiveAt = now();
      return { agentId, key, stored: true };
    case "delete":
      if (!key) throw new Error("'key' is required for delete action.");
      const existed = key in agent.memorySlots;
      delete agent.memorySlots[key];
      return { agentId, key, deleted: existed };
    default:
      throw new Error(`Unknown memory action '${action}'.`);
  }
}
