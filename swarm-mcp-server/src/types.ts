// ─── Agent Types ───────────────────────────────────────────────
export type AgentRole =
  | "coordinator"
  | "code_generator"
  | "code_reviewer"
  | "tester"
  | "architect"
  | "documenter"
  | "debugger"
  | "security_auditor"
  | "devops"
  | "data_analyst"
  | "ux_designer"
  | "custom";

export type AgentStatus = "idle" | "busy" | "waiting" | "completed" | "error" | "terminated";

export type Topology = "hierarchical" | "flat" | "mesh" | "ring" | "star";

export type Strategy = "specialized" | "generalist" | "hybrid" | "round_robin";

export type TaskPriority = "critical" | "high" | "normal" | "low";

export type TaskStatus = "queued" | "assigned" | "in_progress" | "review" | "completed" | "failed";

// ─── Agent ─────────────────────────────────────────────────────
export interface Agent {
  id: string;
  name: string;
  role: AgentRole;
  status: AgentStatus;
  parentId: string | null;
  childIds: string[];
  capabilities: string[];
  currentTaskId: string | null;
  completedTasks: number;
  memorySlots: Record<string, unknown>;
  createdAt: string;
  lastActiveAt: string;
}

// ─── Task ──────────────────────────────────────────────────────
export interface Task {
  id: string;
  title: string;
  description: string;
  priority: TaskPriority;
  status: TaskStatus;
  assignedAgentId: string | null;
  requiredRole: AgentRole | null;
  parentTaskId: string | null;
  subtaskIds: string[];
  result: string | null;
  createdAt: string;
  updatedAt: string;
}

// ─── Message ───────────────────────────────────────────────────
export interface AgentMessage {
  id: string;
  fromAgentId: string;
  toAgentId: string | "broadcast";
  type: "task_assignment" | "result" | "status_update" | "query" | "directive";
  payload: unknown;
  timestamp: string;
}

// ─── Swarm ─────────────────────────────────────────────────────
export interface SwarmConfig {
  topology: Topology;
  maxAgents: number;
  strategy: Strategy;
  name: string;
}

export interface Swarm {
  id: string;
  config: SwarmConfig;
  agents: Map<string, Agent>;
  tasks: Map<string, Task>;
  messageLog: AgentMessage[];
  status: "initializing" | "active" | "paused" | "terminated";
  createdAt: string;
}
