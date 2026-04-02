import { create } from 'zustand'

export interface Agent {
  id: string
  name: string
  type: string
  status: 'idle' | 'busy' | 'error' | 'offline'
  current_task?: string
  capabilities: string[]
  last_seen: string
  stats: {
    tasks_completed: number
    tasks_failed: number
    avg_execution_time: number
  }
}

interface AgentState {
  agents: Agent[]
  selectedAgent: Agent | null
  isLoading: boolean
  error: string | null
  fetchAgents: () => Promise<void>
  selectAgent: (agent: Agent | null) => void
  updateAgentStatus: (id: string, status: Agent['status']) => void
}

export const useAgentStore = create<AgentState>((set) => ({
  agents: [],
  selectedAgent: null,
  isLoading: false,
  error: null,

  fetchAgents: async () => {
    set({ isLoading: true, error: null })
    try {
      const response = await fetch('/api/agents')
      if (!response.ok) throw new Error('Failed to fetch agents')
      const agents = await response.json()
      set({ agents, isLoading: false })
    } catch (error) {
      set({ error: (error as Error).message, isLoading: false })
    }
  },

  selectAgent: (agent) => set({ selectedAgent: agent }),

  updateAgentStatus: (id, status) => {
    set((state) => ({
      agents: state.agents.map((a) =>
        a.id === id ? { ...a, status } : a
      ),
    }))
  },
}))
