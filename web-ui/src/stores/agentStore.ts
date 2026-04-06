import { useEffect } from 'react'
import { create } from 'zustand'

export interface AgentMetrics {
  total_executions: number
  success_rate: number
  avg_latency_ms: number
  total_tokens?: number
  errors?: number
}

export interface Agent {
  id: string
  name: string
  type: string
  status: 'idle' | 'busy' | 'paused' | 'error' | 'offline'
  capabilities: string[]
  metrics: AgentMetrics
  last_active?: string
  description?: string
  // Legacy compatibility
  current_task?: string
  last_seen?: string
  stats?: {
    tasks_completed: number
    tasks_failed: number
    avg_execution_time: number
  }
}

export interface AgentLog {
  id: string
  agent_name: string
  timestamp: string
  status: 'success' | 'error' | 'running'
  latency?: number
  tokens?: number
  message?: string
}

export interface AgentExecution {
  id: string
  agent_id: string
  goal_id?: string
  status: 'running' | 'completed' | 'failed'
  started_at: string
  completed_at?: string
  result?: any
  error?: string
}

interface AgentState {
  agents: Agent[]
  selectedAgent: Agent | null
  logs: AgentLog[]
  executions: AgentExecution[]
  overview: Agent[]
  isLoading: boolean
  error: string | null
  wsConnected: boolean
  
  // Actions
  fetchAgents: () => Promise<void>
  fetchOverview: () => Promise<void>
  selectAgent: (agent: Agent | null) => void
  fetchAgentLogs: (agentId: string, limit?: number) => Promise<void>
  fetchAgentMetrics: (agentId: string) => Promise<void>
  fetchAgentHistory: (agentId: string, limit?: number) => Promise<void>
  pauseAgent: (agentId: string) => Promise<boolean>
  resumeAgent: (agentId: string) => Promise<boolean>
  restartAgent: (agentId: string) => Promise<boolean>
  updateAgentStatus: (agentId: string, status: Agent['status']) => void
  setWsConnected: (connected: boolean) => void
  clearError: () => void
}

export const useAgentStore = create<AgentState>((set, get) => ({
  agents: [],
  selectedAgent: null,
  logs: [],
  executions: [],
  overview: [],
  isLoading: false,
  error: null,
  wsConnected: false,

  fetchAgents: async () => {
    set({ isLoading: true, error: null })
    try {
      const response = await fetch('/api/agents')
      if (!response.ok) {
        throw new Error('Failed to fetch agents')
      }
      const agents = await response.json()
      set({ agents, isLoading: false })
    } catch (error) {
      set({ error: (error as Error).message, isLoading: false })
    }
  },

  fetchOverview: async () => {
    set({ isLoading: true, error: null })
    try {
      const response = await fetch('/api/agents/overview')
      if (!response.ok) {
        throw new Error('Failed to fetch agents overview')
      }
      const overview = await response.json()
      set({ overview, isLoading: false })
    } catch (error) {
      set({ error: (error as Error).message, isLoading: false })
    }
  },

  selectAgent: (agent) => {
    set({ selectedAgent: agent })
    if (agent) {
      get().fetchAgentLogs(agent.id)
      get().fetchAgentMetrics(agent.id)
    }
  },

  fetchAgentLogs: async (agentId: string, limit = 50) => {
    try {
      const response = await fetch(`/api/agents/${agentId}/logs?limit=${limit}`)
      if (!response.ok) {
        throw new Error('Failed to fetch agent logs')
      }
      const logs = await response.json()
      set({ logs })
    } catch (error) {
      console.error('Failed to fetch agent logs:', error)
    }
  },

  fetchAgentMetrics: async (agentId: string) => {
    try {
      const response = await fetch(`/api/agents/${agentId}/metrics`)
      if (!response.ok) {
        throw new Error('Failed to fetch agent metrics')
      }
      const metrics = await response.json()
      
      // Update the selected agent's metrics
      set(state => ({
        selectedAgent: state.selectedAgent?.id === agentId 
          ? { ...state.selectedAgent, metrics }
          : state.selectedAgent,
        overview: state.overview.map(a => 
          a.id === agentId ? { ...a, metrics } : a
        )
      }))
    } catch (error) {
      console.error('Failed to fetch agent metrics:', error)
    }
  },

  fetchAgentHistory: async (agentId: string, limit = 20) => {
    try {
      const response = await fetch(`/api/agents/${agentId}/history?limit=${limit}`)
      if (!response.ok) {
        throw new Error('Failed to fetch agent history')
      }
      const history = await response.json()
      set({
        executions: history.map((h: any) => ({
          id: h.id || crypto.randomUUID(),
          agent_id: agentId,
          status: h.status,
          started_at: h.timestamp,
          result: h.result
        }))
      })
    } catch (error) {
      console.error('Failed to fetch agent history:', error)
    }
  },

  pauseAgent: async (agentId: string) => {
    try {
      const response = await fetch(`/api/agents/${agentId}/pause`, {
        method: 'POST'
      })
      if (!response.ok) return false
      
      get().updateAgentStatus(agentId, 'paused')
      return true
    } catch (error) {
      console.error('Failed to pause agent:', error)
      return false
    }
  },

  resumeAgent: async (agentId: string) => {
    try {
      const response = await fetch(`/api/agents/${agentId}/resume`, {
        method: 'POST'
      })
      if (!response.ok) return false
      
      get().updateAgentStatus(agentId, 'idle')
      return true
    } catch (error) {
      console.error('Failed to resume agent:', error)
      return false
    }
  },

  restartAgent: async (agentId: string) => {
    try {
      const response = await fetch(`/api/agents/${agentId}/restart`, {
        method: 'POST'
      })
      if (!response.ok) return false
      
      get().updateAgentStatus(agentId, 'idle')
      return true
    } catch (error) {
      console.error('Failed to restart agent:', error)
      return false
    }
  },

  updateAgentStatus: (agentId: string, status: Agent['status']) => {
    set((state) => ({
      agents: state.agents.map(a => 
        a.id === agentId ? { ...a, status } : a
      ),
      overview: state.overview.map(a => 
        a.id === agentId ? { ...a, status } : a
      ),
      selectedAgent: state.selectedAgent?.id === agentId 
        ? { ...state.selectedAgent, status }
        : state.selectedAgent
    }))
  },

  setWsConnected: (connected: boolean) => {
    set({ wsConnected: connected })
  },

  clearError: () => {
    set({ error: null })
  }
}))

// WebSocket hook for real-time agent updates
export function useAgentWebSocket() {
  const { setWsConnected, updateAgentStatus, fetchOverview } = useAgentStore()

  useEffect(() => {
    let ws: WebSocket | null = null
    let reconnectTimeout: NodeJS.Timeout

    const connect = () => {
      try {
        ws = new WebSocket(`ws://${window.location.host}/ws`)
        
        ws.onopen = () => {
          console.log('Agent WebSocket connected')
          setWsConnected(true)
        }
        
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            
            // Handle agent-related messages
            if (data.type === 'agent_status_change') {
              updateAgentStatus(data.agent_id, data.status)
            } else if (data.type === 'agent_paused' || data.type === 'agent_resumed' || data.type === 'agent_restarted') {
              fetchOverview()
            }
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error)
          }
        }
        
        ws.onclose = () => {
          console.log('Agent WebSocket disconnected')
          setWsConnected(false)
          // Attempt to reconnect after 5 seconds
          reconnectTimeout = setTimeout(connect, 5000)
        }
        
        ws.onerror = (error) => {
          console.error('Agent WebSocket error:', error)
          setWsConnected(false)
        }
      } catch (error) {
        console.error('Failed to create WebSocket:', error)
        setWsConnected(false)
      }
    }

    connect()
    
    return () => {
      if (reconnectTimeout) clearTimeout(reconnectTimeout)
      if (ws) {
        ws.close()
      }
    }
  }, [setWsConnected, updateAgentStatus, fetchOverview])
}
