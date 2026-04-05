import { create } from 'zustand'
import { useEffect } from 'react'

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
  wsConnected: boolean
  fetchAgents: () => Promise<void>
  selectAgent: (agent: Agent | null) => void
  updateAgentStatus: (id: string, status: Agent['status']) => void
  setAgents: (agents: Agent[]) => void
  setWsConnected: (connected: boolean) => void
}

export const useAgentStore = create<AgentState>((set) => ({
  agents: [],
  selectedAgent: null,
  isLoading: false,
  error: null,
  wsConnected: false,

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

  setAgents: (agents) => set({ agents }),

  setWsConnected: (connected) => set({ wsConnected: connected }),
}))

// WebSocket hook for live agent updates
export function useAgentWebSocket() {
  const { setAgents, setWsConnected, fetchAgents } = useAgentStore()

  useEffect(() => {
    let ws: WebSocket | null = null
    let reconnectTimeout: NodeJS.Timeout

    const connect = () => {
      const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`
      
      ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        console.log('Agent WebSocket connected')
        setWsConnected(true)
      }

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          
          switch (message.type) {
            case 'initial':
              if (message.payload?.agents) {
                setAgents(message.payload.agents)
              }
              break
            case 'update':
              // Refresh agents on update
              fetchAgents()
              break
          }
        } catch (error) {
          console.error('Agent WebSocket message error:', error)
        }
      }

      ws.onclose = () => {
        console.log('Agent WebSocket disconnected')
        setWsConnected(false)
        
        // Reconnect after 3 seconds
        reconnectTimeout = setTimeout(connect, 3000)
      }

      ws.onerror = (error) => {
        console.error('Agent WebSocket error:', error)
        ws?.close()
      }
    }

    connect()

    return () => {
      clearTimeout(reconnectTimeout)
      ws?.close()
    }
  }, [setAgents, setWsConnected, fetchAgents])
}
