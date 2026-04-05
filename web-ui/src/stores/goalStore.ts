import { create } from 'zustand'

export interface Goal {
  id: string
  description: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  priority: number
  created_at: string
  updated_at: string
  progress?: number
  cycles?: number
  max_cycles?: number
}

interface GoalState {
  goals: Goal[]
  selectedGoal: Goal | null
  isLoading: boolean
  error: string | null
  wsConnected: boolean
  fetchGoals: () => Promise<void>
  addGoal: (description: string) => Promise<void>
  cancelGoal: (id: string) => Promise<void>
  selectGoal: (goal: Goal | null) => void
  updateGoalStatus: (id: string, status: Goal['status']) => void
  setGoals: (goals: Goal[]) => void
  setWsConnected: (connected: boolean) => void
}

export const useGoalStore = create<GoalState>((set, get) => ({
  goals: [],
  selectedGoal: null,
  isLoading: false,
  error: null,
  wsConnected: false,

  fetchGoals: async () => {
    set({ isLoading: true, error: null })
    try {
      const response = await fetch('/api/goals')
      if (!response.ok) throw new Error('Failed to fetch goals')
      const goals = await response.json()
      set({ goals, isLoading: false })
    } catch (error) {
      set({ error: (error as Error).message, isLoading: false })
    }
  },

  addGoal: async (description: string) => {
    try {
      const response = await fetch('/api/goals', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description }),
      })
      if (!response.ok) throw new Error('Failed to add goal')
      const newGoal = await response.json()
      set((state) => ({ goals: [newGoal, ...state.goals] }))
    } catch (error) {
      set({ error: (error as Error).message })
    }
  },

  cancelGoal: async (id: string) => {
    try {
      const response = await fetch(`/api/goals/${id}/cancel`, { method: 'POST' })
      if (!response.ok) throw new Error('Failed to cancel goal')
      set((state) => ({
        goals: state.goals.map((g) =>
          g.id === id ? { ...g, status: 'failed' as const } : g
        ),
      }))
    } catch (error) {
      set({ error: (error as Error).message })
    }
  },

  selectGoal: (goal) => set({ selectedGoal: goal }),

  updateGoalStatus: (id, status) => {
    set((state) => ({
      goals: state.goals.map((g) =>
        g.id === id ? { ...g, status } : g
      ),
    }))
  },

  setGoals: (goals) => set({ goals }),

  setWsConnected: (connected) => set({ wsConnected: connected }),
}))

// WebSocket hook for live updates
export function useGoalWebSocket() {
  const { setGoals, setWsConnected, fetchGoals } = useGoalStore()

  useEffect(() => {
    let ws: WebSocket | null = null
    let reconnectTimeout: NodeJS.Timeout
    let pingInterval: NodeJS.Timeout

    const connect = () => {
      const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`
      
      ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        console.log('WebSocket connected')
        setWsConnected(true)
        
        // Send initial ping
        ws?.send(JSON.stringify({ type: 'ping' }))
        
        // Start ping interval
        pingInterval = setInterval(() => {
          if (ws?.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }))
          }
        }, 30000)
      }

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          
          switch (message.type) {
            case 'initial':
              if (message.payload?.goals) {
                setGoals(message.payload.goals)
              }
              break
            case 'update':
              // Refresh goals on update
              fetchGoals()
              break
            case 'goal_created':
            case 'goal_updated':
              fetchGoals()
              break
            case 'pong':
              // Ping acknowledged
              break
          }
        } catch (error) {
          console.error('WebSocket message error:', error)
        }
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected')
        setWsConnected(false)
        clearInterval(pingInterval)
        
        // Reconnect after 3 seconds
        reconnectTimeout = setTimeout(connect, 3000)
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        ws?.close()
      }
    }

    connect()

    return () => {
      clearTimeout(reconnectTimeout)
      clearInterval(pingInterval)
      ws?.close()
    }
  }, [setGoals, setWsConnected, fetchGoals])
}

import { useEffect } from 'react'
