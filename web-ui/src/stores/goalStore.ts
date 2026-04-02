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
  fetchGoals: () => Promise<void>
  addGoal: (description: string) => Promise<void>
  cancelGoal: (id: string) => Promise<void>
  selectGoal: (goal: Goal | null) => void
  updateGoalStatus: (id: string, status: Goal['status']) => void
}

export const useGoalStore = create<GoalState>((set) => ({
  goals: [],
  selectedGoal: null,
  isLoading: false,
  error: null,

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
}))
