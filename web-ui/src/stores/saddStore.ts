import { create } from 'zustand'

export interface Workstream {
  id: string
  name: string
  description: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  dependencies: string[]
  artifacts: string[]
  progress: number
  started_at?: string
  completed_at?: string
}

export interface SADDSession {
  id: string
  title: string
  design_spec: string
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed'
  workstreams: Workstream[]
  created_at: string
  updated_at: string
  artifacts: string[]
}

interface SADDState {
  sessions: SADDSession[]
  activeSessionId: string | null
  isLoading: boolean
  error: string | null
  
  // Actions
  fetchSessions: () => Promise<void>
  createSession: (title: string, designSpec: string) => Promise<string>
  startSession: (id: string) => Promise<void>
  pauseSession: (id: string) => Promise<void>
  resumeSession: (id: string) => Promise<void>
  stopSession: (id: string) => Promise<void>
  deleteSession: (id: string) => Promise<void>
  setActiveSession: (id: string | null) => void
  getSession: (id: string) => SADDSession | undefined
}

export const useSADDStore = create<SADDState>((set, get) => ({
  sessions: [],
  activeSessionId: null,
  isLoading: false,
  error: null,

  fetchSessions: async () => {
    set({ isLoading: true, error: null })
    try {
      const response = await fetch('/api/sadd/sessions')
      if (!response.ok) throw new Error('Failed to fetch sessions')
      const sessions = await response.json()
      set({ sessions, isLoading: false })
    } catch (error) {
      set({ error: (error as Error).message, isLoading: false })
    }
  },

  createSession: async (title, designSpec) => {
    set({ isLoading: true, error: null })
    try {
      const response = await fetch('/api/sadd/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, design_spec: designSpec }),
      })
      if (!response.ok) throw new Error('Failed to create session')
      const session = await response.json()
      set((state) => ({
        sessions: [session, ...state.sessions],
        activeSessionId: session.id,
        isLoading: false,
      }))
      return session.id
    } catch (error) {
      set({ error: (error as Error).message, isLoading: false })
      throw error
    }
  },

  startSession: async (id) => {
    try {
      const response = await fetch(`/api/sadd/sessions/${id}/start`, { method: 'POST' })
      if (!response.ok) throw new Error('Failed to start session')
      
      set((state) => ({
        sessions: state.sessions.map((s) =>
          s.id === id ? { ...s, status: 'running' as const } : s
        ),
      }))
    } catch (error) {
      set({ error: (error as Error).message })
    }
  },

  pauseSession: async (id) => {
    try {
      const response = await fetch(`/api/sadd/sessions/${id}/pause`, { method: 'POST' })
      if (!response.ok) throw new Error('Failed to pause session')
      
      set((state) => ({
        sessions: state.sessions.map((s) =>
          s.id === id ? { ...s, status: 'paused' as const } : s
        ),
      }))
    } catch (error) {
      set({ error: (error as Error).message })
    }
  },

  resumeSession: async (id) => {
    try {
      const response = await fetch(`/api/sadd/sessions/${id}/resume`, { method: 'POST' })
      if (!response.ok) throw new Error('Failed to resume session')
      
      set((state) => ({
        sessions: state.sessions.map((s) =>
          s.id === id ? { ...s, status: 'running' as const } : s
        ),
      }))
    } catch (error) {
      set({ error: (error as Error).message })
    }
  },

  stopSession: async (id) => {
    try {
      const response = await fetch(`/api/sadd/sessions/${id}/stop`, { method: 'POST' })
      if (!response.ok) throw new Error('Failed to stop session')
      
      set((state) => ({
        sessions: state.sessions.map((s) =>
          s.id === id ? { ...s, status: 'failed' as const } : s
        ),
      }))
    } catch (error) {
      set({ error: (error as Error).message })
    }
  },

  deleteSession: async (id) => {
    try {
      const response = await fetch(`/api/sadd/sessions/${id}`, { method: 'DELETE' })
      if (!response.ok) throw new Error('Failed to delete session')
      
      set((state) => ({
        sessions: state.sessions.filter((s) => s.id !== id),
        activeSessionId: state.activeSessionId === id ? null : state.activeSessionId,
      }))
    } catch (error) {
      set({ error: (error as Error).message })
    }
  },

  setActiveSession: (id) => set({ activeSessionId: id }),

  getSession: (id) => {
    return get().sessions.find((s) => s.id === id)
  },
}))
