import { create } from 'zustand'

export interface TerminalCommand {
  id: string
  command: string
  output: string
  exit_code: number
  timestamp: string
}

export interface TerminalSession {
  id: string
  commands: TerminalCommand[]
  cwd: string
  command_count: number
  last_active?: string
}

interface TerminalState {
  sessions: TerminalSession[]
  activeSessionId: string | null
  currentCwd: string
  isExecuting: boolean
  error: string | null
  
  // Actions
  createSession: () => string
  setActiveSession: (sessionId: string | null) => void
  executeCommand: (command: string, sessionId?: string) => Promise<void>
  fetchSessions: () => Promise<void>
  fetchSessionHistory: (sessionId: string) => Promise<void>
  clearSession: (sessionId: string) => Promise<void>
  deleteSession: (sessionId: string) => Promise<void>
  addOutput: (sessionId: string, command: TerminalCommand) => void
  clearError: () => void
}

export const useTerminalStore = create<TerminalState>((set, get) => ({
  sessions: [],
  activeSessionId: null,
  currentCwd: '/home/westonaaron675/aura-cli',
  isExecuting: false,
  error: null,

  createSession: () => {
    const sessionId = crypto.randomUUID()
    const newSession: TerminalSession = {
      id: sessionId,
      commands: [],
      cwd: '/home/westonaaron675/aura-cli',
      command_count: 0
    }
    set(state => ({
      sessions: [...state.sessions, newSession],
      activeSessionId: sessionId
    }))
    return sessionId
  },

  setActiveSession: (sessionId) => {
    set({ activeSessionId: sessionId })
    if (sessionId) {
      get().fetchSessionHistory(sessionId)
    }
  },

  executeCommand: async (command: string, sessionId?: string) => {
    const sid = sessionId || get().activeSessionId || get().createSession()
    set({ isExecuting: true, error: null })

    try {
      const response = await fetch('/api/terminal/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command,
          session_id: sid,
          cwd: get().currentCwd
        })
      })

      if (!response.ok) {
        throw new Error('Failed to execute command')
      }

      const result = await response.json()

      if (result.error) {
        const errorCommand: TerminalCommand = {
          id: crypto.randomUUID(),
          command,
          output: `Error: ${result.error}`,
          exit_code: 1,
          timestamp: new Date().toISOString()
        }
        get().addOutput(sid, errorCommand)
        set({ error: result.error, isExecuting: false })
      } else {
        const newCommand: TerminalCommand = {
          id: crypto.randomUUID(),
          command,
          output: result.output,
          exit_code: result.exit_code,
          timestamp: new Date().toISOString()
        }
        get().addOutput(sid, newCommand)
        set({ 
          currentCwd: result.cwd || get().currentCwd,
          isExecuting: false
        })
      }
    } catch (error) {
      set({ error: (error as Error).message, isExecuting: false })
    }
  },

  fetchSessions: async () => {
    try {
      const response = await fetch('/api/terminal/sessions')
      if (!response.ok) return
      
      const data = await response.json()
      set({ sessions: data.sessions })
    } catch (error) {
      console.error('Failed to fetch sessions:', error)
    }
  },

  fetchSessionHistory: async (sessionId: string) => {
    try {
      const response = await fetch(`/api/terminal/sessions/${sessionId}`)
      if (!response.ok) return
      
      const data = await response.json()
      set(state => ({
        sessions: state.sessions.map(s => 
          s.id === sessionId 
            ? { ...s, commands: data.commands, cwd: data.cwd }
            : s
        ),
        currentCwd: data.cwd
      }))
    } catch (error) {
      console.error('Failed to fetch session history:', error)
    }
  },

  clearSession: async (sessionId: string) => {
    try {
      await fetch(`/api/terminal/sessions/${sessionId}/clear`, {
        method: 'POST'
      })
      
      set(state => ({
        sessions: state.sessions.map(s => 
          s.id === sessionId 
            ? { ...s, commands: [], command_count: 0 }
            : s
        )
      }))
    } catch (error) {
      console.error('Failed to clear session:', error)
    }
  },

  deleteSession: async (sessionId: string) => {
    try {
      await fetch(`/api/terminal/sessions/${sessionId}`, {
        method: 'DELETE'
      })
      
      set(state => ({
        sessions: state.sessions.filter(s => s.id !== sessionId),
        activeSessionId: state.activeSessionId === sessionId 
          ? null 
          : state.activeSessionId
      }))
    } catch (error) {
      console.error('Failed to delete session:', error)
    }
  },

  addOutput: (sessionId: string, command: TerminalCommand) => {
    set(state => ({
      sessions: state.sessions.map(s => 
        s.id === sessionId 
          ? { 
              ...s, 
              commands: [...s.commands, command],
              command_count: s.command_count + 1,
              last_active: command.timestamp
            }
          : s
      )
    }))
  },

  clearError: () => {
    set({ error: null })
  }
}))
