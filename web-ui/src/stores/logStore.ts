import { create } from 'zustand'

export interface LogEntry {
  id: string
  timestamp: string
  level: 'DEBUG' | 'INFO' | 'WARN' | 'ERROR'
  event: string
  message?: string
  details?: Record<string, unknown>
}

interface LogState {
  logs: LogEntry[]
  isConnected: boolean
  filter: {
    level: string | null
    search: string
  }
  addLog: (log: LogEntry) => void
  setConnected: (connected: boolean) => void
  setFilter: (filter: Partial<LogState['filter']>) => void
  clearLogs: () => void
}

const MAX_LOGS = 1000

export const useLogStore = create<LogState>((set) => ({
  logs: [],
  isConnected: false,
  filter: {
    level: null,
    search: '',
  },

  addLog: (log) => {
    set((state) => {
      const newLogs = [log, ...state.logs]
      if (newLogs.length > MAX_LOGS) {
        newLogs.pop()
      }
      return { logs: newLogs }
    })
  },

  setConnected: (connected) => set({ isConnected: connected }),

  setFilter: (filter) =>
    set((state) => ({ filter: { ...state.filter, ...filter } })),

  clearLogs: () => set({ logs: [] }),
}))
