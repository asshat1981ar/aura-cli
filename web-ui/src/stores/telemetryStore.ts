import { create } from 'zustand'

export interface TelemetryRecord {
  id: string
  agent_name: string
  timestamp: string
  status: 'success' | 'error' | 'running'
  latency?: number
  tokens?: number
  message?: string
  details?: Record<string, any>
}

export interface TelemetrySummary {
  total_records: number
  avg_latency_ms: number
  total_tokens: number
  success_rate: number
  by_agent: Record<string, {
    count: number
    avg_latency: number
    success_rate: number
  }>
  by_hour: Array<{
    hour: string
    count: number
    avg_latency: number
    errors: number
  }>
}

export interface SystemStats {
  goals: {
    total: number
    pending: number
    running: number
    completed: number
    failed: number
    completion_rate: number
  }
  agents: {
    total: number
    active: number
    idle: number
    paused: number
  }
  telemetry: {
    total_records: number
    avg_latency_ms: number
    throughput_goals_per_hour: number
  }
  system: {
    uptime_seconds: number
    memory_usage_mb: number
    cpu_percent: number
  }
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'critical'
  checks: {
    agents: 'pass' | 'warn' | 'fail'
    queue: 'pass' | 'warn' | 'fail'
    api: 'pass' | 'warn' | 'fail'
  }
  timestamp: string
}

interface TelemetryState {
  records: TelemetryRecord[]
  summary: TelemetrySummary | null
  stats: SystemStats | null
  health: HealthStatus | null
  isLoading: boolean
  error: string | null
  
  // Actions
  fetchRecords: (limit?: number) => Promise<void>
  fetchSummary: () => Promise<void>
  fetchStats: () => Promise<void>
  fetchHealth: () => Promise<void>
  refreshAll: () => Promise<void>
  clearError: () => void
}

export const useTelemetryStore = create<TelemetryState>((set, get) => ({
  records: [],
  summary: null,
  stats: null,
  health: null,
  isLoading: false,
  error: null,

  fetchRecords: async (limit = 100) => {
    try {
      const response = await fetch(`/api/telemetry?limit=${limit}`)
      if (!response.ok) throw new Error('Failed to fetch telemetry records')
      const records = await response.json()
      set({ records })
    } catch (error) {
      console.error('Failed to fetch telemetry records:', error)
    }
  },

  fetchSummary: async () => {
    try {
      const response = await fetch('/api/telemetry/summary')
      if (!response.ok) throw new Error('Failed to fetch telemetry summary')
      const summary = await response.json()
      set({ summary })
    } catch (error) {
      console.error('Failed to fetch telemetry summary:', error)
    }
  },

  fetchStats: async () => {
    try {
      const response = await fetch('/api/stats')
      if (!response.ok) throw new Error('Failed to fetch system stats')
      const stats = await response.json()
      set({ stats })
    } catch (error) {
      console.error('Failed to fetch system stats:', error)
    }
  },

  fetchHealth: async () => {
    try {
      const response = await fetch('/api/health')
      if (!response.ok) throw new Error('Failed to fetch health status')
      const health = await response.json()
      set({ health })
    } catch (error) {
      console.error('Failed to fetch health status:', error)
    }
  },

  refreshAll: async () => {
    set({ isLoading: true, error: null })
    try {
      await Promise.all([
        get().fetchRecords(),
        get().fetchSummary(),
        get().fetchStats(),
        get().fetchHealth()
      ])
      set({ isLoading: false })
    } catch (error) {
      set({ error: (error as Error).message, isLoading: false })
    }
  },

  clearError: () => {
    set({ error: null })
  }
}))
