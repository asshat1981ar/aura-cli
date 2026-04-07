import { create } from 'zustand'

export interface NotificationSettings {
  goal_completion: boolean
  agent_status: boolean
  errors: boolean
  webhook_events: boolean
}

export interface ApiSettings {
  endpoint: string
  websocket: string
  timeout: number
}

export interface FeatureSettings {
  auto_refresh: boolean
  confirm_destructive: boolean
  show_telemetry: boolean
}

export interface Settings {
  theme: 'light' | 'dark' | 'system'
  refresh_interval: number
  notifications: NotificationSettings
  api: ApiSettings
  features: FeatureSettings
}

export interface MCPServerConfig {
  type: 'stdio' | 'http'
  command?: string
  args?: string[]
  url?: string
  env?: Record<string, string>
  headers?: Record<string, string>
}

export interface MCPSettings {
  mcpServers: Record<string, MCPServerConfig>
}

interface SettingsState {
  settings: Settings | null
  mcpConfig: MCPSettings | null
  isLoading: boolean
  saving: boolean
  error: string | null
  
  // Actions
  fetchSettings: () => Promise<void>
  updateSettings: (settings: Partial<Settings>) => Promise<boolean>
  fetchMCPConfig: () => Promise<void>
  updateMCPConfig: (config: MCPSettings) => Promise<boolean>
  testMCPConnection: (serverId: string) => Promise<{ success: boolean; latency?: number }>
  clearError: () => void
}

const DEFAULT_SETTINGS: Settings = {
  theme: 'system',
  refresh_interval: 5,
  notifications: {
    goal_completion: true,
    agent_status: true,
    errors: true,
    webhook_events: false
  },
  api: {
    endpoint: 'http://localhost:8000',
    websocket: 'ws://localhost:8000/ws',
    timeout: 30
  },
  features: {
    auto_refresh: true,
    confirm_destructive: true,
    show_telemetry: true
  }
}

export const useSettingsStore = create<SettingsState>((set, get) => ({
  settings: null,
  mcpConfig: null,
  isLoading: false,
  saving: false,
  error: null,

  fetchSettings: async () => {
    set({ isLoading: true, error: null })
    try {
      const response = await fetch('/api/settings')
      if (!response.ok) throw new Error('Failed to fetch settings')
      const settings = await response.json()
      set({ settings, isLoading: false })
    } catch (error) {
      console.error('Failed to fetch settings:', error)
      set({ settings: DEFAULT_SETTINGS, isLoading: false })
    }
  },

  updateSettings: async (newSettings: Partial<Settings>) => {
    set({ saving: true, error: null })
    try {
      const currentSettings = get().settings || DEFAULT_SETTINGS
      const mergedSettings = { ...currentSettings, ...newSettings }
      
      const response = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mergedSettings)
      })
      
      if (!response.ok) throw new Error('Failed to update settings')
      
      set({ settings: mergedSettings, saving: false })
      return true
    } catch (error) {
      set({ error: (error as Error).message, saving: false })
      return false
    }
  },

  fetchMCPConfig: async () => {
    set({ isLoading: true, error: null })
    try {
      const response = await fetch('/api/settings/mcp')
      if (!response.ok) throw new Error('Failed to fetch MCP config')
      const mcpConfig = await response.json()
      set({ mcpConfig, isLoading: false })
    } catch (error) {
      console.error('Failed to fetch MCP config:', error)
      set({ mcpConfig: { mcpServers: {} }, isLoading: false })
    }
  },

  updateMCPConfig: async (config: MCPSettings) => {
    set({ saving: true, error: null })
    try {
      const response = await fetch('/api/settings/mcp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      })
      
      if (!response.ok) throw new Error('Failed to update MCP config')
      
      set({ mcpConfig: config, saving: false })
      return true
    } catch (error) {
      set({ error: (error as Error).message, saving: false })
      return false
    }
  },

  testMCPConnection: async (serverId: string) => {
    try {
      const response = await fetch(`/api/settings/mcp/test?server_id=${serverId}`, {
        method: 'POST'
      })
      
      if (!response.ok) return { success: false }
      
      const result = await response.json()
      return { success: result.status === 'connected', latency: result.latency_ms }
    } catch (error) {
      return { success: false }
    }
  },

  clearError: () => {
    set({ error: null })
  }
}))
