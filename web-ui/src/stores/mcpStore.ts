import { create } from 'zustand'

export interface MCPToolParameter {
  type: string
  description?: string
}

export interface MCPTool {
  name: string
  description: string
  parameters: Record<string, MCPToolParameter>
}

export interface MCPServer {
  id: string
  name: string
  type: 'stdio' | 'http'
  status: 'connected' | 'disconnected' | 'error'
  tools_count: number
  config: {
    command?: string
    args?: string[]
    url?: string
  }
}

export interface MCPToolExecution {
  id: string
  server_id: string
  tool_name: string
  status: 'running' | 'success' | 'error'
  params: Record<string, any>
  result?: any
  started_at: string
  completed_at?: string
}

interface MCPState {
  servers: MCPServer[]
  selectedServer: MCPServer | null
  tools: MCPTool[]
  selectedTool: MCPTool | null
  executions: MCPToolExecution[]
  isLoading: boolean
  error: string | null
  
  // Actions
  fetchServers: () => Promise<void>
  selectServer: (server: MCPServer | null) => void
  fetchTools: (serverId: string) => Promise<void>
  selectTool: (tool: MCPTool | null) => void
  executeTool: (serverId: string, toolName: string, params: Record<string, any>) => Promise<MCPToolExecution | null>
  fetchServerStatus: (serverId: string) => Promise<void>
  clearError: () => void
}

export const useMCPStore = create<MCPState>((set, get) => ({
  servers: [],
  selectedServer: null,
  tools: [],
  selectedTool: null,
  executions: [],
  isLoading: false,
  error: null,

  fetchServers: async () => {
    set({ isLoading: true, error: null })
    try {
      const response = await fetch('/api/mcp/servers')
      if (!response.ok) {
        throw new Error('Failed to fetch MCP servers')
      }
      const servers = await response.json()
      set({ servers, isLoading: false })
    } catch (error) {
      set({ error: (error as Error).message, isLoading: false })
    }
  },

  selectServer: (server) => {
    set({ selectedServer: server, selectedTool: null })
    if (server) {
      get().fetchTools(server.id)
    } else {
      set({ tools: [] })
    }
  },

  fetchTools: async (serverId: string) => {
    set({ isLoading: true, error: null })
    try {
      const response = await fetch(`/api/mcp/servers/${serverId}/tools`)
      if (!response.ok) {
        throw new Error('Failed to fetch tools')
      }
      const data = await response.json()
      set({ tools: data.tools || [], isLoading: false })
    } catch (error) {
      set({ error: (error as Error).message, isLoading: false })
    }
  },

  selectTool: (tool) => {
    set({ selectedTool: tool })
  },

  executeTool: async (serverId: string, toolName: string, params: Record<string, any>) => {
    const executionId = crypto.randomUUID()
    const execution: MCPToolExecution = {
      id: executionId,
      server_id: serverId,
      tool_name: toolName,
      status: 'running',
      params,
      started_at: new Date().toISOString()
    }
    
    set(state => ({
      executions: [execution, ...state.executions].slice(0, 50)
    }))
    
    try {
      const response = await fetch(`/api/mcp/servers/${serverId}/tools/${toolName}/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      })
      
      if (!response.ok) {
        throw new Error('Tool execution failed')
      }
      
      const result = await response.json()
      
      const completedExecution: MCPToolExecution = {
        ...execution,
        status: 'success',
        result: result.result,
        completed_at: new Date().toISOString()
      }
      
      set(state => ({
        executions: [completedExecution, ...state.executions.filter(e => e.id !== executionId)].slice(0, 50)
      }))
      
      return completedExecution
    } catch (error) {
      const failedExecution: MCPToolExecution = {
        ...execution,
        status: 'error',
        result: { error: (error as Error).message },
        completed_at: new Date().toISOString()
      }
      
      set(state => ({
        executions: [failedExecution, ...state.executions.filter(e => e.id !== executionId)].slice(0, 50)
      }))
      
      return failedExecution
    }
  },

  fetchServerStatus: async (serverId: string) => {
    try {
      const response = await fetch(`/api/mcp/servers/${serverId}/status`)
      if (!response.ok) return
      
      const status = await response.json()
      set(state => ({
        servers: state.servers.map(s => 
          s.id === serverId ? { ...s, status: status.status } : s
        )
      }))
    } catch (error) {
      console.error('Failed to fetch server status:', error)
    }
  },

  clearError: () => {
    set({ error: null })
  }
}))
