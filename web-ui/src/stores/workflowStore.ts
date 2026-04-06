import { create } from 'zustand'

export interface WorkflowNode {
  id: string
  name: string
  type: string
  typeVersion: number
  position: [number, number]
  parameters?: Record<string, any>
  credentials?: Record<string, any>
}

export interface Connection {
  node: string
  type: string
  index: number
}

export interface WorkflowListItem {
  id: string
  name: string
  description?: string
  nodes: number
  active: boolean
  created_at: string
  updated_at?: string
  tags?: string[]
  meta?: {
    templateName?: string
    templateDescription?: string
    templateTags?: string
    instanceCreatedAt?: string
  }
}

export interface Workflow {
  id: string
  name: string
  description?: string
  nodes: WorkflowNode[]
  connections: Record<string, { main: Connection[][] }>
  active: boolean
  created_at: string
  updated_at?: string
  tags?: string[]
  meta?: {
    templateName?: string
    templateDescription?: string
    templateTags?: string
    instanceCreatedAt?: string
  }
}

export interface WorkflowExecution {
  id: string
  workflow_id: string
  status: 'running' | 'completed' | 'failed' | 'waiting'
  started_at: string
  completed_at?: string
  data?: Record<string, any>
}

interface WorkflowState {
  workflows: WorkflowListItem[]
  selectedWorkflow: Workflow | null
  executions: WorkflowExecution[]
  isLoading: boolean
  error: string | null
  selectedNode: WorkflowNode | null
  
  // Actions
  fetchWorkflows: () => Promise<void>
  selectWorkflow: (workflow: Workflow | null) => void
  fetchWorkflow: (id: string) => Promise<Workflow | null>
  executeWorkflow: (id: string, data?: Record<string, any>) => Promise<string | null>
  activateWorkflow: (id: string) => Promise<boolean>
  deactivateWorkflow: (id: string) => Promise<boolean>
  selectNode: (node: WorkflowNode | null) => void
  clearError: () => void
}

export const useWorkflowStore = create<WorkflowState>((set) => ({
  workflows: [],
  selectedWorkflow: null,
  executions: [],
  isLoading: false,
  error: null,
  selectedNode: null,

  fetchWorkflows: async () => {
    set({ isLoading: true, error: null })
    try {
      const response = await fetch('/api/workflows')
      if (!response.ok) {
        throw new Error('Failed to fetch workflows')
      }
      const workflows = await response.json()
      set({ workflows, isLoading: false })
    } catch (error) {
      set({ error: (error as Error).message, isLoading: false })
    }
  },

  selectWorkflow: (workflow) => {
    set({ selectedWorkflow: workflow, selectedNode: null })
  },

  fetchWorkflow: async (id) => {
    set({ isLoading: true, error: null })
    try {
      const response = await fetch(`/api/workflows/${id}`)
      if (!response.ok) {
        throw new Error('Failed to fetch workflow')
      }
      const workflow = await response.json()
      set({ selectedWorkflow: workflow, isLoading: false })
      return workflow
    } catch (error) {
      set({ error: (error as Error).message, isLoading: false })
      return null
    }
  },

  executeWorkflow: async (id, data) => {
    try {
      const response = await fetch(`/api/workflows/${id}/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data || {}),
      })
      if (!response.ok) {
        throw new Error('Failed to execute workflow')
      }
      const result = await response.json()
      
      // Add to executions
      const execution: WorkflowExecution = {
        id: result.execution_id,
        workflow_id: id,
        status: 'running',
        started_at: result.started_at,
        data: data
      }
      set(state => ({ 
        executions: [execution, ...state.executions].slice(0, 50)
      }))
      
      return result.execution_id
    } catch (error) {
      set({ error: (error as Error).message })
      return null
    }
  },

  activateWorkflow: async (id) => {
    try {
      const response = await fetch(`/api/workflows/${id}/activate`, {
        method: 'POST',
      })
      if (!response.ok) return false
      
      // Update local state
      set(state => ({
        workflows: state.workflows.map(w => 
          w.id === id ? { ...w, active: true } : w
        )
      }))
      return true
    } catch (error) {
      set({ error: (error as Error).message })
      return false
    }
  },

  deactivateWorkflow: async (id) => {
    try {
      const response = await fetch(`/api/workflows/${id}/deactivate`, {
        method: 'POST',
      })
      if (!response.ok) return false
      
      // Update local state
      set(state => ({
        workflows: state.workflows.map(w => 
          w.id === id ? { ...w, active: false } : w
        )
      }))
      return true
    } catch (error) {
      set({ error: (error as Error).message })
      return false
    }
  },

  selectNode: (node) => {
    set({ selectedNode: node })
  },

  clearError: () => {
    set({ error: null })
  },
}))
