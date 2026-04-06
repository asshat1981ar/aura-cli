import { create } from 'zustand'

export interface N8NWorkflow {
  id: string
  name: string
  description?: string
  nodes: N8NNode[]
  connections: Record<string, Connection[]>
  settings?: Record<string, any>
}

export interface N8NNode {
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

export interface WorkflowExecution {
  id: string
  workflowId: string
  status: 'running' | 'completed' | 'failed' | 'waiting'
  startedAt: string
  finishedAt?: string
  data?: Record<string, any>
}

interface N8NState {
  workflows: N8NWorkflow[]
  executions: WorkflowExecution[]
  activeWorkflowId: string | null
  isLoading: boolean
  error: string | null
  
  // Actions
  fetchWorkflows: () => Promise<void>
  fetchExecutions: (workflowId?: string) => Promise<void>
  triggerWorkflow: (workflowId: string, data?: any) => Promise<void>
  setActiveWorkflow: (id: string | null) => void
  getWorkflow: (id: string) => N8NWorkflow | undefined
}

export const useN8NStore = create<N8NState>((set, get) => ({
  workflows: [],
  executions: [],
  activeWorkflowId: null,
  isLoading: false,
  error: null,

  fetchWorkflows: async () => {
    set({ isLoading: true, error: null })
    try {
      const response = await fetch('/api/n8n/workflows')
      if (!response.ok) throw new Error('Failed to fetch workflows')
      const workflows = await response.json()
      set({ workflows, isLoading: false })
    } catch (error) {
      // Fallback to local files if API fails
      try {
        const localWorkflows = await loadLocalWorkflows()
        set({ workflows: localWorkflows, isLoading: false })
      } catch {
        set({ error: (error as Error).message, isLoading: false })
      }
    }
  },

  fetchExecutions: async (workflowId) => {
    try {
      const url = workflowId 
        ? `/api/n8n/executions?workflowId=${workflowId}`
        : '/api/n8n/executions'
      const response = await fetch(url)
      if (!response.ok) throw new Error('Failed to fetch executions')
      const executions = await response.json()
      set({ executions })
    } catch (error) {
      console.error('Failed to fetch executions:', error)
    }
  },

  triggerWorkflow: async (workflowId, data) => {
    try {
      const response = await fetch(`/api/n8n/workflows/${workflowId}/trigger`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data || {}),
      })
      if (!response.ok) throw new Error('Failed to trigger workflow')
      
      // Refresh executions after trigger
      await get().fetchExecutions(workflowId)
    } catch (error) {
      set({ error: (error as Error).message })
    }
  },

  setActiveWorkflow: (id) => set({ activeWorkflowId: id }),

  getWorkflow: (id) => {
    return get().workflows.find((w) => w.id === id)
  },
}))

// Helper to load workflows from local JSON files
async function loadLocalWorkflows(): Promise<N8NWorkflow[]> {
  const workflowFiles = [
    '/WF-0-master-dispatcher.json',
    '/WF-1-bug-fix-handler.json',
    '/WF-2-feature-handler.json',
    '/WF-3-refactor-handler.json',
    '/WF-4-security-handler.json',
    '/WF-5-docs-handler.json',
    '/WF-6-code-gen-pr-push.json',
  ]
  
  const workflows: N8NWorkflow[] = []
  
  for (const file of workflowFiles) {
    try {
      const response = await fetch(`/n8n-workflows${file}`)
      if (response.ok) {
        const workflow = await response.json()
        workflows.push({
          id: workflow.id || file.replace(/\//g, '').replace('.json', ''),
          name: workflow.name || file,
          nodes: workflow.nodes || [],
          connections: workflow.connections || {},
          settings: workflow.settings,
        })
      }
    } catch (error) {
      console.warn(`Failed to load workflow ${file}:`, error)
    }
  }
  
  return workflows
}
