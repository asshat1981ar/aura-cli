import { describe, it, expect, vi, beforeEach } from 'vitest'
import { useWorkflowStore } from '../workflowStore'

// Mock fetch
global.fetch = vi.fn()

describe('workflowStore', () => {
  beforeEach(() => {
    useWorkflowStore.setState({
      workflows: [],
      selectedWorkflow: null,
      executions: [],
      isLoading: false,
      error: null,
      selectedNode: null,
    })
    vi.clearAllMocks()
  })

  describe('fetchWorkflows', () => {
    it('should fetch and set workflows', async () => {
      const mockWorkflows = [
        {
          id: 'wf-1',
          name: 'Test Workflow',
          nodes: 5,
          active: true,
          created_at: '2024-01-01',
        },
      ]

      ;(fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockWorkflows,
      })

      await useWorkflowStore.getState().fetchWorkflows()

      expect(useWorkflowStore.getState().workflows).toEqual(mockWorkflows)
      expect(useWorkflowStore.getState().isLoading).toBe(false)
      expect(useWorkflowStore.getState().error).toBeNull()
    })

    it('should handle fetch error', async () => {
      ;(fetch as any).mockResolvedValueOnce({
        ok: false,
        statusText: 'Not Found',
      })

      await useWorkflowStore.getState().fetchWorkflows()

      expect(useWorkflowStore.getState().error).toBe('Failed to fetch workflows')
      expect(useWorkflowStore.getState().isLoading).toBe(false)
    })
  })

  describe('selectWorkflow', () => {
    it('should select a workflow', () => {
      const workflow = {
        id: 'wf-1',
        name: 'Test',
        nodes: [],
        connections: {},
        active: true,
        created_at: '2024-01-01',
      }

      useWorkflowStore.getState().selectWorkflow(workflow as any)

      expect(useWorkflowStore.getState().selectedWorkflow).toEqual(workflow)
      expect(useWorkflowStore.getState().selectedNode).toBeNull()
    })
  })

  describe('clearError', () => {
    it('should clear error state', () => {
      useWorkflowStore.setState({ error: 'Some error' })

      useWorkflowStore.getState().clearError()

      expect(useWorkflowStore.getState().error).toBeNull()
    })
  })
})
