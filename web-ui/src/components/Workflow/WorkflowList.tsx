import { useEffect } from 'react'
import { Play, Pause, FileJson, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useWorkflowStore } from '@/stores/workflowStore'

export function WorkflowList() {
  const { 
    workflows, 
    selectedWorkflow, 
    isLoading,
    fetchWorkflows,
    fetchWorkflow,
    activateWorkflow,
    deactivateWorkflow
  } = useWorkflowStore()

  useEffect(() => {
    fetchWorkflows()
  }, [fetchWorkflows])

  const handleToggleActive = async (e: React.MouseEvent, id: string, active: boolean) => {
    e.stopPropagation()
    if (active) {
      await deactivateWorkflow(id)
    } else {
      await activateWorkflow(id)
    }
  }

  if (isLoading) {
    return (
      <div className="p-4 space-y-3">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-16 bg-muted rounded-lg animate-pulse" />
        ))}
      </div>
    )
  }

  if (workflows.length === 0) {
    return (
      <div className="p-4 text-center text-muted-foreground">
        <FileJson className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p className="text-sm">No workflows found</p>
      </div>
    )
  }

  return (
    <div className="divide-y">
      {workflows.map((workflow) => (
        <div
          key={workflow.id}
          onClick={() => fetchWorkflow(workflow.id)}
          className={cn(
            "p-3 cursor-pointer hover:bg-accent transition-colors group",
            selectedWorkflow?.id === workflow.id && "bg-accent"
          )}
        >
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className={cn(
                  "w-2 h-2 rounded-full",
                  workflow.active ? "bg-green-500" : "bg-gray-300"
                )} />
                <h3 className="font-medium text-sm truncate">
                  {workflow.name}
                </h3>
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {workflow.nodes} nodes • {workflow.id}
              </p>
              {workflow.description && (
                <p className="text-xs text-muted-foreground mt-1 truncate">
                  {workflow.description}
                </p>
              )}
              {workflow.tags && workflow.tags.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-2">
                  {workflow.tags.slice(0, 3).map((tag: string) => (
                    <span 
                      key={tag} 
                      className="text-[10px] px-1.5 py-0.5 bg-muted rounded-full"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
            
            <button
              className="p-1.5 rounded hover:bg-muted opacity-0 group-hover:opacity-100 transition-opacity"
              onClick={(e) => handleToggleActive(e, workflow.id, workflow.active)}
            >
              {workflow.active ? (
                <Pause className="w-4 h-4" />
              ) : (
                <Play className="w-4 h-4" />
              )}
            </button>
            
            <ChevronRight className={cn(
              "w-4 h-4 text-muted-foreground transition-transform mt-1",
              selectedWorkflow?.id === workflow.id && "rotate-90"
            )} />
          </div>
        </div>
      ))}
    </div>
  )
}
