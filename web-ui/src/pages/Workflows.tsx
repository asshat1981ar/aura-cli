import { WorkflowVisualizer, WorkflowList } from '@/components/Workflow'
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable'

export function Workflows() {
  return (
    <div className="h-[calc(100vh-8rem)]">
      <ResizablePanelGroup direction="horizontal">
        {/* Workflow List Panel */}
        <ResizablePanel defaultSize={25} minSize={20} maxSize={40}>
          <div className="h-full flex flex-col border-r">
            <div className="p-4 border-b">
              <h2 className="text-lg font-semibold">n8n Workflows</h2>
              <p className="text-sm text-muted-foreground">
                Manage and execute workflows
              </p>
            </div>
            <div className="flex-1 overflow-auto">
              <WorkflowList />
            </div>
          </div>
        </ResizablePanel>
        
        <ResizableHandle />
        
        {/* Visualizer Panel */}
        <ResizablePanel defaultSize={75}>
          <div className="h-full">
            <WorkflowVisualizer />
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  )
}
