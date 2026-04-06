import { MCPServerList, MCPToolExplorer } from '@/components/MCP'
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable'

export function MCP() {
  return (
    <div className="h-[calc(100vh-8rem)]">
      <ResizablePanelGroup direction="horizontal">
        {/* MCP Servers Panel */}
        <ResizablePanel defaultSize={30} minSize={25} maxSize={40}>
          <div className="h-full flex flex-col border-r">
            <div className="p-4 border-b">
              <h2 className="text-lg font-semibold">MCP Servers</h2>
              <p className="text-sm text-muted-foreground">
                Manage Model Context Protocol servers
              </p>
            </div>
            <div className="flex-1 overflow-auto">
              <MCPServerList />
            </div>
          </div>
        </ResizablePanel>
        
        <ResizableHandle />
        
        {/* Tool Explorer Panel */}
        <ResizablePanel defaultSize={35} minSize={30} maxSize={50}>
          <div className="h-full">
            <MCPToolExplorer />
          </div>
        </ResizablePanel>
        
        <ResizableHandle />
        
        {/* Results/Output Panel */}
        <ResizablePanel defaultSize={35} minSize={25}>
          <MCPToolResults />
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  )
}

import { useMCPStore } from '@/stores/mcpStore'
import { Terminal, Play, Clock, CheckCircle2, XCircle, RotateCcw } from 'lucide-react'

function MCPToolResults() {
  const { executions, selectedServer, selectedTool } = useMCPStore()
  
  const filteredExecutions = executions.filter(e => {
    if (selectedTool && e.tool_name !== selectedTool.name) return false
    if (selectedServer && e.server_id !== selectedServer.id) return false
    return true
  })

  return (
    <div className="h-full flex flex-col border-l">
      <div className="p-4 border-b">
        <h2 className="text-lg font-semibold">Execution Results</h2>
        <p className="text-sm text-muted-foreground">
          {selectedTool ? `Showing results for ${selectedTool.name}` : 'All executions'}
        </p>
      </div>
      
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {filteredExecutions.length === 0 ? (
          <div className="text-center text-muted-foreground py-8">
            <Terminal className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No executions yet</p>
            <p className="text-sm mt-1">Select a tool and execute it to see results</p>
          </div>
        ) : (
          filteredExecutions.map((execution) => (
            <div 
              key={execution.id} 
              className="border rounded-lg p-4 bg-card"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Play className="w-4 h-4 text-primary" />
                  <span className="font-medium text-sm">{execution.tool_name}</span>
                  <span className="text-xs text-muted-foreground">
                    ({execution.server_id})
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {execution.status === 'success' ? (
                    <CheckCircle2 className="w-4 h-4 text-green-500" />
                  ) : execution.status === 'error' ? (
                    <XCircle className="w-4 h-4 text-red-500" />
                  ) : (
                    <RotateCcw className="w-4 h-4 animate-spin text-primary" />
                  )}
                  <span className={`
                    text-xs px-2 py-0.5 rounded-full
                    ${execution.status === 'success' ? 'bg-green-100 text-green-700' : ''}
                    ${execution.status === 'error' ? 'bg-red-100 text-red-700' : ''}
                    ${execution.status === 'running' ? 'bg-blue-100 text-blue-700' : ''}
                  `}>
                    {execution.status}
                  </span>
                </div>
              </div>
              
              <div className="flex items-center gap-2 text-xs text-muted-foreground mb-3">
                <Clock className="w-3 h-3" />
                {new Date(execution.started_at).toLocaleString()}
                {execution.completed_at && (
                  <>
                    <span>•</span>
                    <span>
                      {Math.round(
                        (new Date(execution.completed_at).getTime() - 
                         new Date(execution.started_at).getTime()) / 1000
                      )}s
                    </span>
                  </>
                )}
              </div>
              
              {execution.params && Object.keys(execution.params).length > 0 && (
                <div className="mb-3">
                  <h4 className="text-xs font-medium text-muted-foreground mb-1">Parameters</h4>
                  <pre className="p-2 bg-muted rounded text-xs overflow-x-auto">
                    {JSON.stringify(execution.params, null, 2)}
                  </pre>
                </div>
              )}
              
              {execution.result && (
                <div>
                  <h4 className="text-xs font-medium text-muted-foreground mb-1">Result</h4>
                  <pre className={`
                    p-2 rounded text-xs overflow-x-auto max-h-48
                    ${execution.status === 'error' ? 'bg-red-50 text-red-900' : 'bg-muted'}
                  `}>
                    {typeof execution.result === 'object' 
                      ? JSON.stringify(execution.result, null, 2)
                      : String(execution.result)
                    }
                  </pre>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  )
}
