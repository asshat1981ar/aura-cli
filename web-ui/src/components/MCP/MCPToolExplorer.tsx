import { useState } from 'react'
import { 
  Wrench, 
  Play, 
  ChevronDown, 
  ChevronRight,
  Terminal,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  Search
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useMCPStore, MCPTool } from '@/stores/mcpStore'

export function MCPToolExplorer() {
  const { 
    selectedServer, 
    tools, 
    executions,
    isLoading,
    selectTool,
    executeTool
  } = useMCPStore()
  
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedTool, setExpandedTool] = useState<string | null>(null)
  const [paramValues, setParamValues] = useState<Record<string, any>>({})
  const [executing, setExecuting] = useState(false)

  if (!selectedServer) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        <div className="text-center">
          <Wrench className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>Select an MCP server to explore tools</p>
        </div>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    )
  }

  const filteredTools = tools.filter(tool => 
    tool.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    tool.description.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const handleExecute = async (tool: MCPTool) => {
    setExecuting(true)
    await executeTool(selectedServer.id, tool.name, paramValues)
    setExecuting(false)
    setParamValues({})
  }

  const getParamInput = (name: string, param: { type: string; description?: string }) => {
    const value = paramValues[name] || ''
    
    if (param.type === 'boolean') {
      return (
        <select
          value={String(value)}
          onChange={(e) => setParamValues({ ...paramValues, [name]: e.target.value === 'true' })}
          className="w-full px-3 py-2 border rounded-md text-sm bg-background"
        >
          <option value="">Select...</option>
          <option value="true">True</option>
          <option value="false">False</option>
        </select>
      )
    }
    
    if (param.type === 'number') {
      return (
        <input
          type="number"
          value={value}
          onChange={(e) => setParamValues({ ...paramValues, [name]: Number(e.target.value) })}
          placeholder={param.description}
          className="w-full px-3 py-2 border rounded-md text-sm bg-background"
        />
      )
    }
    
    if (param.type === 'array' || param.type === 'object') {
      return (
        <textarea
          value={typeof value === 'object' ? JSON.stringify(value) : value}
          onChange={(e) => {
            try {
              const parsed = JSON.parse(e.target.value)
              setParamValues({ ...paramValues, [name]: parsed })
            } catch {
              setParamValues({ ...paramValues, [name]: e.target.value })
            }
          }}
          placeholder={`${param.description || ''} (JSON)`}
          rows={3}
          className="w-full px-3 py-2 border rounded-md text-sm bg-background font-mono text-xs"
        />
      )
    }
    
    return (
      <input
        type="text"
        value={value}
        onChange={(e) => setParamValues({ ...paramValues, [name]: e.target.value })}
        placeholder={param.description}
        className="w-full px-3 py-2 border rounded-md text-sm bg-background"
      />
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="font-semibold">{selectedServer.name}</h2>
            <p className="text-sm text-muted-foreground">{tools.length} tools available</p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs px-2 py-1 bg-green-100 text-green-700 rounded-full">
              {selectedServer.status}
            </span>
          </div>
        </div>
        
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search tools..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2 border rounded-md text-sm bg-background"
          />
        </div>
      </div>

      {/* Tools List */}
      <div className="flex-1 overflow-auto">
        <div className="divide-y">
          {filteredTools.map((tool) => {
            const isExpanded = expandedTool === tool.name
            const paramCount = Object.keys(tool.parameters || {}).length
            const recentExecutions = executions.filter(e => 
              e.tool_name === tool.name && e.server_id === selectedServer.id
            ).slice(0, 3)
            
            return (
              <div key={tool.name} className="border-b last:border-b-0">
                <div
                  onClick={() => {
                    setExpandedTool(isExpanded ? null : tool.name)
                    selectTool(isExpanded ? null : tool)
                  }}
                  className={cn(
                    "p-3 cursor-pointer hover:bg-accent/50 transition-colors",
                    isExpanded && "bg-accent"
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {isExpanded ? (
                        <ChevronDown className="w-4 h-4 text-muted-foreground" />
                      ) : (
                        <ChevronRight className="w-4 h-4 text-muted-foreground" />
                      )}
                      <Wrench className="w-4 h-4 text-primary" />
                      <span className="font-medium text-sm">{tool.name}</span>
                      {paramCount > 0 && (
                        <span className="text-xs text-muted-foreground">
                          ({paramCount} params)
                        </span>
                      )}
                    </div>
                    {recentExecutions.length > 0 && (
                      <div className="flex items-center gap-1">
                        {recentExecutions[0].status === 'success' ? (
                          <CheckCircle2 className="w-4 h-4 text-green-500" />
                        ) : recentExecutions[0].status === 'error' ? (
                          <XCircle className="w-4 h-4 text-red-500" />
                        ) : (
                          <Loader2 className="w-4 h-4 animate-spin text-primary" />
                        )}
                        <span className="text-xs text-muted-foreground">
                          {recentExecutions.length}
                        </span>
                      </div>
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1 ml-6">
                    {tool.description}
                  </p>
                </div>

                {/* Expanded Tool Form */}
                {isExpanded && (
                  <div className="px-4 pb-4 bg-accent/30">
                    {paramCount > 0 ? (
                      <div className="space-y-3 mb-4">
                        <h4 className="text-xs font-medium text-muted-foreground uppercase">
                          Parameters
                        </h4>
                        {Object.entries(tool.parameters || {}).map(([name, param]) => (
                          <div key={name}>
                            <label className="block text-xs font-medium mb-1">
                              {name}
                              <span className="text-muted-foreground ml-1">({param.type})</span>
                            </label>
                            {getParamInput(name, param)}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-xs text-muted-foreground mb-4">
                        No parameters required
                      </p>
                    )}
                    
                    <button
                      onClick={() => handleExecute(tool)}
                      disabled={executing}
                      className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {executing ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Play className="w-4 h-4" />
                      )}
                      Execute
                    </button>
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {filteredTools.length === 0 && (
          <div className="p-8 text-center text-muted-foreground">
            <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No tools match your search</p>
          </div>
        )}
      </div>

      {/* Recent Executions */}
      {executions.length > 0 && (
        <div className="border-t max-h-48 overflow-auto">
          <div className="p-3 border-b bg-muted/50">
            <h4 className="text-xs font-medium flex items-center gap-2">
              <Terminal className="w-3 h-3" />
              Recent Executions
            </h4>
          </div>
          <div className="divide-y">
            {executions.slice(0, 10).map((execution) => (
              <div key={execution.id} className="px-3 py-2 text-xs">
                <div className="flex items-center justify-between">
                  <span className="font-medium">{execution.tool_name}</span>
                  <div className="flex items-center gap-2">
                    {execution.status === 'success' ? (
                      <CheckCircle2 className="w-3 h-3 text-green-500" />
                    ) : execution.status === 'error' ? (
                      <XCircle className="w-3 h-3 text-red-500" />
                    ) : (
                      <Loader2 className="w-3 h-3 animate-spin text-primary" />
                    )}
                    <Clock className="w-3 h-3 text-muted-foreground" />
                    <span className="text-muted-foreground">
                      {new Date(execution.started_at).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
                {execution.result && execution.status !== 'running' && (
                  <pre className="mt-1 p-2 bg-muted rounded text-[10px] overflow-x-auto max-h-20">
                    {JSON.stringify(execution.result, null, 2)}
                  </pre>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
