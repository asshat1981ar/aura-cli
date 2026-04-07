import { useEffect, useMemo, useRef, useState } from 'react'
import { 
  Play, 
  Pause, 
  ZoomIn, 
  ZoomOut, 
  Maximize, 
  Settings,
  Info
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useWorkflowStore, WorkflowNode, Connection } from '@/stores/workflowStore'

const NODE_COLORS: Record<string, string> = {
  'n8n-nodes-base.httpRequest': '#7c3aed',
  'n8n-nodes-base.executeWorkflow': '#0ea5e9',
  'n8n-nodes-base.function': '#8b5cf6',
  'n8n-nodes-base.code': '#ec4899',
  'n8n-nodes-base.wait': '#f59e0b',
  'n8n-nodes-base.webhook': '#10b981',
  'n8n-nodes-base.scheduleTrigger': '#6366f1',
  'n8n-nodes-base.manualTrigger': '#ef4444',
  'n8n-nodes-base.agent': '#06b6d4',
  'n8n-nodes-base.lmChat': '#a855f7',
  'n8n-nodes-base.chatTrigger': '#22c55e',
  'default': '#64748b'
}

const NODE_ICONS: Record<string, string> = {
  'n8n-nodes-base.httpRequest': 'HTTP',
  'n8n-nodes-base.executeWorkflow': 'WF',
  'n8n-nodes-base.function': 'Fn',
  'n8n-nodes-base.code': '{}',
  'n8n-nodes-base.wait': '⏱',
  'n8n-nodes-base.webhook': '⚡',
  'n8n-nodes-base.scheduleTrigger': '⏰',
  'n8n-nodes-base.manualTrigger': '▶',
  'n8n-nodes-base.agent': '🤖',
  'n8n-nodes-base.lmChat': '💬',
  'n8n-nodes-base.chatTrigger': '💭',
  'default': '○'
}

function getNodeColor(type: string): string {
  return NODE_COLORS[type] || NODE_COLORS.default
}

function getNodeIcon(type: string): string {
  return NODE_ICONS[type] || NODE_ICONS.default
}

interface PositionedNode extends WorkflowNode {
  x: number
  y: number
}

export function WorkflowVisualizer() {
  const { 
    workflows, 
    selectedWorkflow, 
    selectedNode,
    fetchWorkflows, 
    fetchWorkflow,
    selectNode,
    executeWorkflow,
    activateWorkflow,
    deactivateWorkflow
  } = useWorkflowStore()
  
  const [scale, setScale] = useState(1)
  const [translate, setTranslate] = useState({ x: 50, y: 50 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const svgRef = useRef<SVGSVGElement>(null)
  
  useEffect(() => {
    fetchWorkflows()
  }, [])

  // Auto-select first workflow if none selected
  useEffect(() => {
    if (workflows.length > 0 && !selectedWorkflow) {
      fetchWorkflow(workflows[0].id)
    }
  }, [workflows, selectedWorkflow])

  // Calculate node positions
  const positionedNodes = useMemo<PositionedNode[]>(() => {
    if (!selectedWorkflow?.nodes) return []
    
    return selectedWorkflow.nodes.map((node) => ({
      ...node,
      x: node.position?.[0] || 0,
      y: node.position?.[1] || 0
    }))
  }, [selectedWorkflow])

  // Build connections from workflow data
  const connections = useMemo(() => {
    if (!selectedWorkflow?.connections) return []
    
    const conns: { from: PositionedNode; to: PositionedNode }[] = []
    const nodeMap = new Map(positionedNodes.map(n => [n.id, n]))
    
    Object.entries(selectedWorkflow.connections).forEach(([fromId, connData]) => {
      const fromNode = nodeMap.get(fromId)
      if (!fromNode || !connData?.main) return
      
      connData.main.forEach((slot) => {
        slot?.forEach((target: Connection) => {
          const toNode = nodeMap.get(target.node)
          if (toNode) {
            conns.push({ from: fromNode, to: toNode })
          }
        })
      })
    })
    
    return conns
  }, [selectedWorkflow, positionedNodes])

  // Calculate bounding box for auto-fit
  const boundingBox = useMemo(() => {
    if (positionedNodes.length === 0) return { minX: 0, minY: 0, maxX: 800, maxY: 600 }
    
    const minX = Math.min(...positionedNodes.map(n => n.x))
    const maxX = Math.max(...positionedNodes.map(n => n.x))
    const minY = Math.min(...positionedNodes.map(n => n.y))
    const maxY = Math.max(...positionedNodes.map(n => n.y))
    
    return { minX, minY, maxX, maxY }
  }, [positionedNodes])

  const handleZoomIn = () => setScale(s => Math.min(s * 1.2, 3))
  const handleZoomOut = () => setScale(s => Math.max(s / 1.2, 0.3))
  
  const handleFitToScreen = () => {
    if (!svgRef.current || positionedNodes.length === 0) return
    
    const svg = svgRef.current
    const rect = svg.getBoundingClientRect()
    const width = boundingBox.maxX - boundingBox.minX + 300
    const height = boundingBox.maxY - boundingBox.minY + 200
    
    const scaleX = rect.width / width
    const scaleY = rect.height / height
    const newScale = Math.min(scaleX, scaleY, 1) * 0.9
    
    setScale(newScale)
    setTranslate({
      x: (rect.width - width * newScale) / 2 - boundingBox.minX * newScale + 150,
      y: (rect.height - height * newScale) / 2 - boundingBox.minY * newScale + 100
    })
  }

  const handleSvgMouseDown = (e: React.MouseEvent) => {
    if (e.target === svgRef.current) {
      setIsDragging(true)
      setDragStart({ x: e.clientX - translate.x, y: e.clientY - translate.y })
    }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setTranslate({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      })
    }
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? 0.9 : 1.1
    setScale(s => Math.max(0.3, Math.min(3, s * delta)))
  }

  const handleExecute = async () => {
    if (!selectedWorkflow) return
    await executeWorkflow(selectedWorkflow.id)
  }

  const handleToggleActive = async () => {
    if (!selectedWorkflow) return
    if (selectedWorkflow.active) {
      await deactivateWorkflow(selectedWorkflow.id)
    } else {
      await activateWorkflow(selectedWorkflow.id)
    }
  }

  // Generate SVG path for connection
  const getConnectionPath = (from: PositionedNode, to: PositionedNode): string => {
    const fromX = from.x + 100
    const fromY = from.y + 30
    const toX = to.x
    const toY = to.y + 30
    
    const midX = (fromX + toX) / 2
    return `M ${fromX} ${fromY} C ${midX} ${fromY}, ${midX} ${toY}, ${toX} ${toY}`
  }

  if (!selectedWorkflow) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        <div className="text-center">
          <Settings className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>Select a workflow to visualize</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center justify-between p-2 border-b bg-card">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-sm truncate max-w-[200px]">
            {selectedWorkflow.name || selectedWorkflow.id}
          </span>
          <span className={cn(
            "text-xs px-2 py-0.5 rounded-full",
            selectedWorkflow.active ? "bg-green-100 text-green-700" : "bg-gray-100 text-gray-600"
          )}>
            {selectedWorkflow.active ? 'Active' : 'Inactive'}
          </span>
        </div>
        
        <div className="flex items-center gap-1">
          <button 
            className="p-2 rounded hover:bg-muted transition-colors"
            onClick={handleZoomOut}
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="text-xs w-12 text-center">{Math.round(scale * 100)}%</span>
          <button 
            className="p-2 rounded hover:bg-muted transition-colors"
            onClick={handleZoomIn}
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button 
            className="p-2 rounded hover:bg-muted transition-colors"
            onClick={handleFitToScreen}
          >
            <Maximize className="w-4 h-4" />
          </button>
          <div className="w-px h-6 bg-border mx-2" />
          <button 
            className={cn(
              "px-3 py-1.5 rounded text-sm font-medium flex items-center gap-1 transition-colors",
              selectedWorkflow.active 
                ? "bg-destructive text-destructive-foreground hover:bg-destructive/90" 
                : "bg-primary text-primary-foreground hover:bg-primary/90"
            )}
            onClick={handleToggleActive}
          >
            {selectedWorkflow.active ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {selectedWorkflow.active ? 'Deactivate' : 'Activate'}
          </button>
          <button 
            className="px-3 py-1.5 rounded text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 flex items-center gap-1 transition-colors"
            onClick={handleExecute}
          >
            <Play className="w-4 h-4" />
            Execute
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div className="flex-1 overflow-hidden bg-muted/30 relative">
        <svg
          ref={svgRef}
          className="w-full h-full cursor-grab active:cursor-grabbing"
          onMouseDown={handleSvgMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
        >
          <g transform={`translate(${translate.x}, ${translate.y}) scale(${scale})`}>
            {/* Grid pattern */}
            <defs>
              <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#e2e8f0" strokeWidth="0.5"/>
              </pattern>
            </defs>
            <rect x="-5000" y="-5000" width="10000" height="10000" fill="url(#grid)" />
            
            {/* Connections */}
            {connections.map((conn, idx) => (
              <g key={`conn-${idx}`}>
                <path
                  d={getConnectionPath(conn.from, conn.to)}
                  fill="none"
                  stroke="#94a3b8"
                  strokeWidth="2"
                  className="hover:stroke-primary transition-colors"
                />
                <path
                  d={getConnectionPath(conn.from, conn.to)}
                  fill="none"
                  stroke="transparent"
                  strokeWidth="10"
                  className="cursor-pointer"
                />
              </g>
            ))}
            
            {/* Nodes */}
            {positionedNodes.map((node) => (
              <g
                key={node.id}
                transform={`translate(${node.x}, ${node.y})`}
                className="cursor-pointer"
                onClick={() => selectNode(node)}
              >
                {/* Node shadow */}
                <rect
                  x="2"
                  y="2"
                  width="100"
                  height="60"
                  rx="6"
                  fill="rgba(0,0,0,0.1)"
                />
                
                {/* Node body */}
                <rect
                  x="0"
                  y="0"
                  width="100"
                  height="60"
                  rx="6"
                  fill="white"
                  stroke={selectedNode?.id === node.id ? getNodeColor(node.type) : '#e2e8f0'}
                  strokeWidth={selectedNode?.id === node.id ? 3 : 1}
                  className="transition-all"
                />
                
                {/* Node color bar */}
                <rect
                  x="0"
                  y="0"
                  width="6"
                  height="60"
                  rx="6"
                  fill={getNodeColor(node.type)}
                />
                
                {/* Node icon */}
                <text
                  x="20"
                  y="22"
                  fontSize="14"
                  fontWeight="bold"
                  fill={getNodeColor(node.type)}
                >
                  {getNodeIcon(node.type)}
                </text>
                
                {/* Node name */}
                <text
                  x="20"
                  y="42"
                  fontSize="11"
                  fontWeight="500"
                  fill="#1e293b"
                  className="select-none"
                >
                  {node.name.length > 10 ? node.name.slice(0, 10) + '...' : node.name}
                </text>
                
                {/* Input dot */}
                <circle cx="0" cy="30" r="4" fill="#cbd5e1" stroke="white" strokeWidth="2" />
                
                {/* Output dot */}
                <circle cx="100" cy="30" r="4" fill="#cbd5e1" stroke="white" strokeWidth="2" />
              </g>
            ))}
          </g>
        </svg>
        
        {/* Node details panel */}
        {selectedNode && (
          <div className="absolute right-4 top-4 w-64 bg-card border rounded-lg shadow-lg p-4">
            <div className="flex items-start justify-between mb-3">
              <div>
                <h4 className="font-semibold text-sm">{selectedNode.name}</h4>
                <p className="text-xs text-muted-foreground">{selectedNode.type}</p>
              </div>
              <button 
                className="h-6 w-6 rounded hover:bg-muted flex items-center justify-center transition-colors"
                onClick={() => selectNode(null)}
              >
                ×
              </button>
            </div>
            
            <div className="space-y-2">
              <div className="text-xs">
                <span className="text-muted-foreground">ID:</span>{' '}
                <span className="font-mono">{selectedNode.id}</span>
              </div>
              <div className="text-xs">
                <span className="text-muted-foreground">Type:</span>{' '}
                {selectedNode.type}
              </div>
              <div className="text-xs">
                <span className="text-muted-foreground">Version:</span>{' '}
                {selectedNode.typeVersion}
              </div>
              <div className="text-xs">
                <span className="text-muted-foreground">Position:</span>{' '}
                ({selectedNode.position?.[0] || 0}, {selectedNode.position?.[1] || 0})
              </div>
              
              {selectedNode.parameters && Object.keys(selectedNode.parameters).length > 0 && (
                <div className="mt-3 pt-3 border-t">
                  <h5 className="text-xs font-medium mb-2 flex items-center gap-1">
                    <Info className="w-3 h-3" /> Parameters
                  </h5>
                  <div className="space-y-1">
                    {Object.entries(selectedNode.parameters).slice(0, 5).map(([key, value]) => (
                      <div key={key} className="text-xs">
                        <span className="text-muted-foreground">{key}:</span>{' '}
                        <span className="font-mono truncate max-w-[120px] inline-block align-bottom">
                          {typeof value === 'string' ? value : JSON.stringify(value).slice(0, 30)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
      
      {/* Footer stats */}
      <div className="px-4 py-2 border-t bg-card text-xs text-muted-foreground flex items-center justify-between">
        <div className="flex items-center gap-4">
          <span>{selectedWorkflow.nodes?.length || 0} nodes</span>
          <span>{connections.length} connections</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green-500"></span>
          <span>Zoom: {Math.round(scale * 100)}%</span>
        </div>
      </div>
    </div>
  )
}
