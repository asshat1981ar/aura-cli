import { useEffect } from 'react'
import { Bot, RefreshCw, BarChart3 } from 'lucide-react'
import { AgentGrid, AgentDetail } from '@/components/Agent'
import { useAgentStore } from '@/stores/agentStore'
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable'

export function Agents() {
  const { overview, fetchOverview, isLoading } = useAgentStore()

  useEffect(() => {
    fetchOverview()
    const interval = setInterval(fetchOverview, 5000)
    return () => clearInterval(interval)
  }, [fetchOverview])

  // Calculate stats
  const totalAgents = overview.length
  const activeAgents = overview.filter(a => a.status === 'busy').length
  const idleAgents = overview.filter(a => a.status === 'idle').length
  const pausedAgents = overview.filter(a => a.status === 'paused').length
  const avgSuccessRate = totalAgents > 0 
    ? Math.round(overview.reduce((acc, a) => acc + a.metrics.success_rate, 0) / totalAgents)
    : 0

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col">
      {/* Header with Stats */}
      <div className="p-4 border-b bg-card">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
              <Bot className="w-6 h-6" />
              Agent Observatory
            </h2>
            <p className="text-muted-foreground">
              Monitor and manage AURA agents in real-time
            </p>
          </div>
          <button
            onClick={() => fetchOverview()}
            disabled={isLoading}
            className="flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-accent transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        {/* Stats Row */}
        <div className="grid grid-cols-5 gap-4">
          <StatCard label="Total Agents" value={totalAgents} icon={Bot} />
          <StatCard label="Active" value={activeAgents} icon={BarChart3} color="text-blue-600" />
          <StatCard label="Idle" value={idleAgents} icon={Bot} color="text-green-600" />
          <StatCard label="Paused" value={pausedAgents} icon={Bot} color="text-yellow-600" />
          <StatCard label="Avg Success Rate" value={`${avgSuccessRate}%`} icon={BarChart3} color={avgSuccessRate >= 90 ? 'text-green-600' : 'text-orange-600'} />
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        <ResizablePanelGroup direction="horizontal">
          {/* Agent Grid */}
          <ResizablePanel defaultSize={60} minSize={40}>
            <div className="h-full overflow-auto">
              <AgentGrid />
            </div>
          </ResizablePanel>
          
          <ResizableHandle />
          
          {/* Agent Detail */}
          <ResizablePanel defaultSize={40} minSize={30}>
            <div className="h-full border-l">
              <AgentDetail />
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </div>
  )
}

interface StatCardProps {
  label: string
  value: string | number
  icon: React.ComponentType<{ className?: string }>
  color?: string
}

function StatCard({ label, value, icon: Icon, color }: StatCardProps) {
  return (
    <div className="flex items-center gap-3 p-3 rounded-lg border bg-background">
      <Icon className={`w-5 h-5 ${color || 'text-muted-foreground'}`} />
      <div>
        <p className="text-lg font-bold leading-none">{value}</p>
        <p className="text-xs text-muted-foreground mt-1">{label}</p>
      </div>
    </div>
  )
}
