import { useEffect } from 'react'
import { 
  Bot, 
  Pause, 
  Play, 
  RotateCcw, 
  Terminal,
  BarChart3,
  Clock,
  CheckCircle2,
  XCircle,
  Activity,
  Layers
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useAgentStore } from '@/stores/agentStore'

const STATUS_COLORS = {
  idle: 'bg-green-500',
  busy: 'bg-blue-500',
  paused: 'bg-yellow-500',
  error: 'bg-red-500',
  offline: 'bg-gray-400'
}

export function AgentDetail() {
  const { 
    selectedAgent, 
    logs, 
    fetchAgentLogs,
    fetchAgentHistory,
    pauseAgent,
    resumeAgent,
    restartAgent
  } = useAgentStore()

  useEffect(() => {
    if (selectedAgent) {
      fetchAgentLogs(selectedAgent.id)
      fetchAgentHistory(selectedAgent.id)
    }
  }, [selectedAgent])

  if (!selectedAgent) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        <div className="text-center">
          <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>Select an agent to view details</p>
        </div>
      </div>
    )
  }

  const isPaused = selectedAgent.status === 'paused'
  const isBusy = selectedAgent.status === 'busy'

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center">
              <Bot className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h2 className="text-lg font-semibold">{selectedAgent.name}</h2>
              <div className="flex items-center gap-2">
                <span className={cn("w-2 h-2 rounded-full", STATUS_COLORS[selectedAgent.status])} />
                <span className="text-sm text-muted-foreground capitalize">{selectedAgent.status}</span>
                <span className="text-muted-foreground">•</span>
                <span className="text-sm text-muted-foreground">{selectedAgent.type}</span>
              </div>
            </div>
          </div>
          
          {/* Action Buttons */}
          <div className="flex items-center gap-2">
            {isPaused ? (
              <button
                onClick={() => resumeAgent(selectedAgent.id)}
                className="flex items-center gap-2 px-3 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 transition-colors"
              >
                <Play className="w-4 h-4" />
                Resume
              </button>
            ) : (
              <button
                onClick={() => pauseAgent(selectedAgent.id)}
                disabled={isBusy}
                className="flex items-center gap-2 px-3 py-2 bg-yellow-600 text-white rounded-lg text-sm font-medium hover:bg-yellow-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Pause className="w-4 h-4" />
                Pause
              </button>
            )}
            <button
              onClick={() => restartAgent(selectedAgent.id)}
              disabled={isBusy}
              className="flex items-center gap-2 px-3 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <RotateCcw className="w-4 h-4" />
              Restart
            </button>
          </div>
        </div>

        {/* Capabilities */}
        {selectedAgent.capabilities.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-4">
            <Layers className="w-4 h-4 text-muted-foreground mt-0.5" />
            {selectedAgent.capabilities.map((cap) => (
              <span 
                key={cap}
                className="text-xs px-2 py-1 bg-muted rounded-md"
              >
                {cap}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-4 gap-4 p-4 border-b bg-muted/30">
        <MetricCard
          icon={Activity}
          label="Total Executions"
          value={selectedAgent.metrics.total_executions}
        />
        <MetricCard
          icon={CheckCircle2}
          label="Success Rate"
          value={`${selectedAgent.metrics.success_rate}%`}
          color={selectedAgent.metrics.success_rate >= 90 ? 'text-green-600' : 
                 selectedAgent.metrics.success_rate >= 70 ? 'text-yellow-600' : 'text-red-600'}
        />
        <MetricCard
          icon={Clock}
          label="Avg Latency"
          value={`${selectedAgent.metrics.avg_latency_ms}ms`}
        />
        <MetricCard
          icon={BarChart3}
          label="Status"
          value={selectedAgent.status.charAt(0).toUpperCase() + selectedAgent.status.slice(1)}
        />
      </div>

      {/* Tabs Content */}
      <div className="flex-1 overflow-auto">
        <div className="p-4">
          <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
            <Terminal className="w-4 h-4" />
            Recent Logs
          </h3>
          
          {logs.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-8">No logs available</p>
          ) : (
            <div className="space-y-2">
              {logs.slice(0, 20).map((log, idx) => (
                <div 
                  key={idx}
                  className="flex items-start gap-3 p-3 rounded-lg border bg-card text-sm"
                >
                  {log.status === 'success' ? (
                    <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5" />
                  ) : log.status === 'error' ? (
                    <XCircle className="w-4 h-4 text-red-500 mt-0.5" />
                  ) : (
                    <Activity className="w-4 h-4 text-blue-500 mt-0.5" />
                  )}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{log.status}</span>
                      <span className="text-xs text-muted-foreground">
                        {new Date(log.timestamp).toLocaleString()}
                      </span>
                    </div>
                    {log.latency && (
                      <p className="text-xs text-muted-foreground mt-1">
                        Latency: {log.latency}ms
                        {log.tokens && ` • Tokens: ${log.tokens}`}
                      </p>
                    )}
                    {log.message && (
                      <p className="text-xs mt-1 truncate">{log.message}</p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

interface MetricCardProps {
  icon: React.ComponentType<{ className?: string }>
  label: string
  value: string | number
  color?: string
}

function MetricCard({ icon: Icon, label, value, color }: MetricCardProps) {
  return (
    <div className="p-3 rounded-lg border bg-card">
      <div className="flex items-center gap-2 text-muted-foreground mb-1">
        <Icon className="w-4 h-4" />
        <span className="text-xs">{label}</span>
      </div>
      <p className={cn("text-xl font-bold", color)}>{value}</p>
    </div>
  )
}
