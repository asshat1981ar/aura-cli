import { useEffect, useRef, useState } from 'react'
import { 
  Trash2, 
  Download, 
  Wifi, 
  WifiOff, 
  RefreshCw,
  Activity,
  BarChart3
} from 'lucide-react'
import { useLogStore, LogEntry } from '../stores/logStore'
import { useTelemetryStore } from '../stores/telemetryStore'
import { useWebSocket } from '../hooks/useWebSocket'
import { SystemHealth, TelemetryMetrics } from '../components/Telemetry'
import { 
  LatencyChart, 
  AgentDistributionChart, 
  RequestVolumeChart,
  SuccessRateChart 
} from '../components/Telemetry/TelemetryCharts'
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '../components/ui/resizable'

const LOG_LEVELS = ['DEBUG', 'INFO', 'WARN', 'ERROR'] as const

export function Logs() {
  const { 
    logs, 
    isConnected, 
    filter, 
    addLog, 
    setConnected, 
    setFilter, 
    clearLogs 
  } = useLogStore()
  const { refreshAll, isLoading } = useTelemetryStore()
  const logsEndRef = useRef<HTMLDivElement>(null)
  const [activeTab, setActiveTab] = useState<'logs' | 'metrics'>('logs')

  const { status: wsStatus, retryCount: wsRetryCount } = useWebSocket('ws://localhost:8000/ws', {
    onMessage: (data: unknown) => {
      const msg = data as { type?: string; payload?: LogEntry }
      if (msg?.type === 'log' && msg.payload) {
        addLog(msg.payload)
      }
    },
    onConnect: () => setConnected(true),
    onDisconnect: () => setConnected(false),
    reconnect: true,
  })

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  useEffect(() => {
    refreshAll()
  }, [refreshAll])

  const filteredLogs = logs.filter((log) => {
    const matchesLevel = !filter.level || log.level === filter.level
    const matchesSearch =
      !filter.search ||
      log.event.toLowerCase().includes(filter.search.toLowerCase()) ||
      log.message?.toLowerCase().includes(filter.search.toLowerCase())
    return matchesLevel && matchesSearch
  })

  const exportLogs = () => {
    const blob = new Blob([JSON.stringify(logs, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `aura-logs-${new Date().toISOString()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col">
      {/* Header */}
      <div className="p-4 border-b bg-card">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
              <Activity className="w-6 h-6" />
              Logs & Telemetry
            </h2>
            <p className="text-muted-foreground">
              Monitor system logs, metrics, and health status
            </p>
          </div>
          <div className="flex items-center gap-2">
            <div
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${
                isConnected
                  ? 'bg-green-100 text-green-800'
                  : wsStatus === 'connecting'
                  ? 'bg-yellow-100 text-yellow-800'
                  : 'bg-red-100 text-red-800'
              }`}
            >
              {isConnected ? (
                <Wifi className="w-4 h-4" />
              ) : (
                <WifiOff className="w-4 h-4" />
              )}
              <span className="text-sm font-medium">
                {isConnected
                  ? 'Live'
                  : wsStatus === 'connecting' && wsRetryCount > 0
                  ? `Reconnecting (attempt ${wsRetryCount})…`
                  : wsStatus === 'connecting'
                  ? 'Connecting…'
                  : 'Disconnected'}
              </span>
            </div>
            <button
              onClick={() => refreshAll()}
              disabled={isLoading}
              className="flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-accent transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mt-4">
          <button
            onClick={() => setActiveTab('logs')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === 'logs'
                ? 'bg-primary text-primary-foreground'
                : 'hover:bg-muted'
            }`}
          >
            System Logs
          </button>
          <button
            onClick={() => setActiveTab('metrics')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === 'metrics'
                ? 'bg-primary text-primary-foreground'
                : 'hover:bg-muted'
            }`}
          >
            <BarChart3 className="w-4 h-4 inline mr-2" />
            Metrics & Health
          </button>
        </div>
      </div>

      {/* Content */}
      {activeTab === 'logs' ? (
        <ResizablePanelGroup direction="horizontal" className="flex-1">
          {/* Logs Panel */}
          <ResizablePanel defaultSize={70} minSize={50}>
            <div className="h-full flex flex-col">
              {/* Filters */}
              <div className="flex gap-4 p-4 border-b">
                <select
                  value={filter.level || ''}
                  onChange={(e) => setFilter({ level: e.target.value || null })}
                  className="px-4 py-2 border rounded-lg bg-background"
                >
                  <option value="">All Levels</option>
                  {LOG_LEVELS.map((level) => (
                    <option key={level} value={level}>
                      {level}
                    </option>
                  ))}
                </select>
                <input
                  type="text"
                  value={filter.search}
                  onChange={(e) => setFilter({ search: e.target.value })}
                  placeholder="Filter logs..."
                  className="flex-1 px-4 py-2 border rounded-lg bg-background"
                />
                <button
                  onClick={exportLogs}
                  className="flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-accent transition-colors"
                >
                  <Download className="w-4 h-4" />
                  Export
                </button>
                <button
                  onClick={clearLogs}
                  className="flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-destructive/10 hover:text-destructive transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                  Clear
                </button>
              </div>

              {/* Logs Table */}
              <div className="flex-1 bg-card border-t overflow-hidden flex flex-col">
                <div className="grid grid-cols-[120px_80px_1fr_200px] gap-4 px-4 py-2 bg-muted/50 text-sm font-medium border-b">
                  <span>Timestamp</span>
                  <span>Level</span>
                  <span>Event</span>
                  <span>Details</span>
                </div>
                <div className="flex-1 overflow-auto p-4 space-y-1 font-mono text-sm">
                  {filteredLogs.map((log) => (
                    <div
                      key={log.id}
                      className="grid grid-cols-[120px_80px_1fr_200px] gap-4 py-1 hover:bg-muted/50 rounded"
                    >
                      <span className="text-muted-foreground text-xs">
                        {new Date(log.timestamp).toLocaleTimeString()}
                      </span>
                      <span
                        className={`text-xs font-bold ${
                          log.level === 'ERROR'
                            ? 'text-red-600'
                            : log.level === 'WARN'
                            ? 'text-yellow-600'
                            : log.level === 'DEBUG'
                            ? 'text-gray-500'
                            : 'text-blue-600'
                        }`}
                      >
                        {log.level}
                      </span>
                      <span className="truncate">{log.event}</span>
                      <span className="text-muted-foreground truncate text-xs">
                        {log.message || JSON.stringify(log.details)}
                      </span>
                    </div>
                  ))}
                  <div ref={logsEndRef} />
                  {filteredLogs.length === 0 && (
                    <div className="text-center py-8 text-muted-foreground">
                      No logs match your filters.
                    </div>
                  )}
                </div>
              </div>
            </div>
          </ResizablePanel>

          <ResizableHandle />

          {/* Side Panel */}
          <ResizablePanel defaultSize={30} minSize={25}>
            <div className="h-full overflow-auto p-4 space-y-4">
              <SystemHealth />
              <TelemetryMetrics />
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      ) : (
        /* Metrics Dashboard */
        <div className="flex-1 overflow-auto p-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Charts */}
            <div className="space-y-6">
              <div className="p-4 border rounded-xl bg-card">
                <h3 className="font-semibold mb-4">Latency Over Time</h3>
                <LatencyChart />
              </div>
              <div className="p-4 border rounded-xl bg-card">
                <h3 className="font-semibold mb-4">Request Volume</h3>
                <RequestVolumeChart />
              </div>
            </div>
            <div className="space-y-6">
              <div className="p-4 border rounded-xl bg-card">
                <h3 className="font-semibold mb-4">Agent Distribution</h3>
                <AgentDistributionChart />
              </div>
              <div className="p-4 border rounded-xl bg-card">
                <h3 className="font-semibold mb-4">Success Rate by Agent</h3>
                <SuccessRateChart />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
