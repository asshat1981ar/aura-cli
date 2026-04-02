import { useEffect, useRef } from 'react'
import { Trash2, Download, Wifi, WifiOff } from 'lucide-react'
import { useLogStore, LogEntry } from '../stores/logStore'
import { useWebSocket } from '../hooks/useWebSocket'

const LOG_LEVELS = ['DEBUG', 'INFO', 'WARN', 'ERROR'] as const

export function Logs() {
  const { logs, isConnected, filter, addLog, setConnected, setFilter, clearLogs } =
    useLogStore()
  const logsEndRef = useRef<HTMLDivElement>(null)

  useWebSocket('ws://localhost:8000/ws/logs', {
    onMessage: (data) => {
      if (
        typeof data === 'object' &&
        data !== null &&
        'type' in data &&
        (data as { type: string }).type === 'log' &&
        'payload' in data
      ) {
        addLog((data as { payload: unknown }).payload as LogEntry)
      }
    },
    onConnect: () => setConnected(true),
    onDisconnect: () => setConnected(false),
    reconnect: true,
  })

  useEffect(() => {
    // Scroll to bottom when new logs arrive
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

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
    <div className="space-y-4 h-[calc(100vh-8rem)] flex flex-col">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Logs</h2>
          <p className="text-muted-foreground">
            Real-time system logs and events.
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${
              isConnected
                ? 'bg-green-100 text-green-800 dark:bg-green-900/30'
                : 'bg-red-100 text-red-800 dark:bg-red-900/30'
            }`}
          >
            {isConnected ? (
              <Wifi className="w-4 h-4" />
            ) : (
              <WifiOff className="w-4 h-4" />
            )}
            <span className="text-sm font-medium">
              {isConnected ? 'Live' : 'Disconnected'}
            </span>
          </div>
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
      </div>

      <div className="flex gap-4">
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
      </div>

      <div className="flex-1 bg-card border rounded-xl overflow-hidden flex flex-col">
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
  )
}
