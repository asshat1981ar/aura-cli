import { useEffect } from 'react'
import { 
  Server, 
  Terminal, 
  Globe, 
  CheckCircle2, 
  XCircle, 
  AlertCircle,
  ChevronRight
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useMCPStore } from '@/stores/mcpStore'

const STATUS_ICONS = {
  connected: CheckCircle2,
  disconnected: XCircle,
  error: AlertCircle
}

const STATUS_COLORS = {
  connected: 'text-green-500',
  disconnected: 'text-gray-400',
  error: 'text-red-500'
}

export function MCPServerList() {
  const { 
    servers, 
    selectedServer, 
    isLoading,
    fetchServers,
    selectServer
  } = useMCPStore()

  useEffect(() => {
    fetchServers()
  }, [fetchServers])

  if (isLoading && servers.length === 0) {
    return (
      <div className="p-4 space-y-3">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-16 bg-muted rounded-lg animate-pulse" />
        ))}
      </div>
    )
  }

  if (servers.length === 0) {
    return (
      <div className="p-4 text-center text-muted-foreground">
        <Server className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p className="text-sm">No MCP servers configured</p>
      </div>
    )
  }

  return (
    <div className="divide-y">
      {servers.map((server) => {
        const StatusIcon = STATUS_ICONS[server.status]
        return (
          <div
            key={server.id}
            onClick={() => selectServer(server.id === selectedServer?.id ? null : server)}
            className={cn(
              "p-3 cursor-pointer hover:bg-accent transition-colors group",
              selectedServer?.id === server.id && "bg-accent"
            )}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  {server.type === 'http' ? (
                    <Globe className="w-4 h-4 text-blue-500" />
                  ) : (
                    <Terminal className="w-4 h-4 text-purple-500" />
                  )}
                  <h3 className="font-medium text-sm truncate">
                    {server.name}
                  </h3>
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  {server.type.toUpperCase()} • {server.tools_count} tools
                </p>
                {server.config.url && (
                  <p className="text-xs text-muted-foreground truncate mt-1">
                    {server.config.url}
                  </p>
                )}
                {server.config.command && (
                  <p className="text-xs text-muted-foreground truncate mt-1">
                    {server.config.command} {server.config.args?.join(' ')}
                  </p>
                )}
              </div>
              
              <div className="flex items-center gap-2">
                <StatusIcon className={cn("w-4 h-4", STATUS_COLORS[server.status])} />
                <ChevronRight className={cn(
                  "w-4 h-4 text-muted-foreground transition-transform",
                  selectedServer?.id === server.id && "rotate-90"
                )} />
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
