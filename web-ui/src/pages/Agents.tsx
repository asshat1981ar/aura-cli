import { useEffect } from 'react'
import { Bot, RefreshCw } from 'lucide-react'
import { useAgentStore } from '../stores/agentStore'
import { StatusBadge } from '../components/StatusBadge'

export function Agents() {
  const { agents, fetchAgents, isLoading } = useAgentStore()

  useEffect(() => {
    fetchAgents()
    const interval = setInterval(fetchAgents, 5000)
    return () => clearInterval(interval)
  }, [fetchAgents])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Agents</h2>
          <p className="text-muted-foreground">
            Monitor and manage AURA agents.
          </p>
        </div>
        <button
          onClick={() => fetchAgents()}
          disabled={isLoading}
          className="flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-accent transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {agents.map((agent) => (
          <div
            key={agent.id}
            className="bg-card border rounded-xl p-6 hover:shadow-lg transition-shadow"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="p-3 bg-primary/10 rounded-lg">
                  <Bot className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold">{agent.name}</h3>
                  <p className="text-sm text-muted-foreground">{agent.type}</p>
                </div>
              </div>
              <StatusBadge status={agent.status} />
            </div>

            <div className="space-y-3">
              {agent.current_task && (
                <div>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider">
                    Current Task
                  </p>
                  <p className="text-sm truncate">{agent.current_task}</p>
                </div>
              )}

              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider">
                  Capabilities
                </p>
                <div className="flex flex-wrap gap-1 mt-1">
                  {agent.capabilities.map((cap) => (
                    <span
                      key={cap}
                      className="px-2 py-0.5 text-xs bg-muted rounded-full"
                    >
                      {cap}
                    </span>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 pt-4 border-t">
                <div className="text-center">
                  <p className="text-lg font-bold">
                    {agent.stats.tasks_completed}
                  </p>
                  <p className="text-xs text-muted-foreground">Completed</p>
                </div>
                <div className="text-center">
                  <p className="text-lg font-bold">{agent.stats.tasks_failed}</p>
                  <p className="text-xs text-muted-foreground">Failed</p>
                </div>
                <div className="text-center">
                  <p className="text-lg font-bold">
                    {Math.round(agent.stats.avg_execution_time)}s
                  </p>
                  <p className="text-xs text-muted-foreground">Avg Time</p>
                </div>
              </div>

              <p className="text-xs text-muted-foreground text-right">
                Last seen: {new Date(agent.last_seen).toLocaleTimeString()}
              </p>
            </div>
          </div>
        ))}
      </div>

      {agents.length === 0 && (
        <div className="text-center py-12 text-muted-foreground bg-card border rounded-xl">
          <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p className="text-lg font-medium">No agents connected</p>
          <p>Agents will appear here when they register with the system.</p>
        </div>
      )}
    </div>
  )
}
