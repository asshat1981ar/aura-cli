import { useEffect } from 'react'
import { Bot, Loader2 } from 'lucide-react'
import { AgentCard } from './AgentCard'
import { useAgentStore } from '@/stores/agentStore'

export function AgentGrid() {
  const { overview, selectedAgent, isLoading, fetchOverview, selectAgent } = useAgentStore()

  useEffect(() => {
    fetchOverview()
    
    // Refresh every 10 seconds
    const interval = setInterval(fetchOverview, 10000)
    return () => clearInterval(interval)
  }, [fetchOverview])

  if (isLoading && overview.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    )
  }

  if (overview.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
        <p>No agents registered</p>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-4">
      {overview.map((agent) => (
        <AgentCard
          key={agent.id}
          agent={agent}
          isSelected={selectedAgent?.id === agent.id}
          onClick={() => selectAgent(agent.id === selectedAgent?.id ? null : agent)}
        />
      ))}
    </div>
  )
}
