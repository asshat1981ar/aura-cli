import { 
  Bot, 
  Activity, 
  Clock, 
  CheckCircle2, 
  AlertCircle, 
  PauseCircle,
  Power,
  Zap
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Agent } from '@/stores/agentStore'

interface AgentCardProps {
  agent: Agent
  isSelected: boolean
  onClick: () => void
}

const STATUS_CONFIG = {
  idle: { icon: Power, color: 'text-green-500', bg: 'bg-green-500', label: 'Idle' },
  busy: { icon: Zap, color: 'text-blue-500', bg: 'bg-blue-500', label: 'Busy' },
  paused: { icon: PauseCircle, color: 'text-yellow-500', bg: 'bg-yellow-500', label: 'Paused' },
  error: { icon: AlertCircle, color: 'text-red-500', bg: 'bg-red-500', label: 'Error' },
  offline: { icon: Power, color: 'text-gray-400', bg: 'bg-gray-400', label: 'Offline' }
}

export function AgentCard({ agent, isSelected, onClick }: AgentCardProps) {
  const status = STATUS_CONFIG[agent.status]

  return (
    <div
      onClick={onClick}
      className={cn(
        "p-4 rounded-lg border cursor-pointer transition-all hover:shadow-md",
        isSelected 
          ? "border-primary bg-primary/5 ring-1 ring-primary" 
          : "border-border bg-card hover:border-primary/50"
      )}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className={cn(
            "w-10 h-10 rounded-lg flex items-center justify-center",
            isSelected ? "bg-primary/10" : "bg-muted"
          )}>
            <Bot className={cn("w-5 h-5", isSelected ? "text-primary" : "text-muted-foreground")} />
          </div>
          <div>
            <h3 className="font-semibold text-sm">{agent.name}</h3>
            <p className="text-xs text-muted-foreground">{agent.type}</p>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <div className={cn("w-2 h-2 rounded-full animate-pulse", status.bg)} />
          <span className={cn("text-xs font-medium", status.color)}>{status.label}</span>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-3 gap-2 mt-4 pt-3 border-t">
        <div className="text-center">
          <div className="flex items-center justify-center gap-1 text-muted-foreground">
            <Activity className="w-3 h-3" />
            <span className="text-[10px] uppercase">Executions</span>
          </div>
          <p className="text-sm font-semibold mt-0.5">{agent.metrics.total_executions}</p>
        </div>
        <div className="text-center border-x">
          <div className="flex items-center justify-center gap-1 text-muted-foreground">
            <CheckCircle2 className="w-3 h-3" />
            <span className="text-[10px] uppercase">Success</span>
          </div>
          <p className={cn(
            "text-sm font-semibold mt-0.5",
            agent.metrics.success_rate >= 90 ? "text-green-600" : 
            agent.metrics.success_rate >= 70 ? "text-yellow-600" : "text-red-600"
          )}>
            {agent.metrics.success_rate}%
          </p>
        </div>
        <div className="text-center">
          <div className="flex items-center justify-center gap-1 text-muted-foreground">
            <Clock className="w-3 h-3" />
            <span className="text-[10px] uppercase">Latency</span>
          </div>
          <p className="text-sm font-semibold mt-0.5">{agent.metrics.avg_latency_ms}ms</p>
        </div>
      </div>

      {/* Capabilities */}
      {agent.capabilities.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-3">
          {agent.capabilities.slice(0, 3).map((cap) => (
            <span 
              key={cap}
              className="text-[10px] px-2 py-0.5 bg-muted rounded-full text-muted-foreground"
            >
              {cap}
            </span>
          ))}
          {agent.capabilities.length > 3 && (
            <span className="text-[10px] px-2 py-0.5 bg-muted rounded-full text-muted-foreground">
              +{agent.capabilities.length - 3}
            </span>
          )}
        </div>
      )}

      {/* Last Active */}
      {agent.last_active && (
        <p className="text-[10px] text-muted-foreground mt-3">
          Last active: {new Date(agent.last_active).toLocaleString()}
        </p>
      )}
    </div>
  )
}
