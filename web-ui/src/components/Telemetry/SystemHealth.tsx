import { useEffect } from 'react'
import { 
  Heart, 
  CheckCircle2, 
  AlertTriangle, 
  XCircle, 
  Activity,
  Database,
  Server
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useTelemetryStore } from '@/stores/telemetryStore'

const STATUS_CONFIG = {
  healthy: { 
    icon: CheckCircle2, 
    color: 'text-green-500', 
    bg: 'bg-green-500',
    label: 'Healthy' 
  },
  degraded: { 
    icon: AlertTriangle, 
    color: 'text-yellow-500', 
    bg: 'bg-yellow-500',
    label: 'Degraded' 
  },
  critical: { 
    icon: XCircle, 
    color: 'text-red-500', 
    bg: 'bg-red-500',
    label: 'Critical' 
  }
}

const CHECK_ICONS = {
  agents: Server,
  queue: Database,
  api: Activity
}

export function SystemHealth() {
  const { health, fetchHealth, stats } = useTelemetryStore()

  useEffect(() => {
    fetchHealth()
    const interval = setInterval(fetchHealth, 30000)
    return () => clearInterval(interval)
  }, [fetchHealth])

  if (!health) {
    return (
      <div className="p-4 text-center text-muted-foreground">
        <Heart className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p className="text-sm">Loading health status...</p>
      </div>
    )
  }

  const status = STATUS_CONFIG[health.status]
  const StatusIcon = status.icon

  return (
    <div className="space-y-4">
      {/* Overall Status */}
      <div className={cn(
        "p-4 rounded-xl border-2",
        health.status === 'healthy' ? 'border-green-200 bg-green-50' :
        health.status === 'degraded' ? 'border-yellow-200 bg-yellow-50' :
        'border-red-200 bg-red-50'
      )}>
        <div className="flex items-center gap-3">
          <div className={cn(
            "w-12 h-12 rounded-full flex items-center justify-center",
            health.status === 'healthy' ? 'bg-green-100' :
            health.status === 'degraded' ? 'bg-yellow-100' :
            'bg-red-100'
          )}>
            <Heart className={cn("w-6 h-6", status.color)} />
          </div>
          <div>
            <h3 className="font-semibold">{status.label}</h3>
            <p className="text-sm text-muted-foreground">
              Last updated: {new Date(health.timestamp).toLocaleTimeString()}
            </p>
          </div>
          <StatusIcon className={cn("w-6 h-6 ml-auto", status.color)} />
        </div>
      </div>

      {/* Individual Checks */}
      <div className="grid grid-cols-3 gap-3">
        {Object.entries(health.checks).map(([check, result]) => {
          const CheckIcon = CHECK_ICONS[check as keyof typeof CHECK_ICONS]
          return (
            <div 
              key={check}
              className={cn(
                "p-3 rounded-lg border text-center",
                result === 'pass' ? 'bg-green-50 border-green-200' :
                result === 'warn' ? 'bg-yellow-50 border-yellow-200' :
                'bg-red-50 border-red-200'
              )}
            >
              <CheckIcon className={cn(
                "w-5 h-5 mx-auto mb-1",
                result === 'pass' ? 'text-green-600' :
                result === 'warn' ? 'text-yellow-600' :
                'text-red-600'
              )} />
              <p className="text-xs font-medium capitalize">{check}</p>
              <p className={cn(
                "text-[10px] uppercase",
                result === 'pass' ? 'text-green-600' :
                result === 'warn' ? 'text-yellow-600' :
                'text-red-600'
              )}>
                {result}
              </p>
            </div>
          )
        })}
      </div>

      {/* Quick Stats */}
      {stats && (
        <div className="pt-2 border-t">
          <h4 className="text-xs font-medium text-muted-foreground mb-2 uppercase">
            Current Load
          </h4>
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span>Goals</span>
              <span className="font-medium">
                {stats.goals.running} running / {stats.goals.pending} pending
              </span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span>Agents</span>
              <span className="font-medium">
                {stats.agents.active} active / {stats.agents.total} total
              </span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span>Latency</span>
              <span className="font-medium">
                {stats.telemetry.avg_latency_ms}ms avg
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
