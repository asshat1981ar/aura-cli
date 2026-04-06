import { useEffect } from 'react'
import { 
  Activity, 
  Clock, 
  CheckCircle2, 
  Database,
  Zap,
  BarChart3
} from 'lucide-react'
import { useTelemetryStore } from '@/stores/telemetryStore'

export function TelemetryMetrics() {
  const { summary, stats, fetchSummary, fetchStats } = useTelemetryStore()

  useEffect(() => {
    fetchSummary()
    fetchStats()
    
    const interval = setInterval(() => {
      fetchSummary()
      fetchStats()
    }, 10000)
    
    return () => clearInterval(interval)
  }, [fetchSummary, fetchStats])

  const metrics = [
    {
      icon: Database,
      label: 'Total Records',
      value: summary?.total_records || 0,
      subtext: 'telemetry events'
    },
    {
      icon: Clock,
      label: 'Avg Latency',
      value: `${summary?.avg_latency_ms || 0}ms`,
      subtext: 'response time'
    },
    {
      icon: CheckCircle2,
      label: 'Success Rate',
      value: `${summary?.success_rate || 0}%`,
      subtext: 'execution success'
    },
    {
      icon: Activity,
      label: 'Throughput',
      value: `${stats?.telemetry.throughput_goals_per_hour || 0}`,
      subtext: 'goals/hour'
    },
    {
      icon: Zap,
      label: 'Total Tokens',
      value: summary?.total_tokens?.toLocaleString() || '0',
      subtext: 'LLM tokens used'
    },
    {
      icon: BarChart3,
      label: 'Active Agents',
      value: `${stats?.agents.active || 0}/${stats?.agents.total || 0}`,
      subtext: 'agents working'
    }
  ]

  return (
    <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
      {metrics.map((metric) => {
        const Icon = metric.icon
        return (
          <div 
            key={metric.label}
            className="p-3 rounded-lg border bg-card hover:shadow-sm transition-shadow"
          >
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <Icon className="w-4 h-4" />
              <span className="text-xs">{metric.label}</span>
            </div>
            <p className="text-xl font-bold">{metric.value}</p>
            <p className="text-xs text-muted-foreground">{metric.subtext}</p>
          </div>
        )
      })}
    </div>
  )
}
