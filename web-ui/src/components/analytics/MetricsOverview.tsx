import { useEffect, useState } from 'react'
import { Activity, TrendingUp, Clock, CheckCircle, AlertCircle } from 'lucide-react'

interface SystemMetrics {
  avg_cpu_percent: number
  avg_memory_percent: number
  collection_count: number
}

interface GoalMetrics {
  total: number
  completed: number
  failed: number
  success_rate: number
  avg_duration_seconds: number
}

interface MetricsSummary {
  system: SystemMetrics
  goals: GoalMetrics
  last_updated: string
}

export function MetricsOverview() {
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null)
  const [loading, setLoading] = useState(true)

  const fetchMetrics = async () => {
    try {
      const response = await fetch('/api/metrics/summary')
      if (response.ok) {
        const data = await response.json()
        setMetrics(data)
      }
    } catch (error) {
      console.error('Failed to fetch metrics:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchMetrics()
    const interval = setInterval(fetchMetrics, 30000) // Refresh every 30s
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="bg-white rounded-lg shadow p-6 animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-24 mb-2"></div>
            <div className="h-8 bg-gray-200 rounded w-16"></div>
          </div>
        ))}
      </div>
    )
  }

  if (!metrics) {
    return (
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-amber-800">
        <AlertCircle className="w-5 h-5 inline mr-2" />
        Metrics unavailable. Please check system status.
      </div>
    )
  }

  const cards = [
    {
      title: 'Goals Completed',
      value: metrics.goals.completed,
      subtitle: `${metrics.goals.total} total`,
      icon: CheckCircle,
      color: 'text-green-600',
      bgColor: 'bg-green-50',
    },
    {
      title: 'Success Rate',
      value: `${(metrics.goals.success_rate * 100).toFixed(1)}%`,
      subtitle: `${metrics.goals.failed} failed`,
      icon: TrendingUp,
      color: metrics.goals.success_rate > 0.7 ? 'text-green-600' : 'text-amber-600',
      bgColor: metrics.goals.success_rate > 0.7 ? 'bg-green-50' : 'bg-amber-50',
    },
    {
      title: 'Avg Duration',
      value: formatDuration(metrics.goals.avg_duration_seconds),
      subtitle: 'per goal',
      icon: Clock,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
    },
    {
      title: 'System Health',
      value: `${(100 - metrics.system.avg_cpu_percent).toFixed(0)}%`,
      subtitle: `CPU: ${metrics.system.avg_cpu_percent.toFixed(1)}%`,
      icon: Activity,
      color: metrics.system.avg_cpu_percent < 70 ? 'text-green-600' : 'text-red-600',
      bgColor: metrics.system.avg_cpu_percent < 70 ? 'bg-green-50' : 'bg-red-50',
    },
  ]

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {cards.map((card) => (
        <div key={card.title} className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">{card.title}</p>
              <p className={`text-2xl font-bold ${card.color}`}>{card.value}</p>
              <p className="text-xs text-gray-500 mt-1">{card.subtitle}</p>
            </div>
            <div className={`p-3 rounded-full ${card.bgColor}`}>
              <card.icon className={`w-6 h-6 ${card.color}`} />
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  return `${Math.round(seconds / 3600)}h`
}
