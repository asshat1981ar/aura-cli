import { useEffect, useState } from 'react'
import { Lightbulb, AlertTriangle, CheckCircle, Info, TrendingUp, TrendingDown } from 'lucide-react'

interface Insight {
  type: 'success' | 'warning' | 'info' | 'critical'
  category: string
  title: string
  message: string
  recommendation: string
}

export function InsightsPanel() {
  const [insights, setInsights] = useState<Insight[]>([])
  const [loading, setLoading] = useState(true)

  const fetchInsights = async () => {
    try {
      const response = await fetch('/api/analytics/insights')
      if (response.ok) {
        const data = await response.json()
        setInsights(data.insights || [])
      }
    } catch (error) {
      console.error('Failed to fetch insights:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchInsights()
    const interval = setInterval(fetchInsights, 60000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6 animate-pulse">
        <div className="h-6 bg-gray-200 rounded w-32 mb-4"></div>
        <div className="space-y-3">
          {[1, 2].map((i) => (
            <div key={i} className="h-20 bg-gray-200 rounded"></div>
          ))}
        </div>
      </div>
    )
  }

  if (insights.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold flex items-center gap-2 mb-4">
          <Lightbulb className="w-5 h-5 text-amber-500" />
          Insights
        </h3>
        <div className="text-center py-8 text-gray-500">
          <CheckCircle className="w-12 h-12 mx-auto mb-3 text-green-500" />
          <p>All systems operating normally.</p>
          <p className="text-sm">No insights available at this time.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold flex items-center gap-2 mb-4">
        <Lightbulb className="w-5 h-5 text-amber-500" />
        Insights & Recommendations
      </h3>

      <div className="space-y-4">
        {insights.map((insight, index) => (
          <InsightCard key={index} insight={insight} />
        ))}
      </div>
    </div>
  )
}

function InsightCard({ insight }: { insight: Insight }) {
  const styles = {
    success: {
      bg: 'bg-green-50',
      border: 'border-green-200',
      icon: CheckCircle,
      iconColor: 'text-green-600',
    },
    warning: {
      bg: 'bg-amber-50',
      border: 'border-amber-200',
      icon: AlertTriangle,
      iconColor: 'text-amber-600',
    },
    info: {
      bg: 'bg-blue-50',
      border: 'border-blue-200',
      icon: Info,
      iconColor: 'text-blue-600',
    },
    critical: {
      bg: 'bg-red-50',
      border: 'border-red-200',
      icon: AlertTriangle,
      iconColor: 'text-red-600',
    },
  }

  const style = styles[insight.type]
  const Icon = style.icon

  return (
    <div className={`border rounded-lg p-4 ${style.bg} ${style.border}`}>
      <div className="flex items-start gap-3">
        <div className={`mt-0.5 ${style.iconColor}`}>
          <Icon className="w-5 h-5" />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <h4 className="font-semibold text-gray-900">{insight.title}</h4>
            <span className="text-xs px-2 py-0.5 bg-white rounded-full text-gray-600">
              {insight.category}
            </span>
          </div>
          <p className="text-gray-700 mb-2">{insight.message}</p>
          <div className="flex items-center gap-2 text-sm">
            <Lightbulb className="w-4 h-4 text-amber-500" />
            <span className="text-gray-600">{insight.recommendation}</span>
          </div>
        </div>
      </div>
    </div>
  )
}
