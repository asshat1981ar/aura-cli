import { useEffect, useState } from 'react'
import { Trophy, Star, AlertTriangle, CheckCircle, XCircle } from 'lucide-react'

interface AgentPerf {
  agent_id: string
  agent_name: string
  total_tasks: number
  successful_tasks: number
  failed_tasks: number
  success_rate: number
  avg_duration_seconds: number
  avg_tokens: number
  performance_score: number
  recent_tasks: Array<{
    timestamp: number
    task_type: string
    success: boolean
    duration: number
  }>
}

export function AgentPerformanceDashboard() {
  const [performances, setPerformances] = useState<AgentPerf[]>([])
  const [loading, setLoading] = useState(true)

  const fetchPerformances = async () => {
    try {
      const response = await fetch('/api/agents/performance')
      if (response.ok) {
        const data = await response.json()
        setPerformances(data.performances || [])
      }
    } catch (error) {
      console.error('Failed to fetch agent performances:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPerformances()
    const interval = setInterval(fetchPerformances, 60000) // Refresh every minute
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6 animate-pulse">
        <div className="h-6 bg-gray-200 rounded w-40 mb-4"></div>
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-16 bg-gray-200 rounded"></div>
          ))}
        </div>
      </div>
    )
  }

  if (performances.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Agent Performance</h3>
        <p className="text-gray-500">No agent performance data available yet.</p>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Trophy className="w-5 h-5 text-amber-500" />
          Agent Leaderboard
        </h3>
      </div>

      <div className="space-y-3">
        {performances.map((agent, index) => (
          <AgentCard key={agent.agent_id} agent={agent} rank={index + 1} />
        ))}
      </div>
    </div>
  )
}

function AgentCard({ agent, rank }: { agent: AgentPerf; rank: number }) {
  const getRankStyle = (rank: number) => {
    if (rank === 1) return 'bg-amber-50 border-amber-200'
    if (rank === 2) return 'bg-gray-50 border-gray-200'
    if (rank === 3) return 'bg-orange-50 border-orange-200'
    return 'bg-white border-gray-100'
  }

  const getRankIcon = (rank: number) => {
    if (rank === 1) return '🥇'
    if (rank === 2) return '🥈'
    if (rank === 3) return '🥉'
    return `#${rank}`
  }

  const scoreColor = agent.performance_score >= 80 ? 'text-green-600' :
                     agent.performance_score >= 60 ? 'text-amber-600' : 'text-red-600'

  return (
    <div className={`border rounded-lg p-4 ${getRankStyle(rank)}`}>
      <div className="flex items-center gap-4">
        <div className="text-2xl font-bold w-12 text-center">
          {getRankIcon(rank)}
        </div>

        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h4 className="font-semibold">{agent.agent_name}</h4>
            {agent.performance_score >= 80 && (
              <Star className="w-4 h-4 text-amber-500 fill-amber-500" />
            )}
          </div>

          <div className="mt-2 grid grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-gray-500">Tasks</p>
              <p className="font-medium">{agent.total_tasks}</p>
            </div>
            <div>
              <p className="text-gray-500">Success Rate</p>
              <p className={`font-medium ${agent.success_rate >= 0.7 ? 'text-green-600' : 'text-red-600'}`}>
                {(agent.success_rate * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-gray-500">Avg Duration</p>
              <p className="font-medium">{formatDuration(agent.avg_duration_seconds)}</p>
            </div>
            <div>
              <p className="text-gray-500">Score</p>
              <p className={`font-bold ${scoreColor}`}>
                {agent.performance_score.toFixed(1)}
              </p>
            </div>
          </div>
        </div>

        <div className="flex flex-col items-end gap-1">
          <div className="flex items-center gap-1">
            <CheckCircle className="w-4 h-4 text-green-600" />
            <span className="text-sm text-green-600">{agent.successful_tasks}</span>
          </div>
          {agent.failed_tasks > 0 && (
            <div className="flex items-center gap-1">
              <XCircle className="w-4 h-4 text-red-600" />
              <span className="text-sm text-red-600">{agent.failed_tasks}</span>
            </div>
          )}
        </div>
      </div>

      {/* Recent Tasks */}
      {agent.recent_tasks.length > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          <p className="text-xs text-gray-500 mb-2">Recent Tasks:</p>
          <div className="flex gap-2">
            {agent.recent_tasks.slice(-5).map((task, i) => (
              <div
                key={i}
                className={`w-3 h-3 rounded-full ${
                  task.success ? 'bg-green-500' : 'bg-red-500'
                }`}
                title={`${task.task_type} - ${task.success ? 'Success' : 'Failed'} (${formatDuration(task.duration)})`}
              />
            ))}
          </div>
        </div>
      )}

      {/* Warning for low performance */}
      {agent.performance_score < 50 && (
        <div className="mt-3 flex items-center gap-2 text-amber-700 bg-amber-50 p-2 rounded text-sm">
          <AlertTriangle className="w-4 h-4" />
          Performance below threshold - review recommended
        </div>
      )}
    </div>
  )
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  return `${Math.round(seconds / 3600)}h`
}
