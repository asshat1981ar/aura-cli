import { useEffect, useMemo, useState } from 'react'
import {
  Target,
  Bot,
  Activity,
  Clock,
  CheckCircle,
  XCircle,
  TrendingUp,
  Zap,
  Wifi,
  WifiOff,
} from 'lucide-react'
import { StatsCard } from '../components/StatsCard'
import { useGoalStore, useGoalWebSocket } from '../stores/goalStore'
import { useAgentStore, useAgentWebSocket } from '../stores/agentStore'
import {
  GoalTrendChart,
  AgentPerformanceChart,
  GoalDistributionChart,
  ExecutionTimeChart,
} from '../components/Charts'

// Generate mock trend data for initial display
function generateTrendData() {
  const data = []
  const today = new Date()
  for (let i = 6; i >= 0; i--) {
    const date = new Date(today)
    date.setDate(date.getDate() - i)
    data.push({
      date: date.toLocaleDateString('en-US', { weekday: 'short' }),
      completed: Math.floor(Math.random() * 10) + 2,
      failed: Math.floor(Math.random() * 3),
      pending: Math.floor(Math.random() * 5),
    })
  }
  return data
}

// Generate mock execution time data
function generateExecutionData() {
  const data = []
  for (let i = 0; i < 10; i++) {
    data.push({
      time: `${i * 2}:00`,
      duration: Math.random() * 30 + 5,
    })
  }
  return data
}

export function Dashboard() {
  const { goals, fetchGoals, wsConnected: goalsWsConnected } = useGoalStore()
  const { agents, fetchAgents, wsConnected: agentsWsConnected } = useAgentStore()
  const [stats, setStats] = useState<any>(null)
  const [isLoadingStats, setIsLoadingStats] = useState(true)

  // Initialize WebSocket connections
  useGoalWebSocket()
  useAgentWebSocket()

  // Fetch initial data
  useEffect(() => {
    fetchGoals()
    fetchAgents()
    fetchStats()
    
    // Refresh stats every 10 seconds
    const interval = setInterval(fetchStats, 10000)
    return () => clearInterval(interval)
  }, [fetchGoals, fetchAgents])

  async function fetchStats() {
    try {
      const response = await fetch('/api/stats')
      if (response.ok) {
        const data = await response.json()
        setStats(data)
      }
    } catch (error) {
      console.error('Failed to fetch stats:', error)
    } finally {
      setIsLoadingStats(false)
    }
  }

  const pendingGoals = goals.filter((g) => g.status === 'pending').length
  const runningGoals = goals.filter((g) => g.status === 'running').length
  const completedGoals = goals.filter((g) => g.status === 'completed').length
  const failedGoals = goals.filter((g) => g.status === 'failed').length

  const activeAgents = agents.filter((a) => a.status === 'busy').length
  const idleAgents = agents.filter((a) => a.status === 'idle').length

  const trendData = useMemo(() => generateTrendData(), [])
  const executionData = useMemo(() => generateExecutionData(), [])

  const agentPerformanceData = useMemo(
    () =>
      agents.map((agent) => ({
        name: agent.name.split(' ')[0],
        tasks: agent.stats.tasks_completed,
      })),
    [agents]
  )

  const totalTasks = agents.reduce(
    (sum, a) => sum + a.stats.tasks_completed,
    0
  )
  const avgSuccessRate =
    agents.length > 0
      ? agents.reduce(
          (sum, a) =>
            sum +
            (a.stats.tasks_completed /
              (a.stats.tasks_completed + a.stats.tasks_failed || 1)),
          0
        ) / agents.length
      : 0

  const wsConnected = goalsWsConnected || agentsWsConnected

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Dashboard</h2>
          <p className="text-muted-foreground">
            Overview of your AURA system status and activity.
          </p>
        </div>
        <div className="flex items-center gap-4">
          {/* WebSocket Connection Status */}
          <div className={`flex items-center gap-2 text-sm ${wsConnected ? 'text-green-600' : 'text-amber-600'}`}>
            {wsConnected ? (
              <>
                <Wifi className="w-4 h-4" />
                <span>Live</span>
              </>
            ) : (
              <>
                <WifiOff className="w-4 h-4" />
                <span>Reconnecting...</span>
              </>
            )}
          </div>
          
          {/* Live indicator */}
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <div className={`w-2 h-2 rounded-full animate-pulse ${wsConnected ? 'bg-green-500' : 'bg-amber-500'}`} />
            {wsConnected ? 'Real-time updates' : 'Polling mode'}
          </div>
        </div>
      </div>

      {/* Stats from API */}
      {stats && (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <div className="bg-card border rounded-xl p-4">
            <div className="text-sm text-muted-foreground">Avg Latency</div>
            <div className="text-2xl font-bold">{stats.telemetry?.avg_latency_ms?.toFixed(0) || 'N/A'} ms</div>
          </div>
          <div className="bg-card border rounded-xl p-4">
            <div className="text-sm text-muted-foreground">Active Agents</div>
            <div className="text-2xl font-bold">{stats.agents?.active || 0}</div>
          </div>
          <div className="bg-card border rounded-xl p-4">
            <div className="text-sm text-muted-foreground">Total Records</div>
            <div className="text-2xl font-bold">{stats.telemetry?.total_records || 0}</div>
          </div>
          <div className="bg-card border rounded-xl p-4">
            <div className="text-sm text-muted-foreground">System Load</div>
            <div className="text-2xl font-bold">{stats.system?.cpu_percent?.toFixed(1) || 'N/A'}%</div>
          </div>
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatsCard
          title="Pending Goals"
          value={pendingGoals}
          description="Goals waiting to be processed"
          icon={Target}
        />
        <StatsCard
          title="Running Goals"
          value={runningGoals}
          description="Goals currently being executed"
          icon={Activity}
          trend={{ value: 12, isPositive: true }}
        />
        <StatsCard
          title="Active Agents"
          value={activeAgents}
          description={`${idleAgents} agents idle`}
          icon={Bot}
        />
        <StatsCard
          title="Success Rate"
          value={
            completedGoals + failedGoals > 0
              ? `${Math.round(
                  (completedGoals / (completedGoals + failedGoals)) * 100
                )}%`
              : 'N/A'
          }
          description="Last 24 hours"
          icon={CheckCircle}
        />
      </div>

      {/* Charts Row */}
      <div className="grid gap-4 md:grid-cols-2">
        <div className="bg-card border rounded-xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-primary" />
            <h3 className="text-lg font-semibold">Goal Trends (7 Days)</h3>
          </div>
          <GoalTrendChart data={trendData} />
        </div>

        <div className="bg-card border rounded-xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <Zap className="w-5 h-5 text-primary" />
            <h3 className="text-lg font-semibold">Goal Distribution</h3>
          </div>
          <GoalDistributionChart
            pending={pendingGoals}
            running={runningGoals}
            completed={completedGoals}
            failed={failedGoals}
          />
        </div>
      </div>

      {/* Agent Performance */}
      {agents.length > 0 && (
        <div className="bg-card border rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Bot className="w-5 h-5 text-primary" />
              <h3 className="text-lg font-semibold">Agent Performance</h3>
            </div>
            <div className="text-sm text-muted-foreground">
              Total tasks: {totalTasks} | Avg success:{' '}
              {(avgSuccessRate * 100).toFixed(0)}%
            </div>
          </div>
          <AgentPerformanceChart data={agentPerformanceData} />
        </div>
      )}

      {/* Execution Time Chart */}
      <div className="bg-card border rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-4">Execution Time (Last 20h)</h3>
        <ExecutionTimeChart data={executionData} />
      </div>

      {/* Recent Goals & Agent Status */}
      <div className="grid gap-4 md:grid-cols-2">
        <div className="bg-card border rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">Recent Goals</h3>
          {goals.slice(0, 5).map((goal) => (
            <div
              key={goal.id}
              className="flex items-center justify-between py-3 border-b last:border-0"
            >
              <div className="flex items-center gap-3">
                {goal.status === 'completed' ? (
                  <CheckCircle className="w-5 h-5 text-green-500" />
                ) : goal.status === 'failed' ? (
                  <XCircle className="w-5 h-5 text-red-500" />
                ) : goal.status === 'running' ? (
                  <Activity className="w-5 h-5 text-blue-500 animate-pulse" />
                ) : (
                  <Clock className="w-5 h-5 text-yellow-500" />
                )}
                <div>
                  <p className="font-medium truncate max-w-md">
                    {goal.description}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {new Date(goal.created_at).toLocaleString()}
                  </p>
                </div>
              </div>
              <span
                className={`text-xs font-medium px-2.5 py-0.5 rounded-full capitalize ${
                  goal.status === 'completed'
                    ? 'bg-green-100 text-green-800'
                    : goal.status === 'failed'
                    ? 'bg-red-100 text-red-800'
                    : goal.status === 'running'
                    ? 'bg-blue-100 text-blue-800'
                    : 'bg-yellow-100 text-yellow-800'
                }`}
              >
                {goal.status}
              </span>
            </div>
          ))}
          {goals.length === 0 && (
            <p className="text-muted-foreground text-center py-8">
              No goals yet. Create one to get started.
            </p>
          )}
        </div>

        <div className="bg-card border rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">Agent Status</h3>
          {agents.map((agent) => (
            <div
              key={agent.id}
              className="flex items-center justify-between py-3 border-b last:border-0"
            >
              <div className="flex items-center gap-3">
                <div
                  className={`w-2 h-2 rounded-full ${
                    agent.status === 'idle'
                      ? 'bg-green-500'
                      : agent.status === 'busy'
                      ? 'bg-blue-500 animate-pulse'
                      : agent.status === 'error'
                      ? 'bg-red-500'
                      : 'bg-gray-500'
                  }`}
                />
                <div>
                  <p className="font-medium">{agent.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {agent.current_task || 'No active task'}
                  </p>
                </div>
              </div>
              <div className="text-right">
                <span className="text-xs text-muted-foreground capitalize block">
                  {agent.status}
                </span>
                <span className="text-xs text-muted-foreground">
                  {agent.stats.tasks_completed} tasks
                </span>
              </div>
            </div>
          ))}
          {agents.length === 0 && (
            <p className="text-muted-foreground text-center py-8">
              No agents connected.
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
