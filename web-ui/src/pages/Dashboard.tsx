import { useEffect } from 'react'
import {
  Target,
  Bot,
  Activity,
  Clock,
  CheckCircle,
  XCircle,
} from 'lucide-react'
import { StatsCard } from '../components/StatsCard'
import { useGoalStore } from '../stores/goalStore'
import { useAgentStore } from '../stores/agentStore'

export function Dashboard() {
  const { goals, fetchGoals } = useGoalStore()
  const { agents, fetchAgents } = useAgentStore()

  useEffect(() => {
    fetchGoals()
    fetchAgents()
    const interval = setInterval(() => {
      fetchGoals()
      fetchAgents()
    }, 5000)
    return () => clearInterval(interval)
  }, [fetchGoals, fetchAgents])

  const pendingGoals = goals.filter((g) => g.status === 'pending').length
  const runningGoals = goals.filter((g) => g.status === 'running').length
  const completedGoals = goals.filter((g) => g.status === 'completed').length
  const failedGoals = goals.filter((g) => g.status === 'failed').length

  const activeAgents = agents.filter((a) => a.status === 'busy').length
  const idleAgents = agents.filter((a) => a.status === 'idle').length

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Dashboard</h2>
        <p className="text-muted-foreground">
          Overview of your AURA system status and activity.
        </p>
      </div>

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
              <span className="text-xs text-muted-foreground capitalize">
                {agent.status}
              </span>
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
