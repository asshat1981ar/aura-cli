import { MetricsOverview } from '../components/analytics/MetricsOverview'
import { GoalTrends } from '../components/analytics/GoalTrends'
import { AgentPerformanceDashboard } from '../components/analytics/AgentPerformance'
import { InsightsPanel } from '../components/analytics/InsightsPanel'

export function Analytics() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Analytics</h2>
        <p className="text-gray-500">
          Comprehensive metrics and insights for your AURA system.
        </p>
      </div>

      {/* Key Metrics */}
      <MetricsOverview />

      {/* Main Content Grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Goal Trends */}
        <GoalTrends />

        {/* Agent Performance */}
        <AgentPerformanceDashboard />
      </div>

      {/* Insights Panel */}
      <InsightsPanel />
    </div>
  )
}

export default Analytics
