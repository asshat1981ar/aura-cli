import { useEffect, useMemo, useState } from 'react'
import {
  Shield,
  AlertTriangle,
  CheckCircle,
  AlertCircle,
  Search,
  Filter,
  RefreshCw,
  Play,
  Clock,
  CheckSquare
} from 'lucide-react'
import {
  ResponsiveContainer,
  Treemap,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  BarChart,
  Bar,
  Cell,
  PieChart,
  Pie
} from 'recharts'
import { StatsCard } from '../components/StatsCard'
import { useCoverageStore } from '../stores/coverageStore'
import { cn } from '@/lib/utils'

// Treemap custom node
function TreemapNode({ depth, x, y, width, height, name, coverage }: any) {
  const colors = ['#22c55e', '#84cc16', '#eab308', '#f97316', '#ef4444']
  const colorIndex = Math.floor((100 - (coverage || 0)) / 20)
  const fill = colors[Math.min(colorIndex, colors.length - 1)]
  
  if (depth === 1) {
    return (
      <g>
        <rect
          x={x}
          y={y}
          width={width}
          height={height}
          style={{
            fill,
            stroke: '#fff',
            strokeWidth: 2,
            fillOpacity: 0.8,
          }}
        />
        {width > 60 && height > 40 && (
          <>
            <text
              x={x + width / 2}
              y={y + height / 2 - 8}
              textAnchor="middle"
              fill="#fff"
              fontSize={12}
              fontWeight={600}
            >
              {name}
            </text>
            <text
              x={x + width / 2}
              y={y + height / 2 + 10}
              textAnchor="middle"
              fill="#fff"
              fontSize={11}
            >
              {coverage?.toFixed(1)}%
            </text>
          </>
        )}
      </g>
    )
  }
  
  return null
}

const SEVERITY_CONFIG = {
  critical: { color: 'text-red-500 bg-red-500/10', icon: AlertCircle },
  high: { color: 'text-orange-500 bg-orange-500/10', icon: AlertTriangle },
  medium: { color: 'text-yellow-500 bg-yellow-500/10', icon: Shield },
  low: { color: 'text-blue-500 bg-blue-500/10', icon: CheckCircle }
}

export function Coverage() {
  const { 
    coverage, 
    gaps, 
    tests, 
    modules, 
    isLoading, 
    refreshAll, 
    runTests 
  } = useCoverageStore()
  
  const [filterSeverity, setFilterSeverity] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [runningTests, setRunningTests] = useState(false)

  useEffect(() => {
    refreshAll()
    const interval = setInterval(refreshAll, 30000)
    return () => clearInterval(interval)
  }, [refreshAll])

  const handleRunTests = async () => {
    setRunningTests(true)
    await runTests()
    setTimeout(() => {
      refreshAll()
      setRunningTests(false)
    }, 5000)
  }

  const stats = useMemo(() => {
    if (!coverage || !tests) {
      return {
        overallCoverage: 0,
        totalFiles: 0,
        uncoveredFunctions: 0,
        criticalGaps: 0,
        highGaps: 0,
        testPassRate: 0,
        testDuration: 0
      }
    }
    
    const criticalGaps = gaps.filter(g => g.severity === 'critical').length
    const highGaps = gaps.filter(g => g.severity === 'high').length
    const testPassRate = tests.total > 0 ? (tests.passed / tests.total) * 100 : 0
    
    return {
      overallCoverage: coverage.overall,
      totalFiles: Object.keys(coverage.files).length,
      uncoveredFunctions: gaps.length,
      criticalGaps,
      highGaps,
      testPassRate,
      testDuration: tests.duration
    }
  }, [coverage, gaps, tests])

  const filteredGaps = useMemo(() => {
    let filtered = gaps
    
    if (filterSeverity !== 'all') {
      filtered = filtered.filter(g => g.severity === filterSeverity)
    }
    
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(g =>
        g.file_path.toLowerCase().includes(query) ||
        g.function_name.toLowerCase().includes(query)
      )
    }
    
    return filtered.slice(0, 10)
  }, [gaps, filterSeverity, searchQuery])

  const treemapData = useMemo(() => {
    return modules.map(m => ({
      name: m.name,
      size: m.lines_total,
      coverage: m.coverage,
      children: m.children?.map(c => ({
        name: c.name,
        size: c.lines_total,
        coverage: c.coverage,
      }))
    }))
  }, [modules])

  const testStatusData = useMemo(() => {
    if (!tests) return []
    return [
      { name: 'Passed', value: tests.passed, color: '#22c55e' },
      { name: 'Failed', value: tests.failed, color: '#ef4444' },
      { name: 'Skipped', value: tests.skipped, color: '#eab308' }
    ]
  }, [tests])

  if (isLoading && !coverage) {
    return (
      <div className="flex items-center justify-center h-full">
        <RefreshCw className="w-8 h-8 animate-spin text-primary" />
      </div>
    )
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Shield className="w-8 h-8 text-primary" />
            Coverage & Quality Dashboard
          </h1>
          <p className="text-muted-foreground mt-1">
            Test coverage analysis, quality metrics, and gap identification
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => refreshAll()}
            disabled={isLoading}
            className="flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-accent transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <button
            onClick={handleRunTests}
            disabled={runningTests}
            className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
          >
            <Play className={`w-4 h-4 ${runningTests ? 'animate-pulse' : ''}`} />
            {runningTests ? 'Running...' : 'Run Tests'}
          </button>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatsCard
          title="Overall Coverage"
          value={`${stats.overallCoverage.toFixed(1)}%`}
          description={`${stats.totalFiles} files analyzed`}
          icon={Shield}
          trend={{ value: 5.2, isPositive: true }}
        />
        <StatsCard
          title="Uncovered Functions"
          value={stats.uncoveredFunctions}
          description={`${stats.criticalGaps} critical, ${stats.highGaps} high priority`}
          icon={AlertTriangle}
          trend={{ value: 3, isPositive: false }}
        />
        <StatsCard
          title="Test Pass Rate"
          value={`${stats.testPassRate.toFixed(1)}%`}
          description={`${tests?.passed || 0} of ${tests?.total || 0} tests passed`}
          icon={CheckSquare}
          trend={{ value: 1.2, isPositive: true }}
        />
        <StatsCard
          title="Test Duration"
          value={`${stats.testDuration.toFixed(1)}s`}
          description="Last test run"
          icon={Clock}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Coverage Heatmap */}
        <div className="lg:col-span-2 bg-card rounded-xl border p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Coverage Heatmap</h3>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 rounded-full bg-green-500" /> Good (&gt;80%)
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 rounded-full bg-yellow-500" /> Fair (60-80%)
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 rounded-full bg-red-500" /> Poor (&lt;60%)
              </span>
            </div>
          </div>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <Treemap
                data={treemapData}
                dataKey="size"
                aspectRatio={4 / 3}
                stroke="#fff"
                fill="#8884d8"
                content={<TreemapNode />}
              />
            </ResponsiveContainer>
          </div>
        </div>

        {/* Test Status */}
        <div className="bg-card rounded-xl border p-6">
          <h3 className="text-lg font-semibold mb-4">Test Status</h3>
          <div className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={testStatusData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {testStatusData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center gap-4 mt-4">
            {testStatusData.map((item) => (
              <div key={item.name} className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
                <span className="text-xs text-muted-foreground">{item.name}: {item.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Coverage Gaps Table */}
      <div className="bg-card rounded-xl border p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold">Coverage Gaps</h3>
            <p className="text-sm text-muted-foreground">
              Functions requiring test coverage, sorted by impact score
            </p>
          </div>
          <div className="flex items-center gap-3">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search functions..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9 pr-4 py-2 bg-muted rounded-lg text-sm w-64 focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
            
            {/* Severity Filter */}
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-muted-foreground" />
              <select
                value={filterSeverity}
                onChange={(e) => setFilterSeverity(e.target.value)}
                className="px-3 py-2 bg-muted rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value="all">All Severities</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
            </div>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b">
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Severity</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Function</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">File</th>
                <th className="text-center py-3 px-4 text-sm font-medium text-muted-foreground">Line</th>
                <th className="text-center py-3 px-4 text-sm font-medium text-muted-foreground">Complexity</th>
                <th className="text-center py-3 px-4 text-sm font-medium text-muted-foreground">Impact Score</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Reason</th>
              </tr>
            </thead>
            <tbody>
              {filteredGaps.map((gap) => {
                const config = SEVERITY_CONFIG[gap.severity]
                const Icon = config.icon
                return (
                  <tr
                    key={gap.id}
                    className="border-b last:border-0 hover:bg-muted/50 transition-colors"
                  >
                    <td className="py-3 px-4">
                      <span className={cn("inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium", config.color)}>
                        <Icon className="w-3 h-3" />
                        {gap.severity}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-medium">{gap.function_name}</td>
                    <td className="py-3 px-4 text-sm text-muted-foreground font-mono">{gap.file_path}</td>
                    <td className="py-3 px-4 text-center text-sm">{gap.line_number}</td>
                    <td className="py-3 px-4 text-center">
                      <span className={cn("inline-flex items-center justify-center w-8 h-8 rounded-full text-xs font-medium",
                        gap.complexity > 15 ? 'bg-red-500/10 text-red-500' :
                        gap.complexity > 10 ? 'bg-yellow-500/10 text-yellow-500' :
                        'bg-green-500/10 text-green-500'
                      )}>
                        {gap.complexity}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-center">
                      <div className="flex items-center justify-center gap-2">
                        <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
                          <div
                            className={cn("h-full rounded-full",
                              gap.impact_score > 80 ? 'bg-red-500' :
                              gap.impact_score > 60 ? 'bg-orange-500' :
                              gap.impact_score > 40 ? 'bg-yellow-500' :
                              'bg-blue-500'
                            )}
                            style={{ width: `${gap.impact_score}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium">{gap.impact_score.toFixed(0)}</span>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-sm text-muted-foreground">{gap.reason}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {filteredGaps.length === 0 && (
          <div className="text-center py-12 text-muted-foreground">
            <CheckCircle className="w-12 h-12 mx-auto mb-4 text-green-500" />
            <p className="text-lg font-medium">No coverage gaps found</p>
            <p className="text-sm">All functions are adequately covered!</p>
          </div>
        )}
      </div>

      {/* Module Breakdown */}
      <div className="bg-card rounded-xl border p-6">
        <h3 className="text-lg font-semibold mb-4">Module Coverage Breakdown</h3>
        <div className="h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={modules}
              layout="vertical"
              margin={{ left: 20, right: 30, top: 10, bottom: 10 }}
            >
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis
                type="number"
                domain={[0, 100]}
                tickFormatter={(value) => `${value}%`}
                tick={{ fontSize: 11 }}
              />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fontSize: 12 }}
                width={80}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'var(--background)',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                }}
                formatter={(value: number) => [`${value.toFixed(1)}%`, 'Coverage']}
              />
              <Bar dataKey="coverage" radius={[0, 4, 4, 0]}>
                {modules.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={
                      entry.coverage >= 80 ? '#22c55e' :
                      entry.coverage >= 60 ? '#eab308' :
                      '#ef4444'
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
