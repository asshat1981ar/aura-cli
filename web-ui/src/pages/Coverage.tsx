import { useEffect, useMemo, useState } from 'react'
import {
  Shield,
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  FileCode,
  GitBranch,
  AlertCircle,
  Search,
  Filter,
  Download,
  RefreshCw,
} from 'lucide-react'
import {
  ResponsiveContainer,
  Treemap,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  BarChart,
  Bar,
  Cell,
} from 'recharts'
import { StatsCard } from '../components/StatsCard'

// Coverage data types
interface CoverageGap {
  id: string
  file_path: string
  function_name: string
  line_number: number
  complexity: number
  impact_score: number
  severity: 'critical' | 'high' | 'medium' | 'low'
  reason: string
}

interface ModuleCoverage {
  name: string
  path: string
  coverage: number
  lines_total: number
  lines_covered: number
  functions_total: number
  functions_covered: number
  children?: ModuleCoverage[]
}

interface TrendData {
  date: string
  coverage: number
  files_covered: number
  total_files: number
}

// Generate mock coverage gaps
function generateCoverageGaps(): CoverageGap[] {
  const files = [
    'core/orchestrator.py',
    'agents/coder.py',
    'core/mcp_client.py',
    'agents/planner.py',
    'core/verification.py',
    'agents/debugger.py',
    'core/code_analysis.py',
    'memory/store.py',
    'agents/critic.py',
    'core/policy.py',
  ]
  
  const functions = [
    'run_cycle',
    'process_task',
    'validate_output',
    'generate_plan',
    'apply_changes',
    'analyze_error',
    'extract_symbols',
    'query_memory',
    'critique_plan',
    'evaluate_policy',
  ]
  
  const severities: ('critical' | 'high' | 'medium' | 'low')[] = ['critical', 'high', 'high', 'medium', 'medium', 'medium', 'low', 'low', 'low', 'low']
  
  return Array.from({ length: 20 }, (_, i) => ({
    id: `gap_${i + 1}`,
    file_path: files[i % files.length],
    function_name: `${functions[i % functions.length]}_${Math.floor(i / 10) + 1}`,
    line_number: Math.floor(Math.random() * 500) + 1,
    complexity: Math.floor(Math.random() * 20) + 1,
    impact_score: Math.random() * 100,
    severity: severities[i % severities.length],
    reason: ['Untested branch', 'Missing edge case', 'No error handling test', 'Integration not tested'][i % 4],
  })).sort((a, b) => b.impact_score - a.impact_score)
}

// Generate mock module coverage data
function generateModuleCoverage(): ModuleCoverage[] {
  return [
    {
      name: 'core',
      path: 'core',
      coverage: 78.5,
      lines_total: 12500,
      lines_covered: 9813,
      functions_total: 850,
      functions_covered: 667,
      children: [
        { name: 'orchestrator', path: 'core/orchestrator.py', coverage: 82.3, lines_total: 2100, lines_covered: 1728, functions_total: 45, functions_covered: 37 },
        { name: 'mcp_client', path: 'core/mcp_client.py', coverage: 65.2, lines_total: 890, lines_covered: 580, functions_total: 32, functions_covered: 21 },
        { name: 'verification', path: 'core/verification.py', coverage: 91.5, lines_total: 650, lines_covered: 595, functions_total: 28, functions_covered: 26 },
        { name: 'policy', path: 'core/policy.py', coverage: 88.1, lines_total: 420, lines_covered: 370, functions_total: 22, functions_covered: 19 },
        { name: 'code_analysis', path: 'core/code_analysis.py', coverage: 71.4, lines_total: 1200, lines_covered: 857, functions_total: 48, functions_covered: 34 },
      ]
    },
    {
      name: 'agents',
      path: 'agents',
      coverage: 72.8,
      lines_total: 8900,
      lines_covered: 6480,
      functions_total: 620,
      functions_covered: 451,
      children: [
        { name: 'coder', path: 'agents/coder.py', coverage: 76.9, lines_total: 1500, lines_covered: 1154, functions_total: 38, functions_covered: 29 },
        { name: 'planner', path: 'agents/planner.py', coverage: 81.2, lines_total: 980, lines_covered: 796, functions_total: 25, functions_covered: 20 },
        { name: 'debugger', path: 'agents/debugger.py', coverage: 58.3, lines_total: 720, lines_covered: 420, functions_total: 28, functions_covered: 16 },
        { name: 'critic', path: 'agents/critic.py', coverage: 85.7, lines_total: 650, lines_covered: 557, functions_total: 22, functions_covered: 19 },
      ]
    },
    {
      name: 'memory',
      path: 'memory',
      coverage: 85.2,
      lines_total: 3200,
      lines_covered: 2726,
      functions_total: 180,
      functions_covered: 153,
      children: [
        { name: 'store', path: 'memory/store.py', coverage: 88.5, lines_total: 890, lines_covered: 788, functions_total: 45, functions_covered: 40 },
        { name: 'brain', path: 'memory/brain.py', coverage: 82.1, lines_total: 1200, lines_covered: 985, functions_total: 68, functions_covered: 56 },
      ]
    },
    {
      name: 'web-ui',
      path: 'web-ui',
      coverage: 45.3,
      lines_total: 5600,
      lines_covered: 2537,
      functions_total: 340,
      functions_covered: 154,
    },
    {
      name: 'tests',
      path: 'tests',
      coverage: 92.1,
      lines_total: 4200,
      lines_covered: 3868,
      functions_total: 280,
      functions_covered: 258,
    },
  ]
}

// Generate trend data
function generateTrendData(): TrendData[] {
  const data: TrendData[] = []
  const today = new Date()
  let coverage = 68.5
  
  for (let i = 30; i >= 0; i--) {
    const date = new Date(today)
    date.setDate(date.getDate() - i)
    
    // Simulate gradual improvement
    coverage += Math.random() * 1.5 - 0.3
    coverage = Math.min(Math.max(coverage, 65), 85)
    
    data.push({
      date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      coverage: parseFloat(coverage.toFixed(1)),
      files_covered: Math.floor(coverage * 50),
      total_files: 4120,
    })
  }
  
  return data
}

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

export function Coverage() {
  const [coverageGaps, setCoverageGaps] = useState<CoverageGap[]>([])
  const [moduleCoverage, setModuleCoverage] = useState<ModuleCoverage[]>([])
  const [trendData, setTrendData] = useState<TrendData[]>([])
  const [loading, setLoading] = useState(true)
  const [filterSeverity, setFilterSeverity] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')

  useEffect(() => {
    // Simulate fetching coverage data
    const fetchData = () => {
      setCoverageGaps(generateCoverageGaps())
      setModuleCoverage(generateModuleCoverage())
      setTrendData(generateTrendData())
      setLoading(false)
    }
    
    fetchData()
    
    // Refresh every 30 seconds
    const interval = setInterval(fetchData, 30000)
    return () => clearInterval(interval)
  }, [])

  const stats = useMemo(() => {
    const totalLines = moduleCoverage.reduce((sum, m) => sum + m.lines_total, 0)
    const coveredLines = moduleCoverage.reduce((sum, m) => sum + m.lines_covered, 0)
    const overallCoverage = totalLines > 0 ? (coveredLines / totalLines) * 100 : 0
    
    const criticalGaps = coverageGaps.filter(g => g.severity === 'critical').length
    const highGaps = coverageGaps.filter(g => g.severity === 'high').length
    
    const avgComplexity = coverageGaps.length > 0
      ? coverageGaps.reduce((sum, g) => sum + g.complexity, 0) / coverageGaps.length
      : 0
    
    return {
      overallCoverage: overallCoverage.toFixed(1),
      totalFiles: moduleCoverage.reduce((sum, m) => sum + (m.children?.length || 1), 0),
      uncoveredFunctions: coverageGaps.length,
      criticalGaps,
      highGaps,
      avgComplexity: avgComplexity.toFixed(1),
    }
  }, [coverageGaps, moduleCoverage])

  const filteredGaps = useMemo(() => {
    let filtered = coverageGaps
    
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
    
    return filtered.slice(0, 10) // Top 10
  }, [coverageGaps, filterSeverity, searchQuery])

  const treemapData = useMemo(() => {
    return moduleCoverage.map(m => ({
      name: m.name,
      size: m.lines_total,
      coverage: m.coverage,
      children: m.children?.map(c => ({
        name: c.name,
        size: c.lines_total,
        coverage: c.coverage,
      }))
    }))
  }, [moduleCoverage])

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-500 bg-red-500/10'
      case 'high': return 'text-orange-500 bg-orange-500/10'
      case 'medium': return 'text-yellow-500 bg-yellow-500/10'
      case 'low': return 'text-blue-500 bg-blue-500/10'
      default: return 'text-gray-500 bg-gray-500/10'
    }
  }

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <AlertCircle className="w-4 h-4" />
      case 'high': return <AlertTriangle className="w-4 h-4" />
      case 'medium': return <Shield className="w-4 h-4" />
      case 'low': return <CheckCircle className="w-4 h-4" />
      default: return <Shield className="w-4 h-4" />
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <RefreshCw className="w-8 h-8 animate-spin text-primary" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Coverage Dashboard</h1>
          <p className="text-muted-foreground mt-1">
            Test coverage analysis and gap identification
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => window.location.reload()}
            className="flex items-center gap-2 px-4 py-2 bg-secondary text-secondary-foreground rounded-lg hover:bg-secondary/80 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
          <button className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors">
            <Download className="w-4 h-4" />
            Export Report
          </button>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatsCard
          title="Overall Coverage"
          value={`${stats.overallCoverage}%`}
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
          title="Avg Complexity"
          value={stats.avgComplexity}
          description="Of uncovered functions"
          icon={GitBranch}
        />
        <StatsCard
          title="Lines Covered"
          value={`${(parseFloat(stats.overallCoverage) * 50).toFixed(0)}K`}
          description="Out of ~200K total lines"
          icon={FileCode}
          trend={{ value: 2.8, isPositive: true }}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Coverage Heatmap */}
        <div className="bg-card rounded-xl border border-border p-6">
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

        {/* Coverage Trend */}
        <div className="bg-card rounded-xl border border-border p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Coverage Trend (30 Days)</h3>
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-green-500" />
              <span className="text-sm text-green-500">+5.2%</span>
            </div>
          </div>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: 'var(--muted-foreground)', fontSize: 11 }}
                  tickMargin={8}
                  interval={4}
                />
                <YAxis
                  tick={{ fill: 'var(--muted-foreground)', fontSize: 11 }}
                  domain={[60, 90]}
                  tickFormatter={(value) => `${value}%`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--card)',
                    border: '1px solid var(--border)',
                    borderRadius: '8px',
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="coverage"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Coverage Gaps Table */}
      <div className="bg-card rounded-xl border border-border p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold">Top 10 Uncovered Functions</h3>
            <p className="text-sm text-muted-foreground">
              Functions with highest impact scores requiring test coverage
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
                className="pl-9 pr-4 py-2 bg-secondary rounded-lg text-sm w-64 focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
            
            {/* Severity Filter */}
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-muted-foreground" />
              <select
                value={filterSeverity}
                onChange={(e) => setFilterSeverity(e.target.value)}
                className="px-3 py-2 bg-secondary rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
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
              <tr className="border-b border-border">
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
              {filteredGaps.map((gap) => (
                <tr
                  key={gap.id}
                  className="border-b border-border last:border-0 hover:bg-secondary/50 transition-colors"
                >
                  <td className="py-3 px-4">
                    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${getSeverityColor(gap.severity)}`}>
                      {getSeverityIcon(gap.severity)}
                      {gap.severity}
                    </span>
                  </td>
                  <td className="py-3 px-4 font-medium">{gap.function_name}</td>
                  <td className="py-3 px-4 text-sm text-muted-foreground font-mono">{gap.file_path}</td>
                  <td className="py-3 px-4 text-center text-sm">{gap.line_number}</td>
                  <td className="py-3 px-4 text-center">
                    <span className={`inline-flex items-center justify-center w-8 h-8 rounded-full text-xs font-medium ${
                      gap.complexity > 15 ? 'bg-red-500/10 text-red-500' :
                      gap.complexity > 10 ? 'bg-yellow-500/10 text-yellow-500' :
                      'bg-green-500/10 text-green-500'
                    }`}>
                      {gap.complexity}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-center">
                    <div className="flex items-center justify-center gap-2">
                      <div className="w-16 h-2 bg-secondary rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            gap.impact_score > 80 ? 'bg-red-500' :
                            gap.impact_score > 60 ? 'bg-orange-500' :
                            gap.impact_score > 40 ? 'bg-yellow-500' :
                            'bg-blue-500'
                          }`}
                          style={{ width: `${gap.impact_score}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium">{gap.impact_score.toFixed(0)}</span>
                    </div>
                  </td>
                  <td className="py-3 px-4 text-sm text-muted-foreground">{gap.reason}</td>
                </tr>
              ))}
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
      <div className="bg-card rounded-xl border border-border p-6">
        <h3 className="text-lg font-semibold mb-4">Module Coverage Breakdown</h3>
        <div className="h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={moduleCoverage}
              layout="vertical"
              margin={{ left: 20, right: 30, top: 10, bottom: 10 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
              <XAxis
                type="number"
                domain={[0, 100]}
                tickFormatter={(value) => `${value}%`}
                tick={{ fill: 'var(--muted-foreground)', fontSize: 11 }}
              />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fill: 'var(--muted-foreground)', fontSize: 12 }}
                width={80}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'var(--card)',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                }}
                formatter={(value: number) => [`${value.toFixed(1)}%`, 'Coverage']}
              />
              <Bar dataKey="coverage" radius={[0, 4, 4, 0]}>
                {moduleCoverage.map((entry, index) => (
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
