import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
} from 'recharts'

interface GoalChartData {
  date: string
  completed: number
  failed: number
  pending: number
}

interface AgentChartData {
  name: string
  tasks: number
}

const COLORS = {
  completed: '#22c55e',
  failed: '#ef4444',
  pending: '#eab308',
  running: '#3b82f6',
  idle: '#6b7280',
}

export function GoalTrendChart({ data }: { data: GoalChartData[] }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="colorCompleted" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={COLORS.completed} stopOpacity={0.8} />
            <stop offset="95%" stopColor={COLORS.completed} stopOpacity={0} />
          </linearGradient>
          <linearGradient id="colorFailed" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={COLORS.failed} stopOpacity={0.8} />
            <stop offset="95%" stopColor={COLORS.failed} stopOpacity={0} />
          </linearGradient>
        </defs>
        <XAxis dataKey="date" />
        <YAxis />
        <CartesianGrid strokeDasharray="3 3" />
        <Tooltip
          contentStyle={{
            backgroundColor: 'var(--card)',
            border: '1px solid var(--border)',
            borderRadius: '8px',
          }}
        />
        <Area
          type="monotone"
          dataKey="completed"
          stroke={COLORS.completed}
          fillOpacity={1}
          fill="url(#colorCompleted)"
          name="Completed"
        />
        <Area
          type="monotone"
          dataKey="failed"
          stroke={COLORS.failed}
          fillOpacity={1}
          fill="url(#colorFailed)"
          name="Failed"
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}

export function AgentPerformanceChart({ data }: { data: AgentChartData[] }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip
          contentStyle={{
            backgroundColor: 'var(--card)',
            border: '1px solid var(--border)',
            borderRadius: '8px',
          }}
        />
        <Bar dataKey="tasks" fill={COLORS.running} name="Tasks Completed" />
      </BarChart>
    </ResponsiveContainer>
  )
}

export function GoalDistributionChart({
  pending,
  running,
  completed,
  failed,
}: {
  pending: number
  running: number
  completed: number
  failed: number
}) {
  const data = [
    { name: 'Pending', value: pending, color: COLORS.pending },
    { name: 'Running', value: running, color: COLORS.running },
    { name: 'Completed', value: completed, color: COLORS.completed },
    { name: 'Failed', value: failed, color: COLORS.failed },
  ].filter((d) => d.value > 0)

  return (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={60}
          outerRadius={100}
          paddingAngle={5}
          dataKey="value"
          label={({ name, percent }) =>
            `${name}: ${(percent * 100).toFixed(0)}%`
          }
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{
            backgroundColor: 'var(--card)',
            border: '1px solid var(--border)',
            borderRadius: '8px',
          }}
        />
      </PieChart>
    </ResponsiveContainer>
  )
}

export function ExecutionTimeChart({
  data,
}: {
  data: { time: string; duration: number }[]
}) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="time" />
        <YAxis />
        <Tooltip
          contentStyle={{
            backgroundColor: 'var(--card)',
            border: '1px solid var(--border)',
            borderRadius: '8px',
          }}
          formatter={(value: number) => [`${value.toFixed(1)}s`, 'Duration']}
        />
        <Line
          type="monotone"
          dataKey="duration"
          stroke={COLORS.running}
          strokeWidth={2}
          dot={{ fill: COLORS.running }}
          name="Execution Time"
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
