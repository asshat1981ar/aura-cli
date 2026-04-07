import { useMemo } from 'react'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts'
import { useTelemetryStore } from '@/stores/telemetryStore'

const COLORS = ['#10b981', '#f59e0b', '#ef4444', '#6366f1', '#8b5cf6']

export function LatencyChart() {
  const { summary } = useTelemetryStore()

  const data = useMemo(() => {
    if (!summary?.by_hour) return []
    return summary.by_hour.map(h => ({
      time: h.hour.split('T')[1] || h.hour,
      latency: h.avg_latency,
      count: h.count
    }))
  }, [summary])

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-muted-foreground">
        <p className="text-sm">No data available</p>
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
        <XAxis dataKey="time" tick={{ fontSize: 10 }} />
        <YAxis tick={{ fontSize: 10 }} />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: 'var(--background)',
            border: '1px solid var(--border)',
            borderRadius: '6px'
          }}
        />
        <Line 
          type="monotone" 
          dataKey="latency" 
          stroke="#6366f1" 
          strokeWidth={2}
          dot={{ r: 3 }}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

export function AgentDistributionChart() {
  const { summary } = useTelemetryStore()

  const data = useMemo(() => {
    if (!summary?.by_agent) return []
    return Object.entries(summary.by_agent)
      .map(([name, stats]) => ({
        name: name.split('-').pop() || name,
        value: stats.count
      }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 5)
  }, [summary])

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-muted-foreground">
        <p className="text-sm">No data available</p>
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={40}
          outerRadius={70}
          paddingAngle={5}
          dataKey="value"
        >
          {data.map((_, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip 
          contentStyle={{ 
            backgroundColor: 'var(--background)',
            border: '1px solid var(--border)',
            borderRadius: '6px'
          }}
        />
      </PieChart>
    </ResponsiveContainer>
  )
}

export function RequestVolumeChart() {
  const { summary } = useTelemetryStore()

  const data = useMemo(() => {
    if (!summary?.by_hour) return []
    return summary.by_hour.map(h => ({
      time: h.hour.split('T')[1] || h.hour,
      requests: h.count,
      errors: h.errors
    }))
  }, [summary])

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-muted-foreground">
        <p className="text-sm">No data available</p>
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
        <XAxis dataKey="time" tick={{ fontSize: 10 }} />
        <YAxis tick={{ fontSize: 10 }} />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: 'var(--background)',
            border: '1px solid var(--border)',
            borderRadius: '6px'
          }}
        />
        <Bar dataKey="requests" fill="#10b981" radius={[4, 4, 0, 0]} />
        <Bar dataKey="errors" fill="#ef4444" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

export function SuccessRateChart() {
  const { summary } = useTelemetryStore()

  const data = useMemo(() => {
    if (!summary?.by_agent) return []
    return Object.entries(summary.by_agent)
      .map(([name, stats]) => ({
        name: name.split('-').pop() || name,
        rate: stats.success_rate
      }))
      .slice(0, 5)
  }, [summary])

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-muted-foreground">
        <p className="text-sm">No data available</p>
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} layout="vertical">
        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
        <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 10 }} />
        <YAxis dataKey="name" type="category" tick={{ fontSize: 10 }} width={60} />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: 'var(--background)',
            border: '1px solid var(--border)',
            borderRadius: '6px'
          }}
          formatter={(value: number) => `${value}%`}
        />
        <Bar 
          dataKey="rate" 
          fill="#10b981" 
          radius={[0, 4, 4, 0]}
          background={{ fill: '#f3f4f6' }}
        />
      </BarChart>
    </ResponsiveContainer>
  )
}
