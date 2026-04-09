import { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

interface DailyStat {
  date: string
  total: number
  completed: number
  success_rate: number
}

interface TrendData {
  period_days: number
  total_goals: number
  success_rate: number
  trend: string
  message: string
  daily_breakdown: DailyStat[]
}

export function GoalTrends() {
  const [data, setData] = useState<TrendData | null>(null)
  const [loading, setLoading] = useState(true)
  const [days, setDays] = useState(7)

  const fetchTrends = async () => {
    try {
      const response = await fetch(`/api/analytics/goal-trends?days=${days}`)
      if (response.ok) {
        const data = await response.json()
        setData(data)
      }
    } catch (error) {
      console.error('Failed to fetch trends:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchTrends()
  }, [days])

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6 animate-pulse">
        <div className="h-6 bg-gray-200 rounded w-32 mb-4"></div>
        <div className="h-64 bg-gray-200 rounded"></div>
      </div>
    )
  }

  if (!data || data.daily_breakdown.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Goal Trends</h3>
        <p className="text-gray-500">No trend data available yet.</p>
      </div>
    )
  }

  const chartData = data.daily_breakdown.map((day) => ({
    date: formatDate(day.date),
    fullDate: day.date,
    total: day.total,
    completed: day.completed,
    successRate: Math.round(day.success_rate * 100),
  }))

  const trendColor = data.trend === 'improving' ? 'text-green-600' : 
                     data.trend === 'declining' ? 'text-red-600' : 'text-amber-600'
  const trendBg = data.trend === 'improving' ? 'bg-green-50' : 
                  data.trend === 'declining' ? 'bg-red-50' : 'bg-amber-50'

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold">Goal Trends</h3>
          <p className="text-sm text-gray-500">{data.message}</p>
        </div>
        <div className="flex items-center gap-4">
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${trendBg} ${trendColor}`}>
            {data.trend === 'improving' ? '↗️ Improving' : 
             data.trend === 'declining' ? '↘️ Declining' : '➡️ Stable'}
          </span>
          <select
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            className="text-sm border border-gray-300 rounded px-2 py-1"
          >
            <option value={7}>Last 7 days</option>
            <option value={14}>Last 14 days</option>
            <option value={30}>Last 30 days</option>
          </select>
        </div>
      </div>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis yAxisId="left" />
            <YAxis yAxisId="right" orientation="right" domain={[0, 100]} />
            <Tooltip
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-white p-3 border rounded shadow">
                      <p className="font-medium">{label}</p>
                      <p className="text-blue-600">Total: {payload[0].value}</p>
                      <p className="text-green-600">Completed: {payload[1].value}</p>
                      <p className="text-purple-600">Success Rate: {payload[2].value}%</p>
                    </div>
                  )
                }
                return null
              }}
            />
            <Legend />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="total"
              name="Total Goals"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ fill: '#3b82f6' }}
            />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="completed"
              name="Completed"
              stroke="#10b981"
              strokeWidth={2}
              dot={{ fill: '#10b981' }}
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="successRate"
              name="Success Rate %"
              stroke="#8b5cf6"
              strokeWidth={2}
              dot={{ fill: '#8b5cf6' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-4 text-center">
        <div className="p-3 bg-gray-50 rounded">
          <p className="text-2xl font-bold text-blue-600">{data.total_goals}</p>
          <p className="text-sm text-gray-600">Total Goals</p>
        </div>
        <div className="p-3 bg-gray-50 rounded">
          <p className="text-2xl font-bold text-green-600">{(data.success_rate * 100).toFixed(1)}%</p>
          <p className="text-sm text-gray-600">Success Rate</p>
        </div>
        <div className="p-3 bg-gray-50 rounded">
          <p className="text-2xl font-bold text-purple-600">{chartData.length}</p>
          <p className="text-sm text-gray-600">Days Tracked</p>
        </div>
      </div>
    </div>
  )
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr)
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}
