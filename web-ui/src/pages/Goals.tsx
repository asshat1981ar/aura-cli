import { useEffect, useState } from 'react'
import { Plus, Search, Trash2, RotateCcw } from 'lucide-react'
import { useGoalStore } from '../stores/goalStore'
import { StatusBadge } from '../components/StatusBadge'

export function Goals() {
  const { goals, fetchGoals, addGoal, cancelGoal, isLoading } = useGoalStore()
  const [newGoal, setNewGoal] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')

  useEffect(() => {
    fetchGoals()
    const interval = setInterval(fetchGoals, 3000)
    return () => clearInterval(interval)
  }, [fetchGoals])

  const handleAddGoal = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newGoal.trim()) return
    await addGoal(newGoal)
    setNewGoal('')
  }

  const filteredGoals = goals.filter((goal) => {
    const matchesSearch = goal.description
      .toLowerCase()
      .includes(searchQuery.toLowerCase())
    const matchesStatus =
      statusFilter === 'all' || goal.status === statusFilter
    return matchesSearch && matchesStatus
  })

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Goals</h2>
          <p className="text-muted-foreground">
            Manage and monitor your AURA goals.
          </p>
        </div>
        <button
          onClick={() => fetchGoals()}
          className="flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-accent transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      <form onSubmit={handleAddGoal} className="flex gap-4">
        <input
          type="text"
          value={newGoal}
          onChange={(e) => setNewGoal(e.target.value)}
          placeholder="Enter a new goal..."
          className="flex-1 px-4 py-2 border rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-primary"
        />
        <button
          type="submit"
          disabled={isLoading || !newGoal.trim()}
          className="flex items-center gap-2 px-6 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors disabled:opacity-50"
        >
          <Plus className="w-4 h-4" />
          Add Goal
        </button>
      </form>

      <div className="flex gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search goals..."
            className="w-full pl-10 pr-4 py-2 border rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-primary"
          />
        </div>
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="px-4 py-2 border rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-primary"
        >
          <option value="all">All Status</option>
          <option value="pending">Pending</option>
          <option value="running">Running</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
        </select>
      </div>

      <div className="bg-card border rounded-xl overflow-hidden">
        <table className="w-full">
          <thead className="bg-muted/50">
            <tr>
              <th className="px-6 py-3 text-left text-sm font-medium">
                Description
              </th>
              <th className="px-6 py-3 text-left text-sm font-medium">
                Status
              </th>
              <th className="px-6 py-3 text-left text-sm font-medium">
                Progress
              </th>
              <th className="px-6 py-3 text-left text-sm font-medium">
                Created
              </th>
              <th className="px-6 py-3 text-right text-sm font-medium">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {filteredGoals.map((goal) => (
              <tr key={goal.id} className="hover:bg-muted/50">
                <td className="px-6 py-4">
                  <p className="font-medium truncate max-w-md">
                    {goal.description}
                  </p>
                  <p className="text-xs text-muted-foreground">ID: {goal.id}</p>
                </td>
                <td className="px-6 py-4">
                  <StatusBadge status={goal.status} />
                </td>
                <td className="px-6 py-4">
                  {goal.progress !== undefined ? (
                    <div className="flex items-center gap-2">
                      <div className="w-24 h-2 bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full bg-primary transition-all"
                          style={{ width: `${goal.progress}%` }}
                        />
                      </div>
                      <span className="text-sm text-muted-foreground">
                        {goal.progress}%
                      </span>
                    </div>
                  ) : (
                    <span className="text-sm text-muted-foreground">
                      {goal.cycles || 0} / {goal.max_cycles || '?'} cycles
                    </span>
                  )}
                </td>
                <td className="px-6 py-4 text-sm text-muted-foreground">
                  {new Date(goal.created_at).toLocaleString()}
                </td>
                <td className="px-6 py-4 text-right">
                  {goal.status === 'running' && (
                    <button
                      onClick={() => cancelGoal(goal.id)}
                      className="p-2 text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors"
                      title="Cancel goal"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {filteredGoals.length === 0 && (
          <div className="text-center py-12 text-muted-foreground">
            {searchQuery || statusFilter !== 'all'
              ? 'No goals match your filters.'
              : 'No goals yet. Create one to get started.'}
          </div>
        )}
      </div>
    </div>
  )
}
