/**
 * GoalQueue.tsx — Sprint 7
 *
 * Renders a live table of goals (pending / in-progress / done).
 * Features:
 *  • Polls status every 10 s via GET /webhook/status/:id or POST /call goal_status
 *  • Add goal   → POST /webhook/goal   { goal: text }
 *  • Pause goal → POST /call           { tool: "goal_pause", args: { id } }
 *  • Delete goal → POST /call          { tool: "goal_cancel", args: { id } }
 */

import { useEffect, useState, useCallback, useRef } from 'react'
import { Plus, Pause, Trash2, RefreshCw, Target, Play } from 'lucide-react'

// ── Types ─────────────────────────────────────────────────────────────────────

type GoalStatus = 'pending' | 'running' | 'paused' | 'done' | 'failed'

interface Goal {
  id: string
  description: string
  status: GoalStatus
  createdAt: string
  updatedAt?: string
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function authHeaders(): Record<string, string> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  try {
    const stored = localStorage.getItem('aura-auth')
    if (stored) {
      const token = JSON.parse(stored)?.state?.token
      if (token) headers['Authorization'] = `Bearer ${token}`
    }
  } catch { /* ignore */ }
  return headers
}

async function fetchGoals(): Promise<Goal[]> {
  // Try the goals listing endpoint
  for (const url of ['/api/goals', '/goals']) {
    try {
      const res = await fetch(url, { headers: authHeaders() })
      if (res.ok) {
        const data = await res.json()
        if (Array.isArray(data)) return data
        if (Array.isArray(data.goals)) return data.goals
      }
    } catch { /* next */ }
  }
  return []
}

async function addGoal(text: string): Promise<boolean> {
  // Primary path: /webhook/goal
  try {
    const res = await fetch('/webhook/goal', {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({ goal: text }),
    })
    if (res.ok) return true
  } catch { /* fall through */ }

  // Fallback: /execute tool call
  try {
    const res = await fetch('/execute', {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({ tool: 'goal_add', args: { goal: text } }),
    })
    return res.ok
  } catch {
    return false
  }
}

async function pauseGoal(id: string): Promise<boolean> {
  try {
    const res = await fetch('/execute', {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({ tool: 'goal_pause', args: { id } }),
    })
    return res.ok
  } catch {
    return false
  }
}

async function deleteGoal(id: string): Promise<boolean> {
  try {
    const res = await fetch('/execute', {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({ tool: 'goal_cancel', args: { id } }),
    })
    return res.ok
  } catch {
    return false
  }
}

// ── StatusBadge ───────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: GoalStatus }) {
  const map: Record<GoalStatus, { label: string; cls: string }> = {
    pending:  { label: 'Pending',     cls: 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300' },
    running:  { label: 'Running',     cls: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' },
    paused:   { label: 'Paused',      cls: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' },
    done:     { label: 'Done',        cls: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' },
    failed:   { label: 'Failed',      cls: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' },
  }
  const { label, cls } = map[status] ?? map.pending
  return (
    <span
      role="status"
      aria-label={`Goal status: ${label}`}
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${cls}`}
    >
      {label}
    </span>
  )
}

// ── GoalQueueView ─────────────────────────────────────────────────────────────

export function GoalQueueView() {
  const [goals, setGoals] = useState<Goal[]>([])
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [input, setInput] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [statusFilter, setStatusFilter] = useState<GoalStatus | 'all'>('all')
  const inputRef = useRef<HTMLInputElement>(null)

  const reload = useCallback(async () => {
    try {
      const data = await fetchGoals()
      setGoals(data)
      setError(null)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }, [])

  // Poll every 10 s
  useEffect(() => {
    reload()
    const id = setInterval(reload, 10_000)
    return () => clearInterval(id)
  }, [reload])

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return
    setSubmitting(true)
    const ok = await addGoal(input.trim())
    if (ok) {
      setInput('')
      await reload()
    } else {
      setError('Failed to add goal — check server connection')
    }
    setSubmitting(false)
    inputRef.current?.focus()
  }

  const handlePause = async (goal: Goal) => {
    const isPaused = goal.status === 'paused'
    // Optimistic UI update
    setGoals((prev) =>
      prev.map((g) =>
        g.id === goal.id ? { ...g, status: isPaused ? 'running' : 'paused' } : g
      )
    )
    await pauseGoal(goal.id)
    await reload()
  }

  const handleDelete = async (id: string) => {
    // Optimistic remove
    setGoals((prev) => prev.filter((g) => g.id !== id))
    await deleteGoal(id)
    await reload()
  }

  const filtered = statusFilter === 'all'
    ? goals
    : goals.filter((g) => g.status === statusFilter)

  const counts = goals.reduce<Record<string, number>>((acc, g) => {
    acc[g.status] = (acc[g.status] ?? 0) + 1
    return acc
  }, {})

  return (
    <main aria-labelledby="goalqueue-heading" className="space-y-6 p-6">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <Target className="w-7 h-7 text-primary" aria-hidden="true" />
          <div>
            <h1 id="goalqueue-heading" className="text-2xl font-bold tracking-tight">
              Goal Queue
            </h1>
            <p className="text-sm text-muted-foreground">
              Add, pause, and manage AURA goals — refreshes every 10&nbsp;s
            </p>
          </div>
        </div>
        <button
          onClick={() => { setLoading(true); reload() }}
          disabled={loading}
          aria-label="Refresh goal queue"
          className="flex items-center gap-2 px-4 py-2 border rounded-lg bg-card hover:bg-accent transition-colors disabled:opacity-50 text-sm"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} aria-hidden="true" />
          Refresh
        </button>
      </header>

      {/* ── Add Goal Form ──────────────────────────────────────────────────── */}
      <form
        onSubmit={handleAdd}
        aria-label="Add new goal"
        className="flex gap-3"
      >
        <label htmlFor="goal-input" className="sr-only">
          New goal description
        </label>
        <input
          id="goal-input"
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Describe a new goal…"
          className="flex-1 px-4 py-2 border rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-primary text-sm"
          disabled={submitting}
          required
        />
        <button
          type="submit"
          disabled={submitting || !input.trim()}
          aria-label="Submit new goal"
          className="flex items-center gap-2 px-5 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 text-sm"
        >
          {submitting
            ? <RefreshCw className="w-4 h-4 animate-spin" aria-hidden="true" />
            : <Plus className="w-4 h-4" aria-hidden="true" />}
          Add Goal
        </button>
      </form>

      {/* ── Error ──────────────────────────────────────────────────────────── */}
      {error && (
        <div role="alert" className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* ── Summary + Filter ──────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-3">
        {(['all', 'pending', 'running', 'paused', 'done', 'failed'] as const).map((s) => (
          <button
            key={s}
            onClick={() => setStatusFilter(s)}
            aria-pressed={statusFilter === s}
            className={`px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
              statusFilter === s
                ? 'bg-primary text-primary-foreground border-primary'
                : 'bg-card hover:bg-accent border-border'
            }`}
          >
            {s === 'all' ? `All (${goals.length})` : `${s.charAt(0).toUpperCase() + s.slice(1)} (${counts[s] ?? 0})`}
          </button>
        ))}
      </div>

      {/* ── Table ──────────────────────────────────────────────────────────── */}
      <div className="rounded-xl border bg-card overflow-x-auto">
        <table
          className="w-full text-sm"
          aria-label="Goal queue table"
        >
          <thead>
            <tr className="bg-muted/50 border-b">
              <th scope="col" className="px-4 py-3 text-left font-medium text-muted-foreground">Goal</th>
              <th scope="col" className="px-4 py-3 text-left font-medium text-muted-foreground">Status</th>
              <th scope="col" className="px-4 py-3 text-left font-medium text-muted-foreground">Created</th>
              <th scope="col" className="px-4 py-3 text-right font-medium text-muted-foreground">Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading && filtered.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-4 py-10 text-center text-muted-foreground">
                  <RefreshCw className="w-5 h-5 animate-spin mx-auto mb-2" aria-hidden="true" />
                  Loading goals…
                </td>
              </tr>
            ) : filtered.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-4 py-10 text-center text-muted-foreground">
                  No goals found. Add one above!
                </td>
              </tr>
            ) : (
              filtered.map((goal) => (
                <tr
                  key={goal.id}
                  className="border-b last:border-0 hover:bg-muted/30 transition-colors"
                >
                  <td className="px-4 py-3 max-w-xs">
                    <div className="font-medium truncate" title={goal.description}>
                      {goal.description}
                    </div>
                    <div className="text-xs text-muted-foreground font-mono">{goal.id}</div>
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={goal.status} />
                  </td>
                  <td className="px-4 py-3 text-muted-foreground tabular-nums text-xs">
                    {goal.createdAt
                      ? new Date(goal.createdAt).toLocaleString()
                      : '—'}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex justify-end gap-2">
                      {/* Pause / Resume */}
                      {(goal.status === 'running' || goal.status === 'paused') && (
                        <button
                          onClick={() => handlePause(goal)}
                          aria-label={`${goal.status === 'paused' ? 'Resume' : 'Pause'} goal: ${goal.description}`}
                          className="p-1.5 rounded-md border hover:bg-accent transition-colors text-muted-foreground hover:text-foreground"
                        >
                          {goal.status === 'paused'
                            ? <Play className="w-3.5 h-3.5" aria-hidden="true" />
                            : <Pause className="w-3.5 h-3.5" aria-hidden="true" />}
                        </button>
                      )}
                      {/* Delete */}
                      <button
                        onClick={() => handleDelete(goal.id)}
                        aria-label={`Delete goal: ${goal.description}`}
                        className="p-1.5 rounded-md border border-transparent hover:border-red-300 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors text-muted-foreground hover:text-red-600"
                      >
                        <Trash2 className="w-3.5 h-3.5" aria-hidden="true" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </main>
  )
}

export default GoalQueueView
