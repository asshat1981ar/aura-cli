/**
 * AgentObservatory.tsx — Sprint 7
 *
 * Polls /tools every 5 s and renders a live status table of all registered
 * AURA tools / agents with colour-coded status badges.
 *
 * Status mapping:
 *   idle    → green  badge
 *   running → yellow badge
 *   error   → red    badge
 */

import { useEffect, useState, useCallback } from 'react'
import { RefreshCw, Bot } from 'lucide-react'

// ── Types ────────────────────────────────────────────────────────────────────

type AgentStatus = 'idle' | 'running' | 'error'

interface AgentRow {
  name: string
  status: AgentStatus
  lastRun: string
  callCount: number
  description?: string
}

interface ToolDescriptor {
  name: string
  description?: string
  status?: AgentStatus
  last_run?: string
  call_count?: number
  [key: string]: unknown
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function deriveStatus(tool: ToolDescriptor): AgentStatus {
  if (tool.status === 'error') return 'error'
  if (tool.status === 'running') return 'running'
  return 'idle'
}

function formatTime(iso?: string): string {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleTimeString()
  } catch {
    return iso
  }
}

// ── StatusBadge ──────────────────────────────────────────────────────────────

interface StatusBadgeProps {
  status: AgentStatus
}

function StatusBadge({ status }: StatusBadgeProps) {
  const config: Record<AgentStatus, { label: string; classes: string }> = {
    idle:    { label: 'Idle',    classes: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' },
    running: { label: 'Running', classes: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400' },
    error:   { label: 'Error',   classes: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400' },
  }
  const { label, classes } = config[status]
  return (
    <span
      role="status"
      aria-label={`Agent status: ${label}`}
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium ${classes}`}
    >
      <span className="w-1.5 h-1.5 rounded-full bg-current" aria-hidden="true" />
      {label}
    </span>
  )
}

// ── AgentObservatory ─────────────────────────────────────────────────────────

export function AgentObservatory() {
  const [agents, setAgents] = useState<AgentRow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  const fetchTools = useCallback(async () => {
    try {
      const token = localStorage.getItem('aura-auth')
        ? JSON.parse(localStorage.getItem('aura-auth')!).state?.token
        : null

      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (token) headers['Authorization'] = `Bearer ${token}`

      const res = await fetch('/tools', { headers })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const data = await res.json()

      // The /tools endpoint may return { tools: [...] } or an array
      const toolList: ToolDescriptor[] = Array.isArray(data)
        ? data
        : (data.tools ?? Object.values(data))

      const rows: AgentRow[] = toolList.map((t) => ({
        name:       t.name ?? 'unknown',
        status:     deriveStatus(t),
        lastRun:    formatTime(t.last_run),
        callCount:  typeof t.call_count === 'number' ? t.call_count : 0,
        description: t.description,
      }))

      setAgents(rows)
      setLastUpdated(new Date())
      setError(null)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }, [])

  // Poll every 5 s
  useEffect(() => {
    fetchTools()
    const id = setInterval(fetchTools, 5_000)
    return () => clearInterval(id)
  }, [fetchTools])

  // ── Render ─────────────────────────────────────────────────────────────────

  const idleCount   = agents.filter((a) => a.status === 'idle').length
  const runningCount = agents.filter((a) => a.status === 'running').length
  const errorCount  = agents.filter((a) => a.status === 'error').length

  return (
    <main aria-labelledby="observatory-heading" className="space-y-6 p-6">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <Bot className="w-7 h-7 text-primary" aria-hidden="true" />
          <div>
            <h1
              id="observatory-heading"
              className="text-2xl font-bold tracking-tight"
            >
              Agent Observatory
            </h1>
            <p className="text-sm text-muted-foreground">
              Live status of all registered AURA tools — refreshes every&nbsp;5&nbsp;s
            </p>
          </div>
        </div>

        <button
          onClick={() => { setLoading(true); fetchTools() }}
          disabled={loading}
          aria-label="Refresh agent status"
          className="flex items-center gap-2 px-4 py-2 border rounded-lg bg-card hover:bg-accent transition-colors disabled:opacity-50 text-sm"
        >
          <RefreshCw
            className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`}
            aria-hidden="true"
          />
          Refresh
        </button>
      </header>

      {/* ── Summary Badges ─────────────────────────────────────────────────── */}
      <section aria-label="Agent status summary" className="flex flex-wrap gap-3">
        <SummaryCard label="Total"   value={agents.length}  color="text-foreground" />
        <SummaryCard label="Idle"    value={idleCount}      color="text-green-600 dark:text-green-400" />
        <SummaryCard label="Running" value={runningCount}   color="text-yellow-600 dark:text-yellow-400" />
        <SummaryCard label="Error"   value={errorCount}     color="text-red-600 dark:text-red-400" />
        {lastUpdated && (
          <p className="ml-auto self-center text-xs text-muted-foreground">
            Last updated: {lastUpdated.toLocaleTimeString()}
          </p>
        )}
      </section>

      {/* ── Error Banner ───────────────────────────────────────────────────── */}
      {error && (
        <div
          role="alert"
          className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 text-sm"
        >
          Failed to fetch tools: {error}
        </div>
      )}

      {/* ── Table ──────────────────────────────────────────────────────────── */}
      <div className="rounded-xl border bg-card overflow-x-auto">
        <table
          className="w-full text-sm"
          aria-label="Agent observatory status table"
        >
          <thead>
            <tr className="bg-muted/50 border-b">
              <th scope="col" className="px-4 py-3 text-left font-medium text-muted-foreground">
                Agent / Tool
              </th>
              <th scope="col" className="px-4 py-3 text-left font-medium text-muted-foreground">
                Status
              </th>
              <th scope="col" className="px-4 py-3 text-left font-medium text-muted-foreground">
                Last Run
              </th>
              <th scope="col" className="px-4 py-3 text-right font-medium text-muted-foreground">
                Call&nbsp;Count
              </th>
            </tr>
          </thead>
          <tbody>
            {loading && agents.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-4 py-10 text-center text-muted-foreground">
                  <RefreshCw className="w-5 h-5 animate-spin mx-auto mb-2" aria-hidden="true" />
                  Loading agents…
                </td>
              </tr>
            ) : agents.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-4 py-10 text-center text-muted-foreground">
                  No agents registered
                </td>
              </tr>
            ) : (
              agents.map((agent) => (
                <tr
                  key={agent.name}
                  className="border-b last:border-0 hover:bg-muted/30 transition-colors"
                >
                  <td className="px-4 py-3 font-medium">
                    <div>{agent.name}</div>
                    {agent.description && (
                      <div className="text-xs text-muted-foreground mt-0.5 max-w-xs truncate">
                        {agent.description}
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={agent.status} />
                  </td>
                  <td className="px-4 py-3 tabular-nums text-muted-foreground">
                    {agent.lastRun}
                  </td>
                  <td className="px-4 py-3 tabular-nums text-right">
                    {agent.callCount.toLocaleString()}
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

// ── SummaryCard ───────────────────────────────────────────────────────────────

interface SummaryCardProps {
  label: string
  value: number
  color: string
}

function SummaryCard({ label, value, color }: SummaryCardProps) {
  return (
    <div className="flex items-baseline gap-2 px-4 py-2 rounded-lg border bg-card">
      <span className={`text-2xl font-bold tabular-nums ${color}`}>{value}</span>
      <span className="text-sm text-muted-foreground">{label}</span>
    </div>
  )
}

export default AgentObservatory
