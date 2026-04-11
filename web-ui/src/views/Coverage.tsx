/**
 * Coverage.tsx — Sprint 7
 *
 * Displays overall test-coverage percentage and a per-module breakdown.
 *
 * Data sources (tried in order):
 *   1. GET /api/coverage       — JSON from backend
 *   2. GET /coverage.json      — static file shipped with the build
 *   3. Built-in placeholder    — so the page always renders
 *
 * Progress bar colours:
 *   ≥80 %  → green
 *   60–79% → yellow
 *   <60 %  → red
 */

import { useEffect, useState, useCallback } from 'react'
import { RefreshCw, Shield } from 'lucide-react'

// ── Types ─────────────────────────────────────────────────────────────────────

interface ModuleCoverage {
  module: string
  coverage: number        // 0–100
  lines: number
  coveredLines: number
}

interface CoverageData {
  overall: number         // 0–100
  modules: ModuleCoverage[]
  timestamp?: string
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const PLACEHOLDER: CoverageData = {
  overall: 72,
  modules: [
    { module: 'core/',             coverage: 85, lines: 400, coveredLines: 340 },
    { module: 'aura_cli/',         coverage: 78, lines: 300, coveredLines: 234 },
    { module: 'agents/',           coverage: 65, lines: 200, coveredLines: 130 },
    { module: 'tools/',            coverage: 90, lines: 150, coveredLines: 135 },
    { module: 'experimental/',     coverage: 40, lines: 100, coveredLines: 40  },
  ],
}

async function loadCoverage(): Promise<CoverageData> {
  for (const url of ['/api/coverage', '/coverage.json', '/coverage_data.json']) {
    try {
      const res = await fetch(url)
      if (res.ok) {
        const raw = await res.json()
        // Normalise different JSON shapes
        if (typeof raw.overall === 'number') return raw as CoverageData
        if (typeof raw.totals?.percent_covered === 'number') {
          return {
            overall: raw.totals.percent_covered,
            modules: Object.entries(raw.files ?? {}).map(([k, v]: [string, any]) => ({
              module: k,
              coverage: v.summary?.percent_covered ?? 0,
              lines: v.summary?.num_statements ?? 0,
              coveredLines: v.summary?.covered_lines ?? 0,
            })),
          }
        }
      }
    } catch { /* next */ }
  }
  return PLACEHOLDER
}

function coverageColor(pct: number): string {
  if (pct >= 80) return 'bg-green-500'
  if (pct >= 60) return 'bg-yellow-500'
  return 'bg-red-500'
}

function coverageTextColor(pct: number): string {
  if (pct >= 80) return 'text-green-700 dark:text-green-400'
  if (pct >= 60) return 'text-yellow-700 dark:text-yellow-400'
  return 'text-red-700 dark:text-red-400'
}

// ── ProgressBar ───────────────────────────────────────────────────────────────

interface ProgressBarProps {
  value: number           // 0–100
  label: string
  size?: 'sm' | 'md' | 'lg'
}

function ProgressBar({ value, label, size = 'md' }: ProgressBarProps) {
  const pct = Math.max(0, Math.min(100, value))
  const heights: Record<string, string> = { sm: 'h-1.5', md: 'h-2.5', lg: 'h-4' }

  return (
    <div>
      <div
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label}
        className={`w-full rounded-full bg-muted ${heights[size]} overflow-hidden`}
      >
        <div
          className={`${heights[size]} rounded-full transition-all duration-500 ${coverageColor(pct)}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

// ── CoverageView ──────────────────────────────────────────────────────────────

export function CoverageView() {
  const [data, setData] = useState<CoverageData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const result = await loadCoverage()
      setData(result)
      setError(null)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const overall = data?.overall ?? 0

  return (
    <main aria-labelledby="coverage-heading" className="space-y-8 p-6">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <Shield className="w-7 h-7 text-primary" aria-hidden="true" />
          <div>
            <h1 id="coverage-heading" className="text-2xl font-bold tracking-tight">
              Coverage Report
            </h1>
            <p className="text-sm text-muted-foreground">
              Test coverage analysis across all modules
            </p>
          </div>
        </div>
        <button
          onClick={refresh}
          disabled={loading}
          aria-label="Refresh coverage data"
          className="flex items-center gap-2 px-4 py-2 border rounded-lg bg-card hover:bg-accent transition-colors disabled:opacity-50 text-sm"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} aria-hidden="true" />
          Refresh
        </button>
      </header>

      {/* ── Error ──────────────────────────────────────────────────────────── */}
      {error && (
        <div role="alert" className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* ── Overall ────────────────────────────────────────────────────────── */}
      {data && (
        <>
          <section aria-labelledby="overall-heading" className="rounded-xl border bg-card p-6 space-y-3">
            <div className="flex items-baseline justify-between">
              <h2 id="overall-heading" className="font-semibold">Overall Coverage</h2>
              <span className={`text-3xl font-bold tabular-nums ${coverageTextColor(overall)}`}>
                {overall.toFixed(1)}&nbsp;%
              </span>
            </div>
            <ProgressBar value={overall} label={`Overall coverage: ${overall.toFixed(1)}%`} size="lg" />
            <p className="text-xs text-muted-foreground">
              {overall >= 80
                ? '✅ Coverage target met (≥ 80 %)'
                : overall >= 60
                ? '⚠️  Coverage below target — aim for ≥ 80 %'
                : '❌ Coverage critically low — below 60 %'}
            </p>
          </section>

          {/* ── Module Breakdown ───────────────────────────────────────────── */}
          <section aria-labelledby="modules-heading">
            <h2 id="modules-heading" className="text-base font-semibold mb-3">
              Module Breakdown
            </h2>
            <div className="rounded-xl border bg-card overflow-x-auto">
              <table
                className="w-full text-sm"
                aria-label="Per-module test coverage table"
              >
                <thead>
                  <tr className="bg-muted/50 border-b">
                    <th scope="col" className="px-4 py-3 text-left font-medium text-muted-foreground">Module</th>
                    <th scope="col" className="px-4 py-3 text-right font-medium text-muted-foreground">Lines</th>
                    <th scope="col" className="px-4 py-3 text-right font-medium text-muted-foreground">Covered</th>
                    <th scope="col" className="px-4 py-3 text-right font-medium text-muted-foreground">%</th>
                    <th scope="col" className="px-4 py-3 text-left font-medium text-muted-foreground w-40">Progress</th>
                  </tr>
                </thead>
                <tbody>
                  {data.modules
                    .slice()
                    .sort((a, b) => b.coverage - a.coverage)
                    .map((mod) => (
                    <tr
                      key={mod.module}
                      className="border-b last:border-0 hover:bg-muted/30 transition-colors"
                    >
                      <td className="px-4 py-3 font-mono text-xs">{mod.module}</td>
                      <td className="px-4 py-3 text-right tabular-nums text-muted-foreground">{mod.lines}</td>
                      <td className="px-4 py-3 text-right tabular-nums text-muted-foreground">{mod.coveredLines}</td>
                      <td className={`px-4 py-3 text-right tabular-nums font-semibold ${coverageTextColor(mod.coverage)}`}>
                        {mod.coverage.toFixed(1)}%
                      </td>
                      <td className="px-4 py-3 w-40">
                        <ProgressBar
                          value={mod.coverage}
                          label={`${mod.module} coverage: ${mod.coverage.toFixed(1)}%`}
                          size="sm"
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {data.timestamp && (
            <p className="text-xs text-muted-foreground">
              Report generated: {new Date(data.timestamp).toLocaleString()}
            </p>
          )}
        </>
      )}

      {/* Loading */}
      {loading && !data && (
        <div className="flex items-center justify-center py-20 text-muted-foreground">
          <RefreshCw className="w-6 h-6 animate-spin mr-2" aria-hidden="true" />
          Loading coverage data…
        </div>
      )}
    </main>
  )
}

export default CoverageView
