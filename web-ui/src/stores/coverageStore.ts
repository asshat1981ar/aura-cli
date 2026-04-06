import { create } from 'zustand'

export interface CoverageGap {
  id: string
  file_path: string
  function_name: string
  line_number: number
  complexity: number
  impact_score: number
  severity: 'critical' | 'high' | 'medium' | 'low'
  reason: string
}

export interface CoverageFile {
  lines: number
  functions: number
  branches?: number
}

export interface CoverageData {
  overall: number
  files: Record<string, CoverageFile>
  timestamp: string
  mock?: boolean
}

export interface TestSuite {
  total: number
  passed: number
  failed: number
  skipped: number
  duration: number
}

export interface ModuleCoverage {
  name: string
  path: string
  coverage: number
  lines_total: number
  lines_covered: number
  functions_total: number
  functions_covered: number
  children?: ModuleCoverage[]
}

interface CoverageState {
  coverage: CoverageData | null
  gaps: CoverageGap[]
  tests: TestSuite | null
  modules: ModuleCoverage[]
  isLoading: boolean
  error: string | null
  
  // Actions
  fetchCoverage: () => Promise<void>
  fetchGaps: () => Promise<void>
  fetchTests: () => Promise<void>
  runTests: () => Promise<boolean>
  refreshAll: () => Promise<void>
  clearError: () => void
}

export const useCoverageStore = create<CoverageState>((set, get) => ({
  coverage: null,
  gaps: [],
  tests: null,
  modules: [],
  isLoading: false,
  error: null,

  fetchCoverage: async () => {
    try {
      const response = await fetch('/api/coverage')
      if (!response.ok) throw new Error('Failed to fetch coverage')
      const coverage = await response.json()
      
      // Transform files into module structure
      const modules = transformFilesToModules(coverage.files || {})
      
      set({ coverage, modules })
    } catch (error) {
      console.error('Failed to fetch coverage:', error)
      set({ error: (error as Error).message })
    }
  },

  fetchGaps: async () => {
    try {
      const response = await fetch('/api/coverage/gaps')
      if (!response.ok) throw new Error('Failed to fetch coverage gaps')
      const gaps = await response.json()
      set({ gaps })
    } catch (error) {
      console.error('Failed to fetch coverage gaps:', error)
    }
  },

  fetchTests: async () => {
    try {
      const response = await fetch('/api/tests')
      if (!response.ok) throw new Error('Failed to fetch tests')
      const tests = await response.json()
      set({ tests })
    } catch (error) {
      console.error('Failed to fetch tests:', error)
    }
  },

  runTests: async () => {
    try {
      const response = await fetch('/api/tests/run', {
        method: 'POST'
      })
      if (!response.ok) return false
      return true
    } catch (error) {
      console.error('Failed to run tests:', error)
      return false
    }
  },

  refreshAll: async () => {
    set({ isLoading: true, error: null })
    try {
      await Promise.all([
        get().fetchCoverage(),
        get().fetchGaps(),
        get().fetchTests()
      ])
      set({ isLoading: false })
    } catch (error) {
      set({ error: (error as Error).message, isLoading: false })
    }
  },

  clearError: () => {
    set({ error: null })
  }
}))

function transformFilesToModules(files: Record<string, CoverageFile>): ModuleCoverage[] {
  const modules: Record<string, ModuleCoverage> = {}
  
  Object.entries(files).forEach(([path, data]) => {
    const parts = path.split('/')
    const moduleName = parts[0] || 'root'
    const fileName = parts[parts.length - 1]
    
    if (!modules[moduleName]) {
      modules[moduleName] = {
        name: moduleName,
        path: moduleName,
        coverage: 0,
        lines_total: 0,
        lines_covered: 0,
        functions_total: 0,
        functions_covered: 0,
        children: []
      }
    }
    
    const linesCoverage = data.lines || 0
    const linesTotal = 100 // Estimate
    const linesCovered = Math.round(linesTotal * linesCoverage / 100)
    
    modules[moduleName].children?.push({
      name: fileName,
      path: path,
      coverage: linesCoverage,
      lines_total: linesTotal,
      lines_covered: linesCovered,
      functions_total: 10,
      functions_covered: Math.round(10 * (data.functions || 0) / 100)
    })
    
    // Update module totals
    modules[moduleName].lines_total += linesTotal
    modules[moduleName].lines_covered += linesCovered
  })
  
  // Calculate module coverage
  Object.values(modules).forEach(m => {
    m.coverage = m.lines_total > 0 ? Math.round(m.lines_covered / m.lines_total * 100) : 0
  })
  
  return Object.values(modules)
}
