import { lazy, Suspense, ComponentType } from 'react'
import { Loader2 } from 'lucide-react'

// Loading fallback component
export function PageLoader() {
  return (
    <div className="flex items-center justify-center h-[calc(100vh-8rem)]">
      <div className="text-center">
        <Loader2 className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
        <p className="text-muted-foreground">Loading...</p>
      </div>
    </div>
  )
}

export function ComponentLoader() {
  return (
    <div className="flex items-center justify-center p-8">
      <Loader2 className="w-8 h-8 animate-spin text-primary" />
    </div>
  )
}

// Lazy load page components (using named exports pattern)
export const LazyDashboard = lazy(() => import('../pages/Dashboard').then(m => ({ default: m.Dashboard })))
export const LazyGoals = lazy(() => import('../pages/Goals').then(m => ({ default: m.Goals })))
export const LazyAgents = lazy(() => import('../pages/Agents').then(m => ({ default: m.Agents })))
export const LazyLogs = lazy(() => import('../pages/Logs').then(m => ({ default: m.Logs })))
export const LazySettings = lazy(() => import('../pages/Settings').then(m => ({ default: m.Settings })))
export const LazyChat = lazy(() => import('../pages/Chat').then(m => ({ default: m.Chat })))
export const LazyEditor = lazy(() => import('../pages/Editor').then(m => ({ default: m.Editor })))
export const LazySADD = lazy(() => import('../pages/SADD').then(m => ({ default: m.SADD })))
export const LazyCoverage = lazy(() => import('../pages/Coverage').then(m => ({ default: m.Coverage })))
export const LazyWorkflows = lazy(() => import('../pages/Workflows').then(m => ({ default: m.Workflows })))
export const LazyMCP = lazy(() => import('../pages/MCP').then(m => ({ default: m.MCP })))
export const LazyTerminal = lazy(() => import('../pages/Terminal').then(m => ({ default: m.TerminalPage })))

// Lazy load heavy components
export const LazyCharts = lazy(() => import('./Charts').then(m => ({ default: m.GoalTrendChart })))
export const LazyWorkflowVisualizer = lazy(() => import('./Workflow/WorkflowVisualizer').then(m => ({ default: m.WorkflowVisualizer })))
export const LazyTerminalEmulator = lazy(() => import('./Terminal/TerminalEmulator').then(m => ({ default: m.TerminalEmulator })))

// HOC for lazy loading with suspense
export function withLazyLoad<T extends object>(
  Component: ComponentType<T>,
  fallback: React.ReactNode = <ComponentLoader />
) {
  return function LazyLoadedComponent(props: T) {
    return (
      <Suspense fallback={fallback}>
        <Component {...props} />
      </Suspense>
    )
  }
}

// Preload function for predictive loading
export function preloadComponent(componentName: string) {
  const componentMap: Record<string, () => Promise<any>> = {
    'Dashboard': () => import('../pages/Dashboard'),
    'Goals': () => import('../pages/Goals'),
    'Agents': () => import('../pages/Agents'),
    'Logs': () => import('../pages/Logs'),
    'Chat': () => import('../pages/Chat'),
    'Editor': () => import('../pages/Editor'),
    'Workflows': () => import('../pages/Workflows'),
    'MCP': () => import('../pages/MCP'),
    'Terminal': () => import('../pages/Terminal'),
  }
  
  const loader = componentMap[componentName]
  if (loader) {
    loader()
  }
}

// Preload on hover
export function usePreloadOnHover(componentName: string) {
  return {
    onMouseEnter: () => preloadComponent(componentName),
    onFocus: () => preloadComponent(componentName),
  }
}
