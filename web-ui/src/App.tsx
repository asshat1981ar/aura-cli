import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom'
import { Suspense, useEffect, lazy } from 'react'
import { Layout } from './components/Layout'
import { useAuthStore } from './stores/authStore'
import { ErrorBoundary } from './components/ErrorBoundary'
import { PageLoader, preloadComponent } from './components/LazyComponents'

// Lazy load all page components for code splitting
// Using named exports pattern
const Dashboard = lazy(() => import('./pages/Dashboard').then(m => ({ default: m.Dashboard })))
const Goals = lazy(() => import('./pages/Goals').then(m => ({ default: m.Goals })))
const Agents = lazy(() => import('./pages/Agents').then(m => ({ default: m.Agents })))
const Logs = lazy(() => import('./pages/Logs').then(m => ({ default: m.Logs })))
const Settings = lazy(() => import('./pages/Settings').then(m => ({ default: m.Settings })))
const Chat = lazy(() => import('./pages/Chat').then(m => ({ default: m.Chat })))
const Editor = lazy(() => import('./pages/Editor').then(m => ({ default: m.Editor })))
const SADD = lazy(() => import('./pages/SADD').then(m => ({ default: m.SADD })))
const Coverage = lazy(() => import('./pages/Coverage').then(m => ({ default: m.Coverage })))
const Workflows = lazy(() => import('./pages/Workflows').then(m => ({ default: m.Workflows })))
const MCP = lazy(() => import('./pages/MCP').then(m => ({ default: m.MCP })))
const TerminalPage = lazy(() => import('./pages/Terminal').then(m => ({ default: m.TerminalPage })))
const Login = lazy(() => import('./pages/Login').then(m => ({ default: m.Login })))

// Preload adjacent routes based on current route
function RoutePreloader() {
  const location = useLocation()
  
  useEffect(() => {
    // Preload likely next routes based on current route
    const routePreloadMap: Record<string, string[]> = {
      '/': ['Goals', 'Agents', 'Chat'],
      '/goals': ['Agents', 'Logs', 'Dashboard'],
      '/agents': ['Goals', 'Logs', 'Dashboard'],
      '/chat': ['Editor', 'Goals'],
      '/editor': ['SADD', 'Terminal'],
      '/workflows': ['MCP', 'Agents'],
      '/logs': ['Settings', 'Coverage'],
    }
    
    const routesToPreload = routePreloadMap[location.pathname] || []
    
    // Delay preloading to not interfere with current page load
    const timer = setTimeout(() => {
      routesToPreload.forEach(route => preloadComponent(route))
    }, 2000)
    
    return () => clearTimeout(timer)
  }, [location.pathname])
  
  return null
}

function App() {
  const { isAuthenticated } = useAuthStore()

  if (!isAuthenticated) {
    return (
      <ErrorBoundary>
        <Suspense fallback={<PageLoader />}>
          <Login />
        </Suspense>
      </ErrorBoundary>
    )
  }

  return (
    <ErrorBoundary>
      <BrowserRouter>
        <RoutePreloader />
        <Layout>
          <Suspense fallback={<PageLoader />}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/goals" element={<Goals />} />
              <Route path="/agents" element={<Agents />} />
              <Route path="/logs" element={<Logs />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/chat" element={<Chat />} />
              <Route path="/editor" element={<Editor />} />
              <Route path="/sadd" element={<SADD />} />
              <Route path="/coverage" element={<Coverage />} />
              <Route path="/workflows" element={<Workflows />} />
              <Route path="/mcp" element={<MCP />} />
              <Route path="/terminal" element={<TerminalPage />} />
            </Routes>
          </Suspense>
        </Layout>
      </BrowserRouter>
    </ErrorBoundary>
  )
}

export default App
