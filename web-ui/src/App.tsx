import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Dashboard } from './pages/Dashboard'
import { Goals } from './pages/Goals'
import { Agents } from './pages/Agents'
import { Logs } from './pages/Logs'
import { Settings } from './pages/Settings'
import { Login } from './pages/Login'
import { Chat } from './pages/Chat'
import { Editor } from './pages/Editor'
import { SADD } from './pages/SADD'
import { Coverage } from './pages/Coverage'
import { useAuthStore } from './stores/authStore'
import { ErrorBoundary } from './components/ErrorBoundary'

function App() {
  const { isAuthenticated } = useAuthStore()

  if (!isAuthenticated) {
    return (
      <ErrorBoundary>
        <Login />
      </ErrorBoundary>
    )
  }

  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Layout>
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
          </Routes>
        </Layout>
      </BrowserRouter>
    </ErrorBoundary>
  )
}

export default App
