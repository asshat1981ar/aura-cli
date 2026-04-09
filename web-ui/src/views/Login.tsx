/**
 * Login.tsx — Sprint 7
 *
 * JWT login flow:
 *  • Shown when AUTH_ENABLED=true (checked via VITE_AUTH_ENABLED env var
 *    or the backend's /health response)
 *  • POST /auth/login  { username, password }
 *    ← { access_token, token_type, user }
 *  • Stores JWT in localStorage under the key "aura_jwt"
 *  • Exports `ProtectedRoute` wrapper — redirects to /login if no JWT
 */

import { useState, useEffect, ReactNode } from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import { Loader2, LogIn } from 'lucide-react'

// ── Constants ─────────────────────────────────────────────────────────────────

export const JWT_STORAGE_KEY = 'aura_jwt'

// ── Token helpers ─────────────────────────────────────────────────────────────

export function getStoredToken(): string | null {
  try {
    return localStorage.getItem(JWT_STORAGE_KEY)
  } catch {
    return null
  }
}

export function storeToken(token: string): void {
  localStorage.setItem(JWT_STORAGE_KEY, token)
}

export function clearToken(): void {
  localStorage.removeItem(JWT_STORAGE_KEY)
}

export function isAuthenticated(): boolean {
  const token = getStoredToken()
  if (!token) return false
  // Validate JWT expiry if possible (avoid importing jwt-decode)
  try {
    const [, payload] = token.split('.')
    const { exp } = JSON.parse(atob(payload))
    if (exp && Date.now() / 1000 > exp) {
      clearToken()
      return false
    }
  } catch { /* token is not a standard JWT — accept it */ }
  return true
}

// ── API ───────────────────────────────────────────────────────────────────────

interface LoginResponse {
  access_token: string
  token_type: string
  user?: { username: string; role: string }
}

async function loginRequest(username: string, password: string): Promise<LoginResponse> {
  // Try the backend auth endpoint
  const res = await fetch('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (!res.ok) {
    // Attempt legacy endpoint
    const alt = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    })
    if (!alt.ok) throw new Error('Invalid credentials')
    return alt.json()
  }
  return res.json()
}

// ── ProtectedRoute ────────────────────────────────────────────────────────────

interface ProtectedRouteProps {
  children: ReactNode
}

/**
 * Wrap any Route element to require a valid JWT.
 * Redirects to /login preserving the attempted path in `state.from`.
 */
export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const location = useLocation()
  if (!isAuthenticated()) {
    return <Navigate to="/login" state={{ from: location }} replace />
  }
  return <>{children}</>
}

// ── JWTLogin ──────────────────────────────────────────────────────────────────

interface JWTLoginProps {
  /** Called after a successful login so the parent can re-render. */
  onSuccess?: () => void
}

export function JWTLogin({ onSuccess }: JWTLoginProps) {
  const location = useLocation()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError]       = useState<string | null>(null)
  const [loading, setLoading]   = useState(false)
  const [authEnabled, setAuthEnabled] = useState<boolean | null>(null)

  // Detect whether auth is enabled from env var or /health endpoint
  useEffect(() => {
    const envFlag = import.meta.env.VITE_AUTH_ENABLED
    if (envFlag !== undefined) {
      setAuthEnabled(envFlag !== 'false' && envFlag !== '0')
      return
    }
    // Probe the health endpoint
    fetch('/health')
      .then((r) => r.json())
      .then((d) => setAuthEnabled(d?.auth_enabled !== false))
      .catch(() => setAuthEnabled(true))
  }, [])

  // If auth is disabled redirect immediately
  if (authEnabled === false) {
    const from = (location.state as { from?: Location })?.from?.pathname ?? '/'
    return <Navigate to={from} replace />
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const data = await loginRequest(username, password)
      storeToken(data.access_token)
      onSuccess?.()
    } catch (err) {
      setError((err as Error).message ?? 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <div
        className="w-full max-w-sm rounded-2xl border bg-card shadow-lg p-8 space-y-6"
        role="main"
      >
        {/* Logo / Title */}
        <div className="text-center space-y-2">
          <div
            className="w-14 h-14 mx-auto rounded-xl bg-primary flex items-center justify-center"
            aria-hidden="true"
          >
            <span className="text-primary-foreground font-bold text-3xl select-none">A</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight">Sign in to AURA</h1>
          <p className="text-sm text-muted-foreground">
            Enter your credentials to access the dashboard
          </p>
        </div>

        {/* Error */}
        {error && (
          <div
            role="alert"
            aria-live="polite"
            className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 text-sm"
          >
            {error}
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} aria-label="Login form" className="space-y-4" noValidate>
          <div className="space-y-1">
            <label htmlFor="login-username" className="block text-sm font-medium">
              Username
            </label>
            <input
              id="login-username"
              type="text"
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="e.g. admin"
              required
              aria-required="true"
              className="w-full px-4 py-2 border rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-primary text-sm"
            />
          </div>

          <div className="space-y-1">
            <label htmlFor="login-password" className="block text-sm font-medium">
              Password
            </label>
            <input
              id="login-password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              aria-required="true"
              className="w-full px-4 py-2 border rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-primary text-sm"
            />
          </div>

          <button
            type="submit"
            disabled={loading || !username || !password}
            aria-label="Sign in"
            className="w-full py-2.5 px-4 flex items-center justify-center gap-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" />
                Signing in…
              </>
            ) : (
              <>
                <LogIn className="w-4 h-4" aria-hidden="true" />
                Sign In
              </>
            )}
          </button>
        </form>

        <p className="text-center text-xs text-muted-foreground">
          Default credentials:&nbsp;<code className="font-mono">admin / admin</code>
        </p>
      </div>
    </div>
  )
}

export default JWTLogin
