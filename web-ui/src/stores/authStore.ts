import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface User {
  username: string
  role: string
}

interface AuthState {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  login: (username: string, password: string) => Promise<boolean>
  logout: () => void
  setToken: (token: string) => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      isAuthenticated: false,

      login: async (username: string, password: string) => {
        try {
          const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
          })

          if (!response.ok) {
            return false
          }

          const data = await response.json()
          set({
            user: data.user,
            token: data.access_token,
            isAuthenticated: true,
          })
          return true
        } catch (error) {
          console.error('Login error:', error)
          return false
        }
      },

      logout: () => {
        set({ user: null, token: null, isAuthenticated: false })
      },

      setToken: (token: string) => {
        set({ token, isAuthenticated: true })
      },
    }),
    {
      name: 'aura-auth',
      partialize: (state) => ({ token: state.token, user: state.user, isAuthenticated: state.isAuthenticated }),
    }
  )
)
