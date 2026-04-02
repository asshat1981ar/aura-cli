import { Sun, Moon, Bell, LogOut } from 'lucide-react'
import { useThemeStore } from '../stores/themeStore'
import { useAuthStore } from '../stores/authStore'

export function Header() {
  const { isDark, toggleTheme } = useThemeStore()
  const { user, logout } = useAuthStore()

  return (
    <header className="h-16 border-b bg-card px-6 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
          <span className="text-primary-foreground font-bold text-lg">A</span>
        </div>
        <h1 className="text-xl font-bold">AURA Dashboard</h1>
      </div>

      <div className="flex items-center gap-4">
        <button
          onClick={toggleTheme}
          className="p-2 rounded-lg hover:bg-accent transition-colors"
          aria-label="Toggle theme"
        >
          {isDark ? (
            <Sun className="w-5 h-5" />
          ) : (
            <Moon className="w-5 h-5" />
          )}
        </button>

        <button
          className="p-2 rounded-lg hover:bg-accent transition-colors relative"
          aria-label="Notifications"
        >
          <Bell className="w-5 h-5" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
        </button>

        <div className="flex items-center gap-3 pl-4 border-l">
          <div className="text-right">
            <p className="text-sm font-medium">{user?.username || 'Guest'}</p>
            <p className="text-xs text-muted-foreground capitalize">
              {user?.role || 'Viewer'}
            </p>
          </div>
          <button
            onClick={logout}
            className="p-2 rounded-lg hover:bg-destructive/10 hover:text-destructive transition-colors"
            aria-label="Logout"
          >
            <LogOut className="w-5 h-5" />
          </button>
        </div>
      </div>
    </header>
  )
}
