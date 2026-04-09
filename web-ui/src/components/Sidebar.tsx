import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Target,
  Bot,
  FileText,
  Settings,
  MessageSquare,
  Code2,
  Workflow,
  BarChart3,
  GitBranch,
  Puzzle,
  Terminal,
  Menu,
  X
} from 'lucide-react'
import { useState, useEffect } from 'react'
import { cn } from '@/lib/utils'

const navItems = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/analytics', icon: BarChart3, label: 'Analytics' },
  { path: '/chat', icon: MessageSquare, label: 'AI Chat' },
  { path: '/editor', icon: Code2, label: 'Editor' },
  { path: '/goals', icon: Target, label: 'Goals' },
  { path: '/agents', icon: Bot, label: 'Agents' },
  { path: '/sadd', icon: Workflow, label: 'SADD' },
  { path: '/workflows', icon: GitBranch, label: 'Workflows' },
  { path: '/mcp', icon: Puzzle, label: 'MCP Tools' },
  { path: '/terminal', icon: Terminal, label: 'Terminal' },
  { path: '/coverage', icon: BarChart3, label: 'Coverage' },
  { path: '/logs', icon: FileText, label: 'Logs' },
  { path: '/settings', icon: Settings, label: 'Settings' },
]

export function Sidebar() {
  const [isOpen, setIsOpen] = useState(false)
  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 1024)
    }
    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  // Close sidebar when route changes on mobile
  const handleNavClick = () => {
    if (isMobile) {
      setIsOpen(false)
    }
  }

  return (
    <>
      {/* Mobile toggle button */}
      {isMobile && (
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="fixed top-20 left-4 z-40 p-2 bg-card border rounded-lg shadow-lg lg:hidden"
        >
          {isOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      )}

      {/* Backdrop for mobile */}
      {isMobile && isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 lg:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          "min-h-[calc(100vh-4rem)] border-r bg-card transition-transform duration-300",
          "lg:w-64 lg:translate-x-0 lg:static",
          isMobile && [
            "fixed left-0 top-16 z-40 w-64 h-[calc(100vh-4rem)]",
            isOpen ? "translate-x-0" : "-translate-x-full"
          ]
        )}
      >
        <nav className="p-4 space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              onClick={handleNavClick}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 px-4 py-3 rounded-lg transition-colors",
                  isActive
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                )
              }
            >
              <item.icon className="w-5 h-5" />
              <span className="font-medium">{item.label}</span>
            </NavLink>
          ))}
        </nav>
      </aside>
    </>
  )
}
