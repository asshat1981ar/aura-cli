import { useEffect, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { 
  Search, 
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
  Activity,
  Command
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface CommandItem {
  id: string
  title: string
  description: string
  icon: React.ComponentType<{ className?: string }>
  shortcut?: string
  action: () => void
}

export function CommandPalette() {
  const [isOpen, setIsOpen] = useState(false)
  const [search, setSearch] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(0)
  const navigate = useNavigate()

  const commands: CommandItem[] = [
    // Navigation
    { id: 'dashboard', title: 'Dashboard', description: 'View system overview', icon: LayoutDashboard, shortcut: 'G D', action: () => navigate('/') },
    { id: 'chat', title: 'AI Chat', description: 'Chat with AI agents', icon: MessageSquare, shortcut: 'G C', action: () => navigate('/chat') },
    { id: 'editor', title: 'Code Editor', description: 'Browse and edit files', icon: Code2, shortcut: 'G E', action: () => navigate('/editor') },
    { id: 'goals', title: 'Goals', description: 'Manage goal queue', icon: Target, shortcut: 'G G', action: () => navigate('/goals') },
    { id: 'agents', title: 'Agents', description: 'Monitor agents', icon: Bot, shortcut: 'G A', action: () => navigate('/agents') },
    { id: 'sadd', title: 'SADD Manager', description: 'Design decomposition', icon: Workflow, shortcut: 'G S', action: () => navigate('/sadd') },
    { id: 'workflows', title: 'Workflows', description: 'n8n workflows', icon: GitBranch, shortcut: 'G W', action: () => navigate('/workflows') },
    { id: 'mcp', title: 'MCP Tools', description: 'Execute MCP tools', icon: Puzzle, shortcut: 'G M', action: () => navigate('/mcp') },
    { id: 'terminal', title: 'Terminal', description: 'Shell console', icon: Terminal, shortcut: 'G T', action: () => navigate('/terminal') },
    { id: 'coverage', title: 'Coverage', description: 'Test coverage', icon: BarChart3, shortcut: 'G V', action: () => navigate('/coverage') },
    { id: 'logs', title: 'Logs', description: 'System logs', icon: FileText, shortcut: 'G L', action: () => navigate('/logs') },
    { id: 'settings', title: 'Settings', description: 'Configuration', icon: Settings, shortcut: 'G ,', action: () => navigate('/settings') },
    
    // Actions
    { id: 'reload', title: 'Reload Page', description: 'Refresh current page', icon: Activity, shortcut: 'Ctrl R', action: () => window.location.reload() },
  ]

  const filteredCommands = commands.filter(cmd => 
    cmd.title.toLowerCase().includes(search.toLowerCase()) ||
    cmd.description.toLowerCase().includes(search.toLowerCase())
  )

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    // Open with Ctrl+K or Cmd+K
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault()
      setIsOpen(prev => !prev)
    }
    
    // Close with Escape
    if (e.key === 'Escape') {
      setIsOpen(false)
    }
    
    // Navigation shortcuts (G + letter)
    if (e.key === 'g' && !isOpen) {
      const handler = (ev: KeyboardEvent) => {
        const key = ev.key.toLowerCase()
        const cmd = commands.find(c => c.shortcut === `G ${key.toUpperCase()}`)
        if (cmd) {
          ev.preventDefault()
          cmd.action()
        }
        window.removeEventListener('keydown', handler)
      }
      window.addEventListener('keydown', handler, { once: true })
    }
  }, [isOpen, commands])

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  useEffect(() => {
    setSelectedIndex(0)
  }, [search])

  const handleSelect = (index: number) => {
    if (filteredCommands[index]) {
      filteredCommands[index].action()
      setIsOpen(false)
      setSearch('')
    }
  }

  const handleInputKeyDown = (e: React.KeyboardEvent) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault()
        setSelectedIndex(i => Math.min(i + 1, filteredCommands.length - 1))
        break
      case 'ArrowUp':
        e.preventDefault()
        setSelectedIndex(i => Math.max(i - 1, 0))
        break
      case 'Enter':
        e.preventDefault()
        handleSelect(selectedIndex)
        break
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh]">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={() => setIsOpen(false)}
      />
      
      {/* Modal */}
      <div className="relative w-full max-w-2xl mx-4 bg-card rounded-xl shadow-2xl border overflow-hidden">
        {/* Search input */}
        <div className="flex items-center gap-3 px-4 py-4 border-b">
          <Search className="w-5 h-5 text-muted-foreground" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            onKeyDown={handleInputKeyDown}
            placeholder="Type a command or search..."
            className="flex-1 bg-transparent outline-none text-lg"
            autoFocus
          />
          <kbd className="px-2 py-1 bg-muted rounded text-xs">ESC</kbd>
        </div>
        
        {/* Commands list */}
        <div className="max-h-[400px] overflow-auto py-2">
          {filteredCommands.length === 0 ? (
            <div className="px-4 py-8 text-center text-muted-foreground">
              No commands found
            </div>
          ) : (
            filteredCommands.map((cmd, index) => {
              const Icon = cmd.icon
              return (
                <button
                  key={cmd.id}
                  onClick={() => handleSelect(index)}
                  className={cn(
                    "w-full flex items-center gap-3 px-4 py-3 text-left transition-colors",
                    index === selectedIndex ? "bg-accent" : "hover:bg-accent/50"
                  )}
                >
                  <Icon className="w-5 h-5 text-muted-foreground" />
                  <div className="flex-1">
                    <div className="font-medium">{cmd.title}</div>
                    <div className="text-sm text-muted-foreground">{cmd.description}</div>
                  </div>
                  {cmd.shortcut && (
                    <kbd className="px-2 py-1 bg-muted rounded text-xs font-mono">
                      {cmd.shortcut}
                    </kbd>
                  )}
                </button>
              )
            })
          )}
        </div>
        
        {/* Footer */}
        <div className="flex items-center gap-4 px-4 py-2 border-t text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <kbd className="px-1.5 py-0.5 bg-muted rounded">↑↓</kbd>
            <span>Navigate</span>
          </div>
          <div className="flex items-center gap-1">
            <kbd className="px-1.5 py-0.5 bg-muted rounded">↵</kbd>
            <span>Select</span>
          </div>
          <div className="flex items-center gap-1">
            <kbd className="px-1.5 py-0.5 bg-muted rounded">G</kbd>
            <span>Quick nav</span>
          </div>
        </div>
      </div>
    </div>
  )
}
