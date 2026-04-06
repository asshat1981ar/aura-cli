import { useEffect } from 'react'
import { 
  Terminal, 
  Plus, 
  Trash2, 
  X,
  Command,
  Clock
} from 'lucide-react'
import { TerminalEmulator } from '../components/Terminal'
import { useTerminalStore } from '../stores/terminalStore'
import { cn } from '@/lib/utils'

export function TerminalPage() {
  const { 
    sessions, 
    activeSessionId, 
    createSession, 
    setActiveSession, 
    deleteSession,
    fetchSessions
  } = useTerminalStore()

  useEffect(() => {
    fetchSessions()
    // Create initial session if none exists
    if (sessions.length === 0) {
      createSession()
    }
  }, [])

  const handleNewSession = () => {
    createSession()
  }

  const handleCloseSession = (sessionId: string) => {
    deleteSession(sessionId)
  }

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b bg-card">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Command className="w-6 h-6" />
            Terminal
          </h2>
          <p className="text-muted-foreground">
            Execute shell commands directly in the browser
          </p>
        </div>
        <button
          onClick={handleNewSession}
          className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
        >
          <Plus className="w-4 h-4" />
          New Session
        </button>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Session Sidebar */}
        <div className="w-64 border-r bg-card flex flex-col">
          <div className="p-3 border-b">
            <h3 className="text-sm font-medium text-muted-foreground uppercase">
              Sessions ({sessions.length})
            </h3>
          </div>
          
          <div className="flex-1 overflow-auto p-2 space-y-1">
            {sessions.map((session) => (
              <div
                key={session.id}
                onClick={() => setActiveSession(session.id)}
                className={cn(
                  "group flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-colors",
                  activeSessionId === session.id
                    ? "bg-primary/10 border border-primary/20"
                    : "hover:bg-accent border border-transparent"
                )}
              >
                <Terminal className={cn(
                  "w-4 h-4",
                  activeSessionId === session.id ? "text-primary" : "text-muted-foreground"
                )} />
                <div className="flex-1 min-w-0">
                  <p className={cn(
                    "text-sm font-medium truncate",
                    activeSessionId === session.id && "text-primary"
                  )}>
                    Session {session.id.slice(0, 8)}
                  </p>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Clock className="w-3 h-3" />
                    {session.command_count} commands
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    handleCloseSession(session.id)
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-destructive/10 hover:text-destructive transition-all"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}

            {sessions.length === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                <Terminal className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No active sessions</p>
                <button
                  onClick={handleNewSession}
                  className="text-sm text-primary hover:underline mt-2"
                >
                  Create one
                </button>
              </div>
            )}
          </div>

          {/* Session Actions */}
          <div className="p-3 border-t space-y-2">
            <button
              onClick={handleNewSession}
              className="w-full flex items-center justify-center gap-2 px-3 py-2 border rounded-lg hover:bg-accent transition-colors text-sm"
            >
              <Plus className="w-4 h-4" />
              New Session
            </button>
            {sessions.length > 0 && (
              <button
                onClick={() => sessions.forEach(s => deleteSession(s.id))}
                className="w-full flex items-center justify-center gap-2 px-3 py-2 border border-destructive text-destructive rounded-lg hover:bg-destructive/10 transition-colors text-sm"
              >
                <Trash2 className="w-4 h-4" />
                Close All
              </button>
            )}
          </div>
        </div>

        {/* Terminal Area */}
        <div className="flex-1 p-4 bg-muted/30">
          {activeSessionId ? (
            <TerminalEmulator 
              sessionId={activeSessionId}
              onClose={() => handleCloseSession(activeSessionId)}
            />
          ) : (
            <div className="h-full flex items-center justify-center text-muted-foreground">
              <div className="text-center">
                <Terminal className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p className="text-lg font-medium">No Active Session</p>
                <p className="text-sm mb-4">Select a session or create a new one</p>
                <button
                  onClick={handleNewSession}
                  className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 mx-auto"
                >
                  <Plus className="w-4 h-4" />
                  New Session
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
