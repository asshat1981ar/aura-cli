import { useState, useEffect } from 'react'
import { 
  Plus, 
  Play, 
  Pause, 
  Square, 
  Trash2, 
  Clock,
  CheckCircle,
  AlertCircle,
  Loader2,
  FileText,
  Workflow
} from 'lucide-react'
import { useSADDStore, Workstream } from '../../stores/saddStore'

export function SessionManager() {
  const {
    sessions,
    activeSessionId,
    isLoading,
    fetchSessions,
    createSession,
    startSession,
    pauseSession,
    stopSession,
    deleteSession,
    setActiveSession,
  } = useSADDStore()
  
  const [showNewSession, setShowNewSession] = useState(false)
  const [newTitle, setNewTitle] = useState('')
  const [newSpec, setNewSpec] = useState('')
  
  useEffect(() => {
    fetchSessions()
    const interval = setInterval(fetchSessions, 5000) // Refresh every 5s
    return () => clearInterval(interval)
  }, [fetchSessions])
  
  const handleCreate = async () => {
    if (!newTitle.trim()) return
    await createSession(newTitle, newSpec)
    setNewTitle('')
    setNewSpec('')
    setShowNewSession(false)
  }
  
  const activeSession = sessions.find((s) => s.id === activeSessionId)
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">SADD Sessions</h2>
          <p className="text-muted-foreground">
            Spec-Aware Design Decomposition sessions
          </p>
        </div>
        <button
          onClick={() => setShowNewSession(true)}
          className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors"
        >
          <Plus className="w-4 h-4" />
          New Session
        </button>
      </div>
      
      {/* Sessions Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {sessions.map((session) => (
          <SessionCard
            key={session.id}
            session={session}
            isActive={session.id === activeSessionId}
            onSelect={() => setActiveSession(session.id)}
            onStart={() => startSession(session.id)}
            onPause={() => pauseSession(session.id)}
            onStop={() => stopSession(session.id)}
            onDelete={() => deleteSession(session.id)}
          />
        ))}
        
        {sessions.length === 0 && !isLoading && (
          <div className="col-span-full text-center py-12 text-muted-foreground bg-card border rounded-xl">
            <Workflow className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium">No SADD sessions yet</p>
            <p>Create a new session to start decomposing design specs</p>
          </div>
        )}
      </div>
      
      {/* Active Session Details */}
      {activeSession && (
        <div className="bg-card border rounded-xl p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-xl font-semibold">{activeSession.title}</h3>
              <p className="text-sm text-muted-foreground">
                Created {new Date(activeSession.created_at).toLocaleString()}
              </p>
            </div>
            <StatusBadge status={activeSession.status} />
          </div>
          
          {/* Design Spec Preview */}
          <div className="mb-6">
            <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
              <FileText className="w-4 h-4" />
              Design Spec
            </h4>
            <div className="bg-muted p-4 rounded-lg max-h-40 overflow-y-auto">
              <pre className="text-sm whitespace-pre-wrap">
                {activeSession.design_spec || 'No design spec provided'}
              </pre>
            </div>
          </div>
          
          {/* Workstreams */}
          <div>
            <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
              <Workflow className="w-4 h-4" />
              Workstreams
            </h4>
            <div className="space-y-2">
              {activeSession.workstreams.map((workstream) => (
                <WorkstreamCard key={workstream.id} workstream={workstream} />
              ))}
              
              {activeSession.workstreams.length === 0 && (
                <p className="text-muted-foreground text-sm">
                  No workstreams yet. Start the session to generate workstreams.
                </p>
              )}
            </div>
          </div>
          
          {/* Artifacts */}
          {activeSession.artifacts.length > 0 && (
            <div className="mt-6">
              <h4 className="text-sm font-medium mb-2">Artifacts</h4>
              <div className="flex flex-wrap gap-2">
                {activeSession.artifacts.map((artifact, i) => (
                  <span
                    key={i}
                    className="px-3 py-1 bg-muted text-sm rounded-full"
                  >
                    {artifact}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* New Session Modal */}
      {showNewSession && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-card border rounded-xl p-6 w-[600px] max-h-[80vh] overflow-y-auto shadow-xl">
            <h3 className="text-xl font-semibold mb-4">New SADD Session</h3>
            
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-2 block">Session Title</label>
                <input
                  type="text"
                  value={newTitle}
                  onChange={(e) => setNewTitle(e.target.value)}
                  placeholder="e.g., Refactor Authentication Module"
                  className="w-full px-3 py-2 border rounded-lg bg-background"
                />
              </div>
              
              <div>
                <label className="text-sm font-medium mb-2 block">Design Spec</label>
                <textarea
                  value={newSpec}
                  onChange={(e) => setNewSpec(e.target.value)}
                  placeholder="# Design Specification\n\n## Overview\nDescribe what needs to be built...\n\n## Requirements\n- Requirement 1\n- Requirement 2\n\n## Constraints\n- Constraint 1"
                  className="w-full px-3 py-2 border rounded-lg bg-background min-h-[200px] font-mono text-sm"
                />
              </div>
              
              <div className="flex gap-2 pt-4">
                <button
                  onClick={() => setShowNewSession(false)}
                  className="flex-1 px-4 py-2 border rounded-lg hover:bg-accent transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleCreate}
                  disabled={!newTitle.trim() || isLoading}
                  className="flex-1 px-4 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors disabled:opacity-50"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin inline mr-2" />
                      Creating...
                    </>
                  ) : (
                    'Create Session'
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

interface SessionCardProps {
  session: {
    id: string
    title: string
    status: string
    workstreams: Workstream[]
    created_at: string
  }
  isActive: boolean
  onSelect: () => void
  onStart: () => void
  onPause: () => void
  onStop: () => void
  onDelete: () => void
}

function SessionCard({ session, isActive, onSelect, onStart, onPause, onStop, onDelete }: SessionCardProps) {
  const completedWorkstreams = session.workstreams.filter((w) => w.status === 'completed').length
  const totalWorkstreams = session.workstreams.length
  const progress = totalWorkstreams > 0 ? (completedWorkstreams / totalWorkstreams) * 100 : 0
  
  return (
    <div
      onClick={onSelect}
      className={`bg-card border rounded-xl p-4 cursor-pointer transition-all hover:shadow-md ${
        isActive ? 'ring-2 ring-primary' : ''
      }`}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold truncate">{session.title}</h3>
          <p className="text-xs text-muted-foreground">
            {new Date(session.created_at).toLocaleDateString()}
          </p>
        </div>
        <StatusBadge status={session.status} />
      </div>
      
      {/* Progress */}
      <div className="mb-4">
        <div className="flex justify-between text-xs text-muted-foreground mb-1">
          <span>Progress</span>
          <span>{completedWorkstreams}/{totalWorkstreams} workstreams</span>
        </div>
        <div className="h-2 bg-muted rounded-full overflow-hidden">
          <div
            className="h-full bg-primary transition-all"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>
      
      {/* Actions */}
      <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
        {session.status === 'idle' && (
          <button
            onClick={onStart}
            className="p-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
            title="Start"
          >
            <Play className="w-4 h-4" />
          </button>
        )}
        
        {session.status === 'running' && (
          <>
            <button
              onClick={onPause}
              className="p-2 bg-amber-500 text-white rounded-lg hover:bg-amber-600 transition-colors"
              title="Pause"
            >
              <Pause className="w-4 h-4" />
            </button>
            <button
              onClick={onStop}
              className="p-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
              title="Stop"
            >
              <Square className="w-4 h-4" />
            </button>
          </>
        )}
        
        {session.status === 'paused' && (
          <>
            <button
              onClick={onStart}
              className="p-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
              title="Resume"
            >
              <Play className="w-4 h-4" />
            </button>
            <button
              onClick={onStop}
              className="p-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
              title="Stop"
            >
              <Square className="w-4 h-4" />
            </button>
          </>
        )}
        
        <div className="flex-1" />
        
        <button
          onClick={onDelete}
          className="p-2 text-muted-foreground hover:text-destructive hover:bg-destructive/10 rounded-lg transition-colors"
          title="Delete"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const styles = {
    idle: 'bg-gray-100 text-gray-800',
    running: 'bg-blue-100 text-blue-800 animate-pulse',
    paused: 'bg-amber-100 text-amber-800',
    completed: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800',
  }
  
  const icons = {
    idle: Clock,
    running: Loader2,
    paused: Pause,
    completed: CheckCircle,
    failed: AlertCircle,
  }
  
  const Icon = icons[status as keyof typeof icons] || Clock
  
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium capitalize ${
      styles[status as keyof typeof styles] || styles.idle
    }`}>
      <Icon className={`w-3 h-3 ${status === 'running' ? 'animate-spin' : ''}`} />
      {status}
    </span>
  )
}

function WorkstreamCard({ workstream }: { workstream: Workstream }) {
  const statusColors = {
    pending: 'bg-gray-100 border-gray-200',
    running: 'bg-blue-50 border-blue-200',
    completed: 'bg-green-50 border-green-200',
    failed: 'bg-red-50 border-red-200',
  }
  
  return (
    <div className={`p-3 rounded-lg border ${statusColors[workstream.status]}`}>
      <div className="flex items-center justify-between mb-2">
        <h5 className="font-medium text-sm">{workstream.name}</h5>
        <StatusBadge status={workstream.status} />
      </div>
      
      <p className="text-sm text-muted-foreground mb-2">
        {workstream.description}
      </p>
      
      {/* Progress bar */}
      <div className="h-1.5 bg-muted rounded-full overflow-hidden mb-2">
        <div
          className="h-full bg-primary transition-all"
          style={{ width: `${workstream.progress}%` }}
        />
      </div>
      
      {/* Dependencies */}
      {workstream.dependencies.length > 0 && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span>Depends on:</span>
          {workstream.dependencies.map((dep) => (
            <span key={dep} className="px-1.5 py-0.5 bg-muted rounded">
              {dep}
            </span>
          ))}
        </div>
      )}
      
      {/* Artifacts */}
      {workstream.artifacts.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {workstream.artifacts.map((artifact, i) => (
            <span key={i} className="px-2 py-0.5 bg-background text-xs rounded border">
              {artifact}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
