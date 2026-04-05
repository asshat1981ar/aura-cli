import { useState, useRef, useEffect } from 'react'
import { 
  Send, 
  Plus, 
  Trash2, 
  Bot, 
  User, 
  Paperclip,
  X,
  ChevronDown,
  Loader2,
  Wifi,
  WifiOff
} from 'lucide-react'
import { useChatStore, useChatWebSocket, ChatMessage } from '../../stores/chatStore'
import { useAgentStore } from '../../stores/agentStore'


interface ChatInterfaceProps {
  isOpen: boolean
  onClose: () => void
}

export function ChatInterface({ isOpen, onClose }: ChatInterfaceProps) {
  const { 
    sessions, 
    activeSessionId, 
    isLoading, 
    wsConnected,
    createSession, 
    deleteSession, 
    setActiveSession, 
    sendMessage,
    loadSessions
  } = useChatStore()
  
  const { agents } = useAgentStore()
  const [input, setInput] = useState('')
  const [showNewSession, setShowNewSession] = useState(false)
  const [selectedAgent, setSelectedAgent] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  // Initialize WebSocket
  useChatWebSocket()
  
  // Load sessions on mount
  useEffect(() => {
    loadSessions()
  }, [loadSessions])
  
  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [sessions, activeSessionId])
  
  const activeSession = sessions.find((s) => s.id === activeSessionId)
  
  const handleSend = async () => {
    if (!input.trim() || !activeSessionId) return
    
    await sendMessage(input)
    setInput('')
  }
  
  const handleNewSession = () => {
    if (selectedAgent) {
      createSession(selectedAgent)
      setShowNewSession(false)
      setSelectedAgent('')
    }
  }
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }
  
  if (!isOpen) return null
  
  return (
    <div className="fixed inset-x-0 bottom-0 h-[500px] bg-card border-t shadow-2xl flex z-50">
      {/* Sidebar - Session List */}
      <div className="w-64 border-r bg-muted/30 flex flex-col">
        <div className="p-4 border-b flex items-center justify-between">
          <h3 className="font-semibold">Chat Sessions</h3>
          <button
            onClick={() => setShowNewSession(true)}
            className="p-1.5 hover:bg-accent rounded-lg transition-colors"
            title="New session"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>
        
        {/* Connection Status */}
        <div className={`px-4 py-2 text-xs flex items-center gap-2 ${wsConnected ? 'text-green-600' : 'text-amber-600'}`}>
          {wsConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
          {wsConnected ? 'Connected' : 'Reconnecting...'}
        </div>
        
        {/* Session List */}
        <div className="flex-1 overflow-y-auto">
          {sessions.map((session) => (
            <div
              key={session.id}
              onClick={() => setActiveSession(session.id)}
              className={`px-4 py-3 cursor-pointer border-b transition-colors group ${
                session.id === activeSessionId
                  ? 'bg-primary/10 border-primary/20'
                  : 'hover:bg-accent'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-sm truncate">{session.title}</p>
                  <p className="text-xs text-muted-foreground truncate">
                    with {session.agent}
                  </p>
                  {session.messages.length > 0 && (
                    <p className="text-xs text-muted-foreground truncate mt-1">
                      {session.messages[session.messages.length - 1].content.slice(0, 50)}...
                    </p>
                  )}
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    deleteSession(session.id)
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-destructive/10 hover:text-destructive rounded transition-all"
                >
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>
            </div>
          ))}
          
          {sessions.length === 0 && (
            <div className="p-4 text-center text-muted-foreground text-sm">
              No chat sessions yet.
              <br />
              Click + to start one.
            </div>
          )}
        </div>
      </div>
      
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="h-14 border-b flex items-center justify-between px-4">
          <div className="flex items-center gap-3">
            {activeSession ? (
              <>
                <Bot className="w-5 h-5 text-primary" />
                <div>
                  <h3 className="font-medium">{activeSession.title}</h3>
                  <p className="text-xs text-muted-foreground">
                    Chatting with {activeSession.agent}
                  </p>
                </div>
              </>
            ) : (
              <p className="text-muted-foreground">Select or create a chat session</p>
            )}
          </div>
          
          <button
            onClick={onClose}
            className="p-2 hover:bg-accent rounded-lg transition-colors"
          >
            <ChevronDown className="w-5 h-5" />
          </button>
        </div>
        
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {activeSession?.messages.map((message) => (
            <ChatMessageBubble key={message.id} message={message} />
          ))}
          
          {isLoading && (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm">Agent is thinking...</span>
            </div>
          )}
          
          <div ref={messagesEndRef} />
          
          {!activeSession && (
            <div className="h-full flex items-center justify-center text-muted-foreground">
              <div className="text-center">
                <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>Select a chat session or create a new one</p>
              </div>
            </div>
          )}
        </div>
        
        {/* Input Area */}
        <div className="p-4 border-t">
          <div className="flex gap-2">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="p-3 border rounded-lg hover:bg-accent transition-colors"
              title="Attach file"
            >
              <Paperclip className="w-4 h-4" />
            </button>
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              onChange={(e) => {
                // Handle file attachment
                console.log('File selected:', e.target.files?.[0])
              }}
            />
            
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={activeSession ? "Type your message..." : "Select a session first"}
              disabled={!activeSession || isLoading}
              className="flex-1 px-4 py-3 border rounded-lg bg-background resize-none focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50"
              rows={1}
              style={{ minHeight: '48px', maxHeight: '120px' }}
            />
            
            <button
              onClick={handleSend}
              disabled={!input.trim() || !activeSession || isLoading}
              className="px-4 py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            Press Enter to send, Shift+Enter for new line
          </p>
        </div>
      </div>
      
      {/* New Session Modal */}
      {showNewSession && (
        <div className="absolute inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-card border rounded-xl p-6 w-96 shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">New Chat Session</h3>
              <button
                onClick={() => setShowNewSession(false)}
                className="p-1 hover:bg-accent rounded"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-2 block">Select Agent</label>
                <select
                  value={selectedAgent}
                  onChange={(e) => setSelectedAgent(e.target.value)}
                  className="w-full px-3 py-2 border rounded-lg bg-background"
                >
                  <option value="">Choose an agent...</option>
                  {agents.map((agent) => (
                    <option key={agent.id} value={agent.id}>
                      {agent.name} ({agent.capabilities.slice(0, 2).join(', ')}...)
                    </option>
                  ))}
                </select>
              </div>
              
              <div className="flex gap-2">
                <button
                  onClick={() => setShowNewSession(false)}
                  className="flex-1 px-4 py-2 border rounded-lg hover:bg-accent transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleNewSession}
                  disabled={!selectedAgent}
                  className="flex-1 px-4 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors disabled:opacity-50"
                >
                  Start Chat
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function ChatMessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user'
  
  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
        isUser ? 'bg-primary text-primary-foreground' : 'bg-muted'
      }`}>
        {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
      </div>
      
      <div className={`max-w-[70%] ${isUser ? 'items-end' : 'items-start'}`}>
        <div className={`px-4 py-2 rounded-2xl ${
          isUser 
            ? 'bg-primary text-primary-foreground rounded-tr-sm' 
            : 'bg-muted rounded-tl-sm'
        }`}>
          <div className="prose prose-sm dark:prose-invert max-w-none">
            {message.content}
          </div>
        </div>
        
        {message.metadata && (
          <p className="text-xs text-muted-foreground mt-1">
            {message.metadata.model} • {message.metadata.tokens} tokens
          </p>
        )}
        
        <p className="text-xs text-muted-foreground mt-0.5">
          {new Date(message.timestamp).toLocaleTimeString()}
        </p>
      </div>
    </div>
  )
}
