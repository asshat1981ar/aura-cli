import { create } from 'zustand'
import { useEffect, useRef } from 'react'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: string
  agent?: string
  attachments?: Attachment[]
  metadata?: {
    model?: string
    tokens?: number
    latency?: number
  }
}

export interface Attachment {
  id: string
  type: 'file' | 'image' | 'code'
  name: string
  content: string
  language?: string
}

export interface ChatSession {
  id: string
  title: string
  agent: string
  messages: ChatMessage[]
  created_at: string
  updated_at: string
}

interface ChatState {
  sessions: ChatSession[]
  activeSessionId: string | null
  isLoading: boolean
  error: string | null
  wsConnected: boolean
  availableAgents: string[]
  
  // Actions
  createSession: (agent: string, title?: string) => string
  deleteSession: (id: string) => void
  setActiveSession: (id: string | null) => void
  sendMessage: (content: string, attachments?: Attachment[]) => Promise<void>
  addMessage: (sessionId: string, message: ChatMessage) => void
  loadSessions: () => void
  setWsConnected: (connected: boolean) => void
  setAvailableAgents: (agents: string[]) => void
}

const STORAGE_KEY = 'aura_chat_sessions'

export const useChatStore = create<ChatState>((set, get) => ({
  sessions: [],
  activeSessionId: null,
  isLoading: false,
  error: null,
  wsConnected: false,
  availableAgents: [],

  createSession: (agent, title) => {
    const id = `chat-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    const newSession: ChatSession = {
      id,
      title: title || `Chat with ${agent}`,
      agent,
      messages: [],
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }
    
    set((state) => {
      const sessions = [newSession, ...state.sessions]
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions))
      return { sessions, activeSessionId: id }
    })
    
    return id
  },

  deleteSession: (id) => {
    set((state) => {
      const sessions = state.sessions.filter((s) => s.id !== id)
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions))
      return {
        sessions,
        activeSessionId: state.activeSessionId === id ? null : state.activeSessionId,
      }
    })
  },

  setActiveSession: (id) => set({ activeSessionId: id }),

  sendMessage: async (content, attachments) => {
    const { activeSessionId, sessions } = get()
    if (!activeSessionId) return

    const session = sessions.find((s) => s.id === activeSessionId)
    if (!session) return

    // Add user message
    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
      attachments,
    }

    set((state) => ({
      sessions: state.sessions.map((s) =>
        s.id === activeSessionId
          ? { ...s, messages: [...s.messages, userMessage], updated_at: new Date().toISOString() }
          : s
      ),
      isLoading: true,
    }))

    try {
      // Send to API
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: content,
          agent: session.agent,
          session_id: activeSessionId,
          history: session.messages.slice(-10), // Last 10 messages for context
        }),
      })

      if (!response.ok) throw new Error('Failed to send message')

      const data = await response.json()

      // Add assistant message
      const assistantMessage: ChatMessage = {
        id: `msg-${Date.now()}-response`,
        role: 'assistant',
        content: data.response,
        timestamp: new Date().toISOString(),
        agent: session.agent,
        metadata: data.metadata,
      }

      set((state) => {
        const sessions = state.sessions.map((s) =>
          s.id === activeSessionId
            ? { ...s, messages: [...s.messages, assistantMessage], updated_at: new Date().toISOString() }
            : s
        )
        localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions))
        return { sessions, isLoading: false }
      })
    } catch (error) {
      set({ error: (error as Error).message, isLoading: false })
    }
  },

  addMessage: (sessionId, message) => {
    set((state) => ({
      sessions: state.sessions.map((s) =>
        s.id === sessionId ? { ...s, messages: [...s.messages, message] } : s
      ),
    }))
  },

  loadSessions: () => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) {
        const sessions = JSON.parse(stored)
        set({ sessions })
      }
    } catch (error) {
      console.error('Failed to load chat sessions:', error)
    }
  },

  setWsConnected: (connected) => set({ wsConnected: connected }),

  setAvailableAgents: (agents) => set({ availableAgents: agents }),
}))

// WebSocket hook for chat
export function useChatWebSocket() {
  const { setWsConnected, addMessage, activeSessionId } = useChatStore()
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    let reconnectTimeout: NodeJS.Timeout

    const connect = () => {
      const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`
      
      wsRef.current = new WebSocket(wsUrl)

      wsRef.current.onopen = () => {
        console.log('Chat WebSocket connected')
        setWsConnected(true)
      }

      wsRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          
          if (message.type === 'chat_message' && activeSessionId) {
            addMessage(activeSessionId, message.payload)
          }
        } catch (error) {
          console.error('Chat WebSocket message error:', error)
        }
      }

      wsRef.current.onclose = () => {
        console.log('Chat WebSocket disconnected')
        setWsConnected(false)
        reconnectTimeout = setTimeout(connect, 3000)
      }

      wsRef.current.onerror = () => {
        wsRef.current?.close()
      }
    }

    connect()

    return () => {
      clearTimeout(reconnectTimeout)
      wsRef.current?.close()
    }
  }, [setWsConnected, addMessage, activeSessionId])

  return wsRef
}
