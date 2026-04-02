import { useEffect, useRef, useCallback } from 'react'

interface WebSocketOptions {
  onMessage: (data: unknown) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Event) => void
  reconnect?: boolean
  reconnectInterval?: number
}

export function useWebSocket(url: string, options: WebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  const reconnectAttemptsRef = useRef(0)
  const isConnectingRef = useRef(false)

  const connect = useCallback(() => {
    if (isConnectingRef.current || wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    isConnectingRef.current = true

    try {
      const ws = new WebSocket(url)

      ws.onopen = () => {
        isConnectingRef.current = false
        reconnectAttemptsRef.current = 0
        options.onConnect?.()
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          options.onMessage(data)
        } catch {
          options.onMessage(event.data)
        }
      }

      ws.onclose = () => {
        isConnectingRef.current = false
        options.onDisconnect?.()

        if (options.reconnect !== false && reconnectAttemptsRef.current < 5) {
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current++
            connect()
          }, options.reconnectInterval || 3000)
        }
      }

      ws.onerror = (error) => {
        isConnectingRef.current = false
        options.onError?.(error)
      }

      wsRef.current = ws
    } catch (error) {
      isConnectingRef.current = false
      console.error('WebSocket connection error:', error)
    }
  }, [url, options])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    wsRef.current?.close()
    wsRef.current = null
  }, [])

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
      return true
    }
    return false
  }, [])

  useEffect(() => {
    connect()
    return disconnect
  }, [connect, disconnect])

  return { send, disconnect, connect }
}
