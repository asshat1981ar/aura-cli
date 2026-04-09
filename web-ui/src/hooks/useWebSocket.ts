/**
 * useWebSocket — WebSocket hook with exponential-backoff reconnection.
 *
 * Reconnection schedule (backoff formula):
 *   delay = min(1000 * 2^retryCount, maxRetryDelay)
 *
 *   Attempt 0 →  1 s
 *   Attempt 1 →  2 s
 *   Attempt 2 →  4 s
 *   Attempt 3 →  8 s
 *   Attempt 4 → 16 s
 *   Attempt 5+ → maxRetryDelay (default 30 s)
 *
 * Configure `options.maxRetryDelay` (ms) to tune the ceiling.
 * Set `options.reconnect = false` to disable automatic reconnection entirely.
 *
 * Returns `{ send, disconnect, connect, status, retryCount }`.
 * - `status`     — one of: 'connecting' | 'connected' | 'disconnected' | 'error'
 * - `retryCount` — number of reconnection attempts since last clean connect
 */
import { useEffect, useRef, useCallback, useState } from 'react'

export type WebSocketStatus = 'connecting' | 'connected' | 'disconnected' | 'error'

interface WebSocketOptions {
  onMessage: (data: unknown) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Event) => void
  /** Set to false to disable automatic reconnection. Defaults to true. */
  reconnect?: boolean
  /**
   * @deprecated Ignored when exponential backoff is active.
   * Use `maxRetryDelay` to configure the backoff ceiling instead.
   */
  reconnectInterval?: number
  /** Maximum delay (ms) between reconnection attempts. Default: 30 000 (30 s). */
  maxRetryDelay?: number
}

export function useWebSocket(url: string, options: WebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>()
  const retryCountRef = useRef(0)
  const isConnectingRef = useRef(false)

  // Keep a stable ref to options so callbacks never stale-close over old values,
  // and so `connect` itself doesn't need options as a dependency (avoids re-mount loops).
  const optionsRef = useRef(options)
  useEffect(() => {
    optionsRef.current = options
  })

  const [status, setStatus] = useState<WebSocketStatus>('disconnected')
  const [retryCount, setRetryCount] = useState(0)

  // Stable connect — depends only on `url`; reads live options via ref.
  const connect = useCallback(() => {
    if (isConnectingRef.current || wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    isConnectingRef.current = true
    setStatus('connecting')

    try {
      const ws = new WebSocket(url)

      ws.onopen = () => {
        isConnectingRef.current = false
        retryCountRef.current = 0
        setRetryCount(0)
        setStatus('connected')
        optionsRef.current.onConnect?.()
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          optionsRef.current.onMessage(data)
        } catch {
          optionsRef.current.onMessage(event.data)
        }
      }

      ws.onclose = () => {
        isConnectingRef.current = false
        setStatus('disconnected')
        optionsRef.current.onDisconnect?.()

        if (optionsRef.current.reconnect !== false) {
          // Exponential backoff: 1 s, 2 s, 4 s, 8 s, 16 s, then cap at maxRetryDelay.
          const ceiling = optionsRef.current.maxRetryDelay ?? 30_000
          const delay = Math.min(1000 * Math.pow(2, retryCountRef.current), ceiling)
          reconnectTimeoutRef.current = setTimeout(() => {
            retryCountRef.current += 1
            setRetryCount(retryCountRef.current)
            connect()
          }, delay)
        }
      }

      ws.onerror = (error) => {
        isConnectingRef.current = false
        setStatus('error')
        optionsRef.current.onError?.(error)
      }

      wsRef.current = ws
    } catch (error) {
      isConnectingRef.current = false
      setStatus('error')
      console.error('WebSocket connection error:', error)
    }
  }, [url]) // url is the only dep — options are read via ref

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    wsRef.current?.close()
    wsRef.current = null
    retryCountRef.current = 0
    setRetryCount(0)
    setStatus('disconnected')
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

  return { send, disconnect, connect, status, retryCount }
}
