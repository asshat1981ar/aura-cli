import { useState, useEffect, useCallback, useRef } from 'react'

export interface PullRequest {
  id: number
  number: number
  title: string
  state: 'open' | 'closed'
  created_at: string
  updated_at: string
  user: {
    login: string
    avatar_url: string
  }
  head: {
    ref: string
    sha: string
  }
  base: {
    ref: string
  }
  additions: number
  deletions: number
  changed_files: number
  comments: number
  review_comments: number
  html_url: string
}

export interface PRReview {
  id: number
  body: string
  state: 'APPROVED' | 'CHANGES_REQUESTED' | 'COMMENTED'
  user: {
    login: string
  }
  submitted_at: string
}

export interface PRComment {
  id: number
  path: string
  line: number
  body: string
  user: {
    login: string
  }
  created_at: string
}

export interface PREvent {
  type: 'github_pr_event'
  payload: {
    event: string
    action: string
    pr_number: number
    pr_title: string
    repo: string
    sender: string
    status: string
  }
}

export function useGitHubWebSocket() {
  const [events, setEvents] = useState<PREvent[]>([])
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const ws = new WebSocket(`ws://${window.location.host}/ws`)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      console.log('GitHub WebSocket connected')
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'github_pr_event') {
        setEvents(prev => [data, ...prev].slice(0, 100)) // Keep last 100
      }
    }

    ws.onclose = () => {
      setConnected(false)
      console.log('GitHub WebSocket disconnected')
    }

    return () => {
      ws.close()
    }
  }, [])

  const sendPing = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'ping' }))
    }
  }, [])

  return { events, connected, sendPing }
}

export function usePRList() {
  const [prs, setPrs] = useState<PullRequest[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const { events, connected } = useGitHubWebSocket()

  const fetchPRs = useCallback(async () => {
    try {
      setLoading(true)
      const response = await fetch('/api/github/prs')
      if (!response.ok) throw new Error('Failed to fetch PRs')
      const data = await response.json()
      setPrs(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchPRs()
  }, [fetchPRs])

  // Refresh on WebSocket events
  useEffect(() => {
    if (events.length > 0) {
      const latestEvent = events[0]
      if (latestEvent.payload.event === 'pull_request') {
        fetchPRs()
      }
    }
  }, [events, fetchPRs])

  return { prs, loading, error, refresh: fetchPRs, connected }
}

export function usePRDetail(prNumber: number | null) {
  const [pr, setPr] = useState<PullRequest | null>(null)
  const [reviews, setReviews] = useState<PRReview[]>([])
  const [comments, setComments] = useState<PRComment[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchPRDetail = useCallback(async () => {
    if (!prNumber) return
    
    try {
      setLoading(true)
      
      const [prRes, reviewsRes, commentsRes] = await Promise.all([
        fetch(`/api/github/prs/${prNumber}`),
        fetch(`/api/github/prs/${prNumber}/reviews`),
        fetch(`/api/github/prs/${prNumber}/comments`),
      ])

      if (!prRes.ok) throw new Error('Failed to fetch PR')
      
      setPr(await prRes.json())
      setReviews(reviewsRes.ok ? await reviewsRes.json() : [])
      setComments(commentsRes.ok ? await commentsRes.json() : [])
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [prNumber])

  useEffect(() => {
    fetchPRDetail()
  }, [fetchPRDetail])

  return { pr, reviews, comments, loading, error, refresh: fetchPRDetail }
}

export function usePRFilters() {
  const [state, setState] = useState<'all' | 'open' | 'closed'>('open')
  const [search, setSearch] = useState('')
  const [sortBy, setSortBy] = useState<'updated' | 'created'>('updated')

  const filterPRs = useCallback((prs: PullRequest[]) => {
    return prs
      .filter(pr => state === 'all' || pr.state === state)
      .filter(pr => 
        search === '' || 
        pr.title.toLowerCase().includes(search.toLowerCase()) ||
        pr.user.login.toLowerCase().includes(search.toLowerCase())
      )
      .sort((a, b) => {
        const dateA = new Date(sortBy === 'updated' ? a.updated_at : a.created_at)
        const dateB = new Date(sortBy === 'updated' ? b.updated_at : b.created_at)
        return dateB.getTime() - dateA.getTime()
      })
  }, [state, search, sortBy])

  return {
    filters: { state, search, sortBy },
    setters: { setState, setSearch, setSortBy },
    filterPRs,
  }
}
