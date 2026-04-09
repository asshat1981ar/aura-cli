import { useState, useEffect } from 'react'

interface InFlightGoal {
  goal: string
  started_at: string
  started_at_formatted: string
  elapsed_seconds: number
  elapsed_formatted: string
  cycle_limit: number
}

export function GoalResumeCard() {
  const [inFlight, setInFlight] = useState<InFlightGoal | null>(null)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')

  const checkInFlight = async () => {
    try {
      const response = await fetch('/api/goals/in-flight')
      if (response.ok) {
        const data = await response.json()
        setInFlight(data.exists ? data.summary : null)
      }
    } catch (err) {
      console.error('Failed to check in-flight goal:', err)
    }
  }

  const handleResume = async (run: boolean = false) => {
    setLoading(true)
    setMessage('')
    
    try {
      const response = await fetch('/api/goals/resume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run }),
      })
      
      const data = await response.json()
      
      if (response.ok) {
        setMessage(data.message || (run ? 'Goal resumed and running!' : 'Goal re-queued!'))
        setInFlight(null)
      } else {
        setMessage(`Error: ${data.detail || 'Failed to resume'}`)
      }
    } catch (err) {
      setMessage(`Error: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    checkInFlight()
    // Poll every 10 seconds
    const interval = setInterval(checkInFlight, 10000)
    return () => clearInterval(interval)
  }, [])

  if (!inFlight) {
    return (
      <div className="bg-green-50 border border-green-200 rounded-lg p-4">
        <div className="flex items-center gap-2 text-green-700">
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
          <span className="font-medium">No interrupted goals</span>
        </div>
        <p className="text-green-600 text-sm mt-1">
          All goals have completed successfully.
        </p>
      </div>
    )
  }

  return (
    <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0">
          <svg className="w-6 h-6 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        
        <div className="flex-1 min-w-0">
          <h3 className="text-amber-900 font-medium">
            Interrupted Goal Detected
          </h3>
          
          <div className="mt-2 space-y-1 text-sm text-amber-800">
            <p className="truncate" title={inFlight.goal}>
              <span className="font-medium">Goal:</span> {inFlight.goal}
            </p>
            <p>
              <span className="font-medium">Started:</span> {inFlight.started_at_formatted}
            </p>
            <p>
              <span className="font-medium">Interrupted:</span> {inFlight.elapsed_formatted} ago
            </p>
            <p>
              <span className="font-medium">Cycle Limit:</span> {inFlight.cycle_limit}
            </p>
          </div>
          
          {message && (
            <div className={`mt-3 p-2 rounded text-sm ${
              message.includes('Error') 
                ? 'bg-red-100 text-red-700' 
                : 'bg-green-100 text-green-700'
            }`}>
              {message}
            </div>
          )}
          
          <div className="mt-4 flex flex-wrap gap-2">
            <button
              onClick={() => handleResume(false)}
              disabled={loading}
              className="px-4 py-2 bg-amber-100 text-amber-800 rounded-lg hover:bg-amber-200 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium transition-colors"
            >
              {loading ? 'Resuming...' : 'Re-queue Goal'}
            </button>
            
            <button
              onClick={() => handleResume(true)}
              disabled={loading}
              className="px-4 py-2 bg-amber-600 text-white rounded-lg hover:bg-amber-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium transition-colors"
            >
              {loading ? 'Running...' : 'Resume & Run'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
