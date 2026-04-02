interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

const sizes = {
  sm: 'w-4 h-4',
  md: 'w-8 h-8',
  lg: 'w-12 h-12',
}

export function LoadingSpinner({
  size = 'md',
  className = '',
}: LoadingSpinnerProps) {
  return (
    <div className={`${sizes[size]} ${className}`}>
      <svg
        className="animate-spin text-primary"
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        />
      </svg>
    </div>
  )
}

export function LoadingCard() {
  return (
    <div className="bg-card border rounded-xl p-6 animate-pulse">
      <div className="flex items-start justify-between">
        <div className="space-y-3">
          <div className="w-32 h-4 bg-muted rounded" />
          <div className="w-16 h-8 bg-muted rounded" />
          <div className="w-48 h-3 bg-muted rounded" />
        </div>
        <div className="w-12 h-12 bg-muted rounded-lg" />
      </div>
    </div>
  )
}

export function LoadingTable({ rows = 5 }: { rows?: number }) {
  return (
    <div className="bg-card border rounded-xl overflow-hidden animate-pulse">
      <div className="p-4 border-b bg-muted/50">
        <div className="flex gap-4">
          <div className="w-32 h-4 bg-muted rounded" />
          <div className="w-32 h-4 bg-muted rounded" />
          <div className="w-32 h-4 bg-muted rounded" />
        </div>
      </div>
      <div className="divide-y">
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="p-4 flex gap-4">
            <div className="w-8 h-8 bg-muted rounded-full" />
            <div className="flex-1 space-y-2">
              <div className="w-48 h-4 bg-muted rounded" />
              <div className="w-32 h-3 bg-muted rounded" />
            </div>
            <div className="w-16 h-6 bg-muted rounded-full" />
          </div>
        ))}
      </div>
    </div>
  )
}
