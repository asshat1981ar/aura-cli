interface StatusBadgeProps {
  status: 'idle' | 'busy' | 'error' | 'offline' | 'pending' | 'running' | 'completed' | 'failed'
}

const statusStyles = {
  idle: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
  busy: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
  error: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
  offline: 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400',
  pending: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
  running: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400 animate-pulse',
  completed: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
  failed: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
}

export function StatusBadge({ status }: StatusBadgeProps) {
  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize ${
        statusStyles[status]
      }`}
    >
      {status === 'running' && (
        <span className="w-1.5 h-1.5 bg-current rounded-full mr-1.5 animate-pulse-dot" />
      )}
      {status}
    </span>
  )
}
