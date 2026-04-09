import { PullRequest } from '../hooks/useGitHub'

interface PRListProps {
  prs: PullRequest[]
  onSelect: (pr: PullRequest) => void
  selectedId?: number
}

export function PRList({ prs, onSelect, selectedId }: PRListProps) {
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const formatNumber = (num: number) => {
    if (num >= 1000) return `${(num / 1000).toFixed(1)}k`
    return num.toString()
  }

  if (prs.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        <div className="text-4xl mb-2">🔍</div>
        <p>No pull requests found</p>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {prs.map((pr) => (
        <div
          key={pr.id}
          onClick={() => onSelect(pr)}
          className={`
            p-4 rounded-lg cursor-pointer transition-all
            ${selectedId === pr.number 
              ? 'bg-blue-50 border-2 border-blue-500' 
              : 'bg-white border border-gray-200 hover:border-blue-300 hover:shadow-sm'
            }
          `}
        >
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-2">
              <span className={`
                px-2 py-1 text-xs font-medium rounded-full
                ${pr.state === 'open' 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-purple-100 text-purple-800'}
              `}>
                {pr.state === 'open' ? '🟢 Open' : '🟣 Closed'}
              </span>
              <span className="text-sm text-gray-500">
                #{pr.number}
              </span>
            </div>
            <span className="text-xs text-gray-400">
              {formatDate(pr.updated_at)}
            </span>
          </div>

          <h3 className="mt-2 font-medium text-gray-900 line-clamp-2">
            {pr.title}
          </h3>

          <div className="mt-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <img
                src={pr.user.avatar_url}
                alt={pr.user.login}
                className="w-5 h-5 rounded-full"
              />
              <span className="text-sm text-gray-600">{pr.user.login}</span>
            </div>

            <div className="flex items-center gap-3 text-sm">
              {pr.additions > 0 && (
                <span className="text-green-600">
                  +{formatNumber(pr.additions)}
                </span>
              )}
              {pr.deletions > 0 && (
                <span className="text-red-600">
                  -{formatNumber(pr.deletions)}
                </span>
              )}
              {pr.comments > 0 && (
                <span className="text-gray-500 flex items-center gap-1">
                  💬 {pr.comments}
                </span>
              )}
            </div>
          </div>

          <div className="mt-2 flex items-center gap-2 text-xs text-gray-500">
            <span className="bg-gray-100 px-2 py-0.5 rounded">
              {pr.head.ref} → {pr.base.ref}
            </span>
            <span>{pr.changed_files} files</span>
          </div>
        </div>
      ))}
    </div>
  )
}
