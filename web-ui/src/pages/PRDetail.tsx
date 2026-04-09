import { usePRDetail } from '../hooks/useGitHub'

interface PRDetailProps {
  prNumber: number
  onBack: () => void
}

export function PRDetail({ prNumber, onBack }: PRDetailProps) {
  const { pr, reviews, comments, loading, error, refresh } = usePRDetail(prNumber)

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const getReviewStateIcon = (state: string) => {
    switch (state) {
      case 'APPROVED':
        return '✅ Approved'
      case 'CHANGES_REQUESTED':
        return '❌ Changes Requested'
      case 'COMMENTED':
        return '💬 Commented'
      default:
        return state
    }
  }

  const getReviewStateClass = (state: string) => {
    switch (state) {
      case 'APPROVED':
        return 'bg-green-100 text-green-800'
      case 'CHANGES_REQUESTED':
        return 'bg-red-100 text-red-800'
      case 'COMMENTED':
        return 'bg-gray-100 text-gray-800'
      default:
        return 'bg-gray-100'
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  if (error || !pr) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="text-red-500 text-xl mb-2">⚠️ Error</div>
          <p className="text-gray-600">{error || 'PR not found'}</p>
          <div className="mt-4 space-x-2">
            <button
              onClick={refresh}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Retry
            </button>
            <button
              onClick={onBack}
              className="px-4 py-2 bg-gray-200 text-gray-800 rounded hover:bg-gray-300"
            >
              Back
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      {/* Back Button */}
      <button
        onClick={onBack}
        className="mb-4 px-3 py-1 text-sm text-gray-600 hover:text-blue-600 flex items-center gap-1"
      >
        ← Back to list
      </button>

      {/* PR Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
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
            
            <h1 className="text-2xl font-bold text-gray-900 mb-2">
              {pr.title}
            </h1>

            <div className="flex items-center gap-4 text-sm text-gray-600">
              <div className="flex items-center gap-2">
                <img
                  src={pr.user.avatar_url}
                  alt={pr.user.login}
                  className="w-5 h-5 rounded-full"
                />
                <span className="font-medium">{pr.user.login}</span>
              </div>
              <span>opened {formatDate(pr.created_at)}</span>
              <span>updated {formatDate(pr.updated_at)}</span>
            </div>
          </div>

          <a
            href={pr.html_url}
            target="_blank"
            rel="noopener noreferrer"
            className="px-4 py-2 bg-gray-900 text-white rounded-lg hover:bg-gray-800 flex items-center gap-2"
          >
            View on GitHub ↗
          </a>
        </div>

        {/* Stats */}
        <div className="mt-6 flex items-center gap-6 pt-4 border-t border-gray-200">
          <div className="flex items-center gap-2">
            <span className="text-green-600 font-medium">+{pr.additions}</span>
            <span className="text-red-600 font-medium">-{pr.deletions}</span>
            <span className="text-gray-500">lines</span>
          </div>
          <div className="flex items-center gap-2 text-gray-600">
            <span>📁</span>
            <span>{pr.changed_files} files changed</span>
          </div>
          <div className="flex items-center gap-2 text-gray-600">
            <span>💬</span>
            <span>{pr.comments} comments</span>
          </div>
        </div>

        {/* Branch Info */}
        <div className="mt-4 flex items-center gap-2 text-sm">
          <span className="text-gray-500">Want to merge</span>
          <code className="px-2 py-1 bg-blue-50 text-blue-700 rounded font-mono">
            {pr.head.ref}
          </code>
          <span className="text-gray-500">into</span>
          <code className="px-2 py-1 bg-gray-100 text-gray-700 rounded font-mono">
            {pr.base.ref}
          </code>
        </div>
      </div>

      {/* Reviews */}
      {reviews.length > 0 && (
        <div className="mt-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Reviews ({reviews.length})
          </h2>
          <div className="space-y-3">
            {reviews.map((review) => (
              <div
                key={review.id}
                className="bg-white rounded-lg shadow-sm border border-gray-200 p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-gray-900">
                      {review.user.login}
                    </span>
                    <span className={`px-2 py-0.5 text-xs rounded-full ${getReviewStateClass(review.state)}`}>
                      {getReviewStateIcon(review.state)}
                    </span>
                  </div>
                  <span className="text-sm text-gray-500">
                    {formatDate(review.submitted_at)}
                  </span>
                </div>
                {review.body && (
                  <div className="mt-2 text-gray-700 whitespace-pre-wrap">
                    {review.body}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Comments */}
      {comments.length > 0 && (
        <div className="mt-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Review Comments ({comments.length})
          </h2>
          <div className="space-y-3">
            {comments.map((comment) => (
              <div
                key={comment.id}
                className="bg-white rounded-lg shadow-sm border border-gray-200 p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-gray-900">
                      {comment.user.login}
                    </span>
                    <code className="px-2 py-0.5 bg-gray-100 text-gray-700 text-xs rounded">
                      {comment.path}:{comment.line}
                    </code>
                  </div>
                  <span className="text-sm text-gray-500">
                    {formatDate(comment.created_at)}
                  </span>
                </div>
                <div className="text-gray-700 whitespace-pre-wrap">
                  {comment.body}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {reviews.length === 0 && comments.length === 0 && (
        <div className="mt-6 text-center py-12 bg-gray-50 rounded-lg">
          <div className="text-4xl mb-2">📝</div>
          <p className="text-gray-500">No reviews or comments yet</p>
        </div>
      )}
    </div>
  )
}
