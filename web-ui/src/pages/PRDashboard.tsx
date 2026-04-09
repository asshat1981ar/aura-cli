import { useState } from 'react'
import { usePRList, usePRFilters, PullRequest } from '../hooks/useGitHub'
import { PRList } from '../components/PRList'
import { PRDetail } from './PRDetail'

export default function PRDashboard() {
  const { prs, loading, error, refresh, connected } = usePRList()
  const { filters, setters, filterPRs } = usePRFilters()
  const [selectedPR, setSelectedPR] = useState<PullRequest | null>(null)

  const filteredPRs = filterPRs(prs)

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="text-red-500 text-xl mb-2">⚠️ Error</div>
          <p className="text-gray-600">{error}</p>
          <button
            onClick={refresh}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex">
      {/* Sidebar - PR List */}
      <div className="w-96 border-r border-gray-200 flex flex-col bg-gray-50">
        {/* Header */}
        <div className="p-4 bg-white border-b border-gray-200">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-xl font-bold text-gray-900">
              Pull Requests
              <span className="ml-2 text-sm font-normal text-gray-500">
                ({filteredPRs.length})
              </span>
            </h1>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`}></span>
              <button
                onClick={refresh}
                className="p-1 text-gray-400 hover:text-blue-600 transition-colors"
                title="Refresh"
              >
                🔄
              </button>
            </div>
          </div>

          {/* Search */}
          <input
            type="text"
            placeholder="Search PRs..."
            value={filters.search}
            onChange={(e) => setters.setSearch(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />

          {/* Filters */}
          <div className="mt-3 flex items-center gap-2">
            <select
              value={filters.state}
              onChange={(e) => setters.setState(e.target.value as any)}
              className="px-3 py-1.5 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
            >
              <option value="open">Open</option>
              <option value="closed">Closed</option>
              <option value="all">All</option>
            </select>

            <select
              value={filters.sortBy}
              onChange={(e) => setters.setSortBy(e.target.value as any)}
              className="px-3 py-1.5 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
            >
              <option value="updated">Last Updated</option>
              <option value="created">Created</option>
            </select>
          </div>
        </div>

        {/* PR List */}
        <div className="flex-1 overflow-y-auto p-4">
          <PRList
            prs={filteredPRs}
            onSelect={setSelectedPR}
            selectedId={selectedPR?.number}
          />
        </div>
      </div>

      {/* Main Content - PR Detail */}
      <div className="flex-1 overflow-y-auto">
        {selectedPR ? (
          <PRDetail prNumber={selectedPR.number} onBack={() => setSelectedPR(null)} />
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-400">
              <div className="text-6xl mb-4">📋</div>
              <p className="text-lg">Select a pull request to view details</p>
              <p className="text-sm mt-2">
                {filteredPRs.length} PR{filteredPRs.length !== 1 ? 's' : ''} available
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
