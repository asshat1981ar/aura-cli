import { useAuthStore } from '../stores/authStore'

const API_BASE = '/api'

async function fetchWithAuth(url: string, options: RequestInit = {}) {
  const token = useAuthStore.getState().token

  const response = await fetch(`${API_BASE}${url}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` }),
      ...options.headers,
    },
  })

  if (response.status === 401) {
    useAuthStore.getState().logout()
    throw new Error('Unauthorized')
  }

  return response
}

export const api = {
  // Goals
  getGoals: () => fetchWithAuth('/goals').then((r) => r.json()),
  createGoal: (description: string) =>
    fetchWithAuth('/goals', {
      method: 'POST',
      body: JSON.stringify({ description }),
    }).then((r) => r.json()),
  cancelGoal: (id: string) =>
    fetchWithAuth(`/goals/${id}/cancel`, { method: 'POST' }),

  // Agents
  getAgents: () => fetchWithAuth('/agents').then((r) => r.json()),
  getAgentLogs: (id: string) =>
    fetchWithAuth(`/agents/${id}/logs`).then((r) => r.json()),

  // Stats
  getStats: () => fetchWithAuth('/stats').then((r) => r.json()),

  // Health
  getHealth: () => fetchWithAuth('/health').then((r) => r.json()),
}
