import axios from 'axios'
import { useAuthStore } from '../store/authStore'

const api = axios.create({ baseURL: '/api/v1' })

// Attach JWT token to every request
api.interceptors.request.use((config) => {
  const token = useAuthStore.getState().accessToken
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// Auto-logout on 401
api.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401) useAuthStore.getState().logout()
    return Promise.reject(err)
  }
)

export default api

// ── Auth ──────────────────────────────────────────────────────────────────────
export const authAPI = {
  register: (data: { email: string; username: string; password: string }) =>
    api.post('/auth/register', data),
  login: (data: { email: string; password: string }) =>
    api.post('/auth/login', data),
}

// ── Conversations ─────────────────────────────────────────────────────────────
export const conversationsAPI = {
  list: () => api.get('/conversations/'),
  create: (title?: string) => api.post('/conversations/', { title: title ?? 'New Conversation' }),
  get: (id: string) => api.get(`/conversations/${id}`),
  delete: (id: string) => api.delete(`/conversations/${id}`),
}

// ── HITL ──────────────────────────────────────────────────────────────────────
export const hitlAPI = {
  decide: (thread_id: string, approved: boolean, reason?: string) =>
    api.post('/chat/hitl-decision', { thread_id, approved, reason }),
}
