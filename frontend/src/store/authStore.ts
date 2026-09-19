import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface AuthState {
  accessToken: string | null
  refreshToken: string | null
  user: { id: string; username: string; email: string } | null
  setAuth: (tokens: { access_token: string; refresh_token: string }, user: AuthState['user']) => void
  logout: () => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      accessToken: null,
      refreshToken: null,
      user: null,
      setAuth: (tokens, user) =>
        set({
          accessToken: tokens.access_token,
          refreshToken: tokens.refresh_token,
          user,
        }),
      logout: () => set({ accessToken: null, refreshToken: null, user: null }),
    }),
    { name: 'music-agent-auth' }
  )
)
