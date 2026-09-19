import { useEffect, useState } from 'react'
import { conversationsAPI } from '../services/api'
import { useChatStore } from '../store/chatStore'
import { useAuthStore } from '../store/authStore'
import Sidebar from '../components/layout/Sidebar'
import ChatWindow from '../components/chat/ChatWindow'
import HITLModal from '../components/chat/HITLModal'
import { v4 as uuidv4 } from 'uuid'

export default function DashboardPage() {
  const { setConversations, setActiveConversation, setMessages, activeConversationId } = useChatStore()
  const logout = useAuthStore((s) => s.logout)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadConversations = async () => {
      try {
        const res = await conversationsAPI.list()
        const convs = res.data.conversations
        setConversations(convs)

        if (convs.length > 0) {
          const firstId = convs[0].id
          setActiveConversation(firstId)

          // Fetch and restore message history for the first conversation
          try {
            const msgRes = await conversationsAPI.getMessages(firstId)
            const msgs = msgRes.data.messages.map((m: any) => ({
              id: m.id ?? uuidv4(),
              role: m.role as 'user' | 'assistant',
              content: m.content,
              sources: m.sources ?? [],
            }))
            setMessages(firstId, msgs)
          } catch {
            // Non-critical: messages just won't be restored for this session
          }
        }
      } catch {
        logout()
      } finally {
        setLoading(false)
      }
    }
    loadConversations()
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-violet-400 text-lg animate-pulse">Loading your music agent...</div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-gray-950 overflow-hidden">
      <Sidebar />
      <main className="flex-1 flex flex-col min-w-0">
        {activeConversationId ? (
          <ChatWindow conversationId={activeConversationId} />
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-center px-8">
            <div className="text-6xl mb-4">🎵</div>
            <h2 className="text-2xl font-bold text-white mb-2">Music Knowledge & Discovery Agent</h2>
            <p className="text-gray-400 max-w-md">
              Ask me anything about music theory, artist history, album deep-dives, 
              or get real-time chart information. Start a new conversation in the sidebar.
            </p>
          </div>
        )}
      </main>
      <HITLModal />
    </div>
  )
}
