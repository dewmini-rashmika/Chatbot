import { useState } from 'react'
import { PlusCircle, MessageSquare, Trash2, LogOut, Music } from 'lucide-react'
import { v4 as uuidv4 } from 'uuid'
import { conversationsAPI } from '../../services/api'
import { useChatStore } from '../../store/chatStore'
import { useAuthStore } from '../../store/authStore'
import clsx from 'clsx'

export default function Sidebar() {
  const {
    conversations,
    setConversations,
    activeConversationId,
    setActiveConversation,
    setMessages,
    loadedConversations,
  } = useChatStore()
  const { user, logout } = useAuthStore()
  const [creating, setCreating] = useState(false)

  const createConversation = async () => {
    setCreating(true)
    try {
      const res = await conversationsAPI.create()
      const newConv = res.data
      setConversations([newConv, ...conversations])
      setActiveConversation(newConv.id)
      // New conversation starts with empty messages — no fetch needed
      setMessages(newConv.id, [])
    } finally {
      setCreating(false)
    }
  }

  const selectConversation = async (id: string) => {
    setActiveConversation(id)

    // Only fetch from API if we haven't loaded this conversation yet
    if (!loadedConversations.has(id)) {
      try {
        const msgRes = await conversationsAPI.getMessages(id)
        const msgs = msgRes.data.messages.map((m: any) => ({
          id: m.id ?? uuidv4(),
          role: m.role as 'user' | 'assistant',
          content: m.content,
          sources: m.sources ?? [],
        }))
        setMessages(id, msgs)
      } catch {
        // Non-critical — show empty chat if fetch fails
        setMessages(id, [])
      }
    }
  }

  const deleteConversation = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    await conversationsAPI.delete(id)
    const updated = conversations.filter((c) => c.id !== id)
    setConversations(updated)
    if (activeConversationId === id && updated.length > 0) {
      selectConversation(updated[0].id)
    }
  }

  return (
    <aside className="w-64 bg-gray-900 border-r border-gray-800 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-8 h-8 bg-violet-600 rounded-lg flex items-center justify-center">
            <Music className="w-4 h-4 text-white" />
          </div>
          <span className="font-semibold text-white text-sm">Music Agent</span>
        </div>
        <button
          onClick={createConversation}
          disabled={creating}
          className="w-full flex items-center gap-2 bg-violet-600 hover:bg-violet-500 disabled:opacity-50 text-white text-sm font-medium rounded-lg px-3 py-2 transition-colors"
        >
          <PlusCircle className="w-4 h-4" />
          {creating ? 'Creating...' : 'New Chat'}
        </button>
      </div>

      {/* Conversation List */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {conversations.length === 0 ? (
          <p className="text-gray-500 text-xs text-center mt-4">No conversations yet</p>
        ) : (
          conversations.map((conv) => (
            <button
              key={conv.id}
              onClick={() => selectConversation(conv.id)}
              className={clsx(
                'w-full flex items-center gap-2 rounded-lg px-3 py-2 text-left text-sm transition-colors group',
                activeConversationId === conv.id
                  ? 'bg-gray-800 text-white'
                  : 'text-gray-400 hover:bg-gray-800/50 hover:text-white'
              )}
            >
              <MessageSquare className="w-4 h-4 shrink-0" />
              <span className="flex-1 truncate">{conv.title}</span>
              <button
                onClick={(e) => deleteConversation(e, conv.id)}
                className="opacity-0 group-hover:opacity-100 p-0.5 text-gray-500 hover:text-red-400 transition-opacity"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </button>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-gray-800">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 bg-violet-900 rounded-full flex items-center justify-center text-xs text-violet-300 font-medium">
            {user?.username?.[0]?.toUpperCase() ?? 'U'}
          </div>
          <span className="flex-1 text-gray-400 text-xs truncate">{user?.username}</span>
          <button onClick={logout} className="text-gray-500 hover:text-red-400 transition-colors">
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </div>
    </aside>
  )
}
