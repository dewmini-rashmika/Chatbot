import { create } from 'zustand'

export interface Source {
  source: string
  id: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  isStreaming?: boolean
}

export interface Conversation {
  id: string
  thread_id: string
  title: string
  updated_at: string
}

interface ChatState {
  conversations: Conversation[]
  activeConversationId: string | null
  messages: Record<string, Message[]>
  /** Track which conversation IDs have already been fetched from the DB */
  loadedConversations: Set<string>
  hitlPending: { thread_id: string; action: string } | null
  setConversations: (convs: Conversation[]) => void
  setActiveConversation: (id: string) => void
  setMessages: (convId: string, msgs: Message[]) => void
  addMessage: (convId: string, msg: Message) => void
  updateLastMessage: (convId: string, chunk: string) => void
  finalizeLastMessage: (convId: string, sources?: Message['sources'], content?: string) => void
  setHITLPending: (data: ChatState['hitlPending']) => void
}

export const useChatStore = create<ChatState>((set) => ({
  conversations: [],
  activeConversationId: null,
  messages: {},
  loadedConversations: new Set(),
  hitlPending: null,

  setConversations: (convs) => set({ conversations: convs }),

  setActiveConversation: (id) =>
    set((s) => ({
      activeConversationId: id,
      messages: s.messages[id] ? s.messages : { ...s.messages, [id]: [] },
    })),

  setMessages: (convId, msgs) =>
    set((s) => ({
      messages: { ...s.messages, [convId]: msgs },
      loadedConversations: new Set([...s.loadedConversations, convId]),
    })),

  addMessage: (convId, msg) =>
    set((s) => ({
      messages: {
        ...s.messages,
        [convId]: [...(s.messages[convId] ?? []), msg],
      },
    })),

  updateLastMessage: (convId, chunk) =>
    set((s) => {
      const msgs = [...(s.messages[convId] ?? [])]
      const last = msgs[msgs.length - 1]
      if (last && last.isStreaming) {
        msgs[msgs.length - 1] = { ...last, content: last.content + chunk }
      }
      return { messages: { ...s.messages, [convId]: msgs } }
    }),

  finalizeLastMessage: (convId, sources, content) =>
    set((s) => {
      const msgs = [...(s.messages[convId] ?? [])]
      const last = msgs[msgs.length - 1]
      if (last && last.isStreaming) {
        msgs[msgs.length - 1] = {
          ...last,
          isStreaming: false,
          sources: sources ?? [],
          content: content !== undefined ? content : last.content,
        }
      }
      return { messages: { ...s.messages, [convId]: msgs } }
    }),

  setHITLPending: (data) => set({ hitlPending: data }),
}))
