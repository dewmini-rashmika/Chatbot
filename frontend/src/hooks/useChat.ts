import { useAuthStore } from '../store/authStore'
import { useChatStore } from '../store/chatStore'
import { v4 as uuidv4 } from 'uuid'

/**
 * Custom hook for streaming chat via Server-Sent Events.
 * Handles all SSE event types: token, agent_start, tool_start/end,
 * hitl_required, guardrail_blocked, final_answer, done.
 */
export function useChat() {
  const token = useAuthStore((s) => s.accessToken)
  const { addMessage, updateLastMessage, finalizeLastMessage, setHITLPending } = useChatStore()

  const sendMessage = async (conversationId: string, message: string) => {
    // Add user message immediately
    addMessage(conversationId, {
      id: uuidv4(),
      role: 'user',
      content: message,
    })

    // Add a placeholder streaming assistant message
    const assistantId = uuidv4()
    addMessage(conversationId, {
      id: assistantId,
      role: 'assistant',
      content: '',
      isStreaming: true,
    })

    const response = await fetch('/api/v1/chat/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ message, conversation_id: conversationId }),
    })

    if (!response.body) return

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          const parsed = JSON.parse(line.slice(6))
          const { event, data } = parsed

          switch (event) {
            case 'token':
              updateLastMessage(conversationId, data)
              break

            case 'final_answer':
              finalizeLastMessage(conversationId, data.sources, data.content)
              break

            case 'hitl_required':
              setHITLPending(data)
              finalizeLastMessage(conversationId)
              break

            case 'guardrail_blocked':
              updateLastMessage(conversationId, `⚠️ ${data}`)
              finalizeLastMessage(conversationId)
              break

            case 'agent_start':
            case 'thinking':
              // Update status text during thinking — only if empty
              const msgs = useChatStore.getState().messages[conversationId]
              if (msgs && msgs.length > 0 && msgs[msgs.length - 1].content === '') {
                updateLastMessage(conversationId, `_${data}_\n\n`)
              }
              break

            case 'done':
              finalizeLastMessage(conversationId)
              break
          }
        } catch {
          // Malformed SSE line — skip
        }
      }
    }
  }

  return { sendMessage }
}
