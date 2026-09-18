import { useRef, useEffect, useState } from 'react'
import { Send } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useChatStore } from '../../store/chatStore'
import { useChat } from '../../hooks/useChat'
import clsx from 'clsx'

interface Props {
  conversationId: string
}

export default function ChatWindow({ conversationId }: Props) {
  const messages = useChatStore((s) => s.messages[conversationId] ?? [])
  const { sendMessage } = useChat()
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = async () => {
    const text = input.trim()
    if (!text || sending) return
    setInput('')
    setSending(true)
    try {
      await sendMessage(conversationId, text)
    } finally {
      setSending(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="text-4xl mb-3">🎸</div>
            <p className="text-gray-400">Ask me about music theory, artists, albums, or current charts!</p>
          </div>
        )}
        {messages.map((msg) => (
          <div key={msg.id} className={clsx('flex', msg.role === 'user' ? 'justify-end' : 'justify-start')}>
            <div
              className={clsx(
                'max-w-2xl rounded-2xl px-4 py-3 text-sm',
                msg.role === 'user'
                  ? 'bg-violet-600 text-white rounded-br-sm'
                  : 'bg-gray-800 text-gray-100 rounded-bl-sm'
              )}
            >
              {msg.role === 'assistant' ? (
                <>
                  <div className="prose prose-invert prose-sm max-w-none">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {msg.content || (msg.isStreaming ? '▋' : '')}
                    </ReactMarkdown>
                  </div>
                  {/* Source citations */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-2 pt-2 border-t border-gray-700">
                      <p className="text-xs text-gray-500 font-medium mb-1">Sources</p>
                      <div className="flex flex-wrap gap-1">
                        {msg.sources.slice(0, 4).map((s, i) => (
                          <span key={i} className="text-xs bg-gray-700 text-gray-300 rounded-md px-2 py-0.5 truncate max-w-[200px]">
                            {s.source}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <p>{msg.content}</p>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="px-4 py-4 border-t border-gray-800">
        <div className="flex items-end gap-3 bg-gray-800 rounded-2xl px-4 py-3 border border-gray-700 focus-within:border-violet-500 transition-colors">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about music theory, artists, albums, charts..."
            rows={1}
            className="flex-1 bg-transparent text-white placeholder-gray-500 text-sm resize-none focus:outline-none max-h-32"
            style={{ height: 'auto' }}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || sending}
            className="w-8 h-8 flex items-center justify-center bg-violet-600 hover:bg-violet-500 disabled:opacity-40 rounded-lg transition-colors shrink-0"
          >
            <Send className="w-4 h-4 text-white" />
          </button>
        </div>
        <p className="text-center text-gray-600 text-xs mt-2">
          This agent may produce errors. Guardrails are active for copyright protection.
        </p>
      </div>
    </div>
  )
}
