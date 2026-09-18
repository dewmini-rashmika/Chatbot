import { useChatStore } from '../../store/chatStore'
import { hitlAPI } from '../../services/api'
import { CheckCircle, XCircle, AlertTriangle } from 'lucide-react'

import { v4 as uuidv4 } from 'uuid'

/**
 * HITL (Human-in-the-Loop) Modal.
 * Appears when the agent pauses and requests human approval for a write action
 * or a low-confidence answer.
 * 
 * Why this matters for the CV:
 * This is the most visible proof that the system is controllable and safe.
 * Enterprise clients require explicit human approval for any agentic write action.
 */
export default function HITLModal() {
  const { hitlPending, setHITLPending, addMessage, activeConversationId } = useChatStore()

  if (!hitlPending) return null

  const decide = async (approved: boolean) => {
    try {
      const res = await hitlAPI.decide(hitlPending.thread_id, approved)
      if (activeConversationId) {
        addMessage(activeConversationId, {
          id: uuidv4(),
          role: 'assistant',
          content: res.data.content,
          sources: res.data.sources,
          isStreaming: false
        })
      }
    } finally {
      setHITLPending(null)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-amber-500/30 rounded-2xl p-6 max-w-md w-full shadow-2xl">
        {/* Header */}
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 bg-amber-500/20 rounded-full flex items-center justify-center">
            <AlertTriangle className="w-5 h-5 text-amber-400" />
          </div>
          <div>
            <h3 className="text-white font-semibold">Human Approval Required</h3>
            <p className="text-gray-400 text-xs">The agent needs your permission to continue</p>
          </div>
        </div>

        {/* Action description */}
        <div className="bg-gray-800 rounded-xl p-4 mb-6 border border-gray-700">
          <p className="text-sm text-gray-300 leading-relaxed">{hitlPending.action}</p>
        </div>

        {/* Actions */}
        <div className="flex gap-3">
          <button
            onClick={() => decide(false)}
            className="flex-1 flex items-center justify-center gap-2 bg-gray-800 hover:bg-red-900/30 border border-gray-700 hover:border-red-700 text-gray-300 hover:text-red-300 rounded-xl py-3 text-sm font-medium transition-colors"
          >
            <XCircle className="w-4 h-4" />
            Reject
          </button>
          <button
            onClick={() => decide(true)}
            className="flex-1 flex items-center justify-center gap-2 bg-violet-600 hover:bg-violet-500 text-white rounded-xl py-3 text-sm font-medium transition-colors"
          >
            <CheckCircle className="w-4 h-4" />
            Approve
          </button>
        </div>
      </div>
    </div>
  )
}
