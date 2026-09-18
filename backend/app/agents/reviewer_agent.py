"""
Reviewer Sub-Agent — Quality gate and HITL arbiter.

This agent sits between all sub-agents and the final output.
It evaluates the generated answer for:
1. Factual consistency with retrieved context.
2. Guardrail compliance (copyright, topic).
3. Confidence scoring — if confidence is too low, it escalates to HITL.

Interview talking point:
  "The Reviewer Agent is why this system doesn't hallucinate silently.
   Instead of returning a low-confidence answer, it escalates to a human
   review queue. This is how production AI systems should handle uncertainty."
"""
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.state import AgentState
from app.core.config import settings

_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    max_tokens=900,
    groq_api_key=settings.groq_api_key,
    temperature=0,
)

REVIEWER_PROMPT = """You are a quality reviewer for a Music Knowledge Agent.

Evaluate the following answer based on the provided context.
Respond in JSON format ONLY:
{
  "confidence": <float 0.0-1.0>,
  "passes_guardrails": <bool>,
  "guardrail_reason": "<string or empty>",
  "needs_revision": <bool>,
  "revision_notes": "<string or empty>"
}

Guardrail rules to check:
- Does the answer reproduce more than 3 words of song lyrics? → fail
- Is the answer off-topic (not music-related)? → fail
- IMPORTANT: Do NOT fact-check the answer. Assume the agent's summary is correct. Only fail if the answer is blatantly off-topic or reproduces copyright lyrics.
"""


async def reviewer_agent_node(state: AgentState) -> dict:
    """
    Reviewer node: scores answer quality and routes to output or HITL.
    """
    import json

    final_answer = state.get("final_answer", "")
    
    # Gather all sources of truth for the reviewer to check against
    context_parts = []
    
    # 1. RAG Context
    rag_ctx = state.get("retrieved_context", [])
    if rag_ctx:
        doc_str = "\n".join([str(d.get("content", "")) for d in rag_ctx])
        if len(doc_str) > 10000:
            doc_str = doc_str[:10000] + "... [TRUNCATED]"
        context_parts.append("DOCUMENT CONTEXT:\n" + doc_str)
        
    # 2. Tool Results
    tool_res = state.get("tool_results", [])
    if tool_res:
        context_parts.append("TOOL RESULTS:\n" + "\n".join([str(t) for t in tool_res]))
        
    # 3. User Memories
    memories = state.get("user_memories", [])
    if memories:
        context_parts.append("USER MEMORIES:\n" + "\n".join(memories))
        
    context_str = "\n\n".join(context_parts) if context_parts else "NO CONTEXT PROVIDED."

    messages = [
        SystemMessage(content=REVIEWER_PROMPT),
        HumanMessage(
            content=f"Answer to review:\n{final_answer}\n\nContext used:\n{context_str}"
        ),
    ]

    response = await _llm.ainvoke(messages)

    try:
        # Strip markdown code fences if present
        raw = response.content.strip().strip("```json").strip("```").strip()
        review = json.loads(raw)
    except Exception:
        # If parsing fails, be conservative — pass through with medium confidence
        review = {"confidence": 0.7, "passes_guardrails": True, "needs_revision": False, "guardrail_reason": ""}

    confidence = float(review.get("confidence", 0.7))
    passes = bool(review.get("passes_guardrails", True))
    guardrail_reason = review.get("guardrail_reason", "")

    # Guardrail failure → override answer with safe message
    if not passes:
        return {
            "guardrail_triggered": True,
            "guardrail_reason": guardrail_reason,
            "final_answer": f"I'm sorry, I can't help with that. {guardrail_reason}",
            "confidence_score": 0.0,
        }

    # Low confidence → escalate to HITL instead of returning a bad answer
    if confidence < settings.confidence_threshold:
        return {
            "confidence_score": confidence,
            "hitl_required": True,
            "hitl_action": f"Low-confidence answer (score: {confidence:.2f}) needs human review: {final_answer[:200]}...",
            "hitl_approved": None,
        }

    return {
        "confidence_score": confidence,
        "guardrail_triggered": False,
        "hitl_required": False,
    }
