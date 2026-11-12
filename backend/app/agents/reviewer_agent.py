"""
Reviewer Sub-Agent — Quality gate and HITL arbiter.

This agent sits between all sub-agents and the final output.
It evaluates the generated answer for:
1. Factual consistency with retrieved context.
2. Guardrail compliance (copyright, topic).
3. Confidence scoring — if confidence is too low, it escalates to HITL.

LLM choice: groq/llama-3.3-70b-versatile (via LiteLLM)
  - Quality gate must be your most capable model.
  - Top-tier reasoning to detect subtle hallucinations and guardrail violations.
  - Swap to "anthropic/claude-3-5-sonnet-20241022" or "openai/gpt-4o"
    via REVIEWER_MODEL env var for the absolute highest quality gate.
"""

import json
import structlog
from app.agents.state import AgentState
from app.core.llm_factory import call_llm, get_model_for_role
from app.core.config import settings

logger = structlog.get_logger(__name__)

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

Scoring guide:
- confidence 0.9-1.0: Answer is fully supported by context, no hallucinations detected.
- confidence 0.7-0.9: Mostly supported, minor gaps acceptable.
- confidence 0.5-0.7: Partially supported, some claims unverified — flag for review.
- confidence < 0.5:   Answer contradicts or ignores context — escalate to human.

Guardrail checks:
- passes_guardrails = false if the answer reproduces copyrighted lyrics verbatim.
- passes_guardrails = false if the answer discusses non-music topics (hacking, medical, etc.).
- passes_guardrails = false if the answer contains harmful, hateful, or inappropriate content.
"""


async def reviewer_agent_node(state: AgentState) -> dict:
    """
    Reviewer node: scores answer quality and routes to output or HITL.
    """
    final_answer      = state.get("final_answer", "")
    retrieved_context = state.get("retrieved_context", [])
    tool_results      = state.get("tool_results", [])
    user_memories     = state.get("user_memories", [])

    model = get_model_for_role("reviewer")
    logger.info("reviewer.start", model=model)

    # ── Build the review payload ──────────────────────────────────────────────
    context_parts: list[str] = []

    if retrieved_context:
        context_parts.append("DOCUMENT CONTEXT:\n" + "\n---\n".join(
            doc.get("content", doc.get("document", str(doc)))[:600]
            for doc in retrieved_context[:5]
        ))

    if tool_results:
        t = str(tool_results)
        if len(t) > 800:
            t = t[:800] + "... [TRUNCATED]"
        context_parts.append(f"TOOL RESULTS:\n{t}")

    if user_memories:
        context_parts.append("USER MEMORIES:\n" + "\n".join(f"- {m}" for m in user_memories))

    context_str = "\n\n".join(context_parts) if context_parts else "NO CONTEXT PROVIDED."

    llm_messages = [
        {"role": "system", "content": REVIEWER_PROMPT},
        {
            "role": "user",
            "content": (
                f"\n\nContext used:\n{context_str}"
                f"\n\nAnswer to review:\n{final_answer}"
            ),
        },
    ]

    # ── Call the judge LLM ────────────────────────────────────────────────────
    try:
        response = await call_llm(
            role="reviewer",
            messages=llm_messages,
            temperature=0.0,   # deterministic judgment
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        review = json.loads(raw)
        confidence     = float(review.get("confidence", 0.5))
        passes         = bool(review.get("passes_guardrails", True))
        guardrail_reason = review.get("guardrail_reason", "")

    except Exception as exc:
        logger.warning("reviewer.parse_error", error=str(exc))
        confidence = 0.5
        passes     = True
        guardrail_reason = ""

    logger.info("reviewer.scored", confidence=confidence, passes=passes, model=model)

    # ── Guardrail block ───────────────────────────────────────────────────────
    if not passes:
        blocked_answer = (
            f"I'm sorry, I can't help with that. {guardrail_reason}".strip()
        )
        return {
            "final_answer":    blocked_answer,
            "confidence_score": 0.0,
            "hitl_required":   False,
            "guardrail_triggered": True,
            "guardrail_reason": guardrail_reason,
        }

    # ── Low-confidence escalation to HITL ────────────────────────────────────
    if confidence < settings.confidence_threshold:
        logger.info(
            "reviewer.hitl_escalation",
            confidence=confidence,
            threshold=settings.confidence_threshold,
        )
        return {
            "confidence_score": confidence,
            "hitl_required":    True,
            "hitl_action":      f"Low-confidence answer (score: {confidence:.2f}) needs human review: {final_answer[:200]}",
        }

    # ── Answer passes quality gate ────────────────────────────────────────────
    return {
        "confidence_score": confidence,
        "hitl_required":    False,
        "guardrail_triggered": False,
    }
