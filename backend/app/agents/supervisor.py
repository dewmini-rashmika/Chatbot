"""
Supervisor Agent — the orchestrator of all sub-agents.

Architecture decision:
  We use the Supervisor Pattern (not ReAct) because:
  - It gives explicit routing logic, making the system predictable and debuggable.
  - Each sub-agent is specialized and stateless — the Supervisor manages delegation.

LLM choice: groq/llama-3.3-70b-versatile (via LiteLLM)
  - Pure classification task: outputs one JSON object with an agent name.
  - Ultra-low latency at ~100 ms on Groq's LPU hardware.
  - Overkill to use a large frontier model just for routing.
"""

import json
import structlog
from app.agents.state import AgentState
from app.core.llm_factory import call_llm, get_model_for_role
from app.core.config import settings

logger = structlog.get_logger(__name__)

SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor of a Music Knowledge & Discovery Agent.
Your job is to analyze the user's query and decide which specialist sub-agent should handle it.

Available sub-agents:
- "chat_agent": For small talk, casual greetings, or general conversational chit-chat that DOES NOT require looking up specific music facts or data.
- "creative_agent": For requests to write original song lyrics, poetry about music, or describe musical aesthetics creatively. NOT for factual questions.
- "rag_agent": For factual questions about music history, theory, artists, albums, genres, awards, or chart positions. This agent searches the knowledge base.
- "tool_agent": For requests that need LIVE or REAL-TIME data — current Spotify charts, now-playing tracks, MusicBrainz lookups, or current web search.
- "finish": If the conversation is clearly over (user said goodbye, etc.).

Respond in JSON format ONLY:
{
  "agent": "<agent_name>",
  "reasoning": "<one sentence why>",
  "creative_intent": "<only if creative_agent, describe the creative task>"
}

Rules:
- Output ONLY valid JSON. No preamble, no markdown fences.
- Default to "rag_agent" for any music knowledge question.
- Use "tool_agent" only when real-time/live data is explicitly needed.
- Use "chat_agent" only for pure small talk with no music-fact requirement.
"""


async def supervisor_node(state: AgentState) -> dict:
    """
    Supervisor node: reads the current state and decides which sub-agent to call next.
    Returns a state update with the 'current_agent' and 'task_description' fields set.
    """
    messages = state["messages"]
    user_query = messages[-1].content if messages else ""

    # Build conversation for the supervisor
    llm_messages = [
        {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
        {"role": "user",   "content": f"User query: {user_query}"},
    ]

    model = get_model_for_role("supervisor")
    logger.info("supervisor.routing", query=user_query[:80], model=model)

    try:
        response = await call_llm(
            role="supervisor",
            messages=llm_messages,
            temperature=0.0,   # deterministic routing
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if the model added them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        decision = json.loads(raw)
        next_agent     = decision.get("agent", "rag_agent").lower()
        creative_intent = decision.get("creative_intent", "")

    except Exception as exc:
        logger.warning("supervisor.parse_error", error=str(exc), raw=locals().get("raw", ""))
        next_agent      = "rag_agent"
        creative_intent = ""

    # Normalise unknown agent names
    valid_agents = {"rag_agent", "tool_agent", "chat_agent", "creative_agent", "finish"}
    if next_agent not in valid_agents:
        next_agent = "rag_agent"

    logger.info("supervisor.decision", next_agent=next_agent, model=model)

    return {
        "current_agent":    next_agent,
        "task_description": creative_intent or user_query,
        "hitl_required":    False,
        "hitl_approved":    False,
        "guardrail_triggered": state.get("guardrail_triggered", False),
    }


def route_after_supervisor(state: AgentState) -> str:
    """
    LangGraph conditional edge function.
    Maps the supervisor's decision to the next graph node.
    """
    routing_map = {
        "rag_agent":      "rag_agent",
        "tool_agent":     "tool_agent",
        "chat_agent":     "chat_agent",
        "creative_agent": "creative_agent",
        "finish":         "reviewer_agent",
    }
    agent = state.get("current_agent", "rag_agent")
    return routing_map.get(agent, "hitl_node")
