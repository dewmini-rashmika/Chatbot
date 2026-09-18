"""
Supervisor Agent — the orchestrator of all sub-agents.

Architecture decision:
  We use the Supervisor Pattern (not ReAct) because:
  - It gives explicit routing logic, making the system predictable and debuggable.
  - Each sub-agent is specialized and stateless — the Supervisor manages delegation.
  - Easier to extend: adding a new capability = adding a new sub-agent + routing rule.
"""
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.state import AgentState
from app.core.config import settings

# The Supervisor uses the faster Flash model for routing decisions
_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    max_tokens=900,
    groq_api_key=settings.groq_api_key,
    temperature=0,
)

SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor of a Music Knowledge & Discovery Agent.
Your job is to analyze the user's query and decide which specialist sub-agent should handle it.

Available sub-agents:
- "chat_agent": For small talk, casual greetings, or general conversational chit-chat that DOES NOT require looking up specific facts, reviews, or database records.
- "creative_agent": For writing original song lyrics, music poetry, or evocative descriptions of musical aesthetics. Use this whenever the user asks for creative or poetic output.
- "rag_agent": For questions about music theory, artist history, album reviews, genre history, 
  influences — anything requiring deep knowledge retrieval from documents or the music database.
- "tool_agent": For actions or live data — current charts, now playing, playlist modification,
  web search, reading specific URLs, real-time Spotify data, or anything requiring external API calls.
- "FINISH": When you have a final_answer ready to return to the user.

Respond with ONLY the sub-agent name to delegate to, nothing else.
"""


async def supervisor_node(state: AgentState) -> dict:
    """
    Supervisor node: reads the current state and decides which sub-agent to call next.
    Returns a state update with the 'current_agent' field set.
    """
    # If guardrails were triggered, end immediately
    if state.get("guardrail_triggered"):
        return {"current_agent": "FINISH"}

    # If HITL is required and not yet resolved, pause
    if state.get("hitl_required") and state.get("hitl_approved") is None:
        return {"current_agent": "HITL_WAIT"}

    # If we already have a confident final answer, finish
    if state.get("final_answer") and state.get("confidence_score", 0) >= settings.confidence_threshold:
        return {"current_agent": "FINISH"}

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=f"User query: {state['messages'][-1].content}"),
    ]

    response = await _llm.ainvoke(messages)
    next_agent = response.content.strip().lower()

    # Validate the routing decision
    valid_agents = {"rag_agent", "tool_agent", "chat_agent", "creative_agent", "finish"}
    if next_agent not in valid_agents:
        next_agent = "rag_agent"  # safe default
        
    # Prevent the supervisor from prematurely finishing before an answer is generated
    if next_agent == "finish" and not state.get("final_answer"):
        next_agent = "chat_agent"

    return {"current_agent": next_agent}


def route_after_supervisor(state: AgentState) -> str:
    """
    LangGraph conditional edge function.
    Maps the supervisor's decision to the next graph node.
    """
    agent = state.get("current_agent", "rag_agent")
    routing_map = {
        "rag_agent": "rag_agent",
        "tool_agent": "tool_agent",
        "chat_agent": "chat_agent",
        "creative_agent": "creative_agent",
        "finish": "reviewer_agent",
        "hitl_wait": "hitl_node",
    }
    return routing_map.get(agent, "rag_agent")
