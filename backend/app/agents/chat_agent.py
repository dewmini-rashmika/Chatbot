"""
Chat Sub-Agent — casual conversation and general music chit-chat.

Handles small talk, greetings, and conversational queries that do NOT require
looking up specific facts. Routes directly to END, bypassing the reviewer.

LLM choice: groq/llama-3.3-70b-versatile (via LiteLLM)
  Swap via CHAT_MODEL env var.
"""

import structlog
from langchain_core.messages import AIMessage
from app.agents.state import AgentState
from app.core.llm_factory import call_llm, get_model_for_role

logger = structlog.get_logger(__name__)

CHAT_SYSTEM_PROMPT = """You are a friendly, conversational Music Knowledge Agent.
Your role is to handle small talk, greetings, and general music chit-chat.
- Be warm and welcoming.
- You can mention famous artists, genres, or basic music trivia using your general knowledge.
- Keep responses concise (1-3 sentences) unless the user asks for more.
- Do NOT make up specific facts, chart positions, or release dates.
"""


async def chat_agent_node(state: AgentState) -> dict:
    """
    Chat sub-agent node: handles small talk and general conversational queries
    without strictly enforcing RAG context checks.
    """
    messages   = state["messages"]
    user_query = messages[-1].content if messages else ""

    model = get_model_for_role("chat")
    logger.info("chat_agent.start", query=user_query[:80], model=model)

    llm_messages = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "user",   "content": user_query},
    ]

    response = await call_llm(
        role="chat",
        messages=llm_messages,
        temperature=0.8,
    )
    answer = response.choices[0].message.content

    return {
        "messages":     [AIMessage(content=answer)],
        "final_answer": answer,
    }
