from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage

from app.agents.state import AgentState
from app.core.config import settings

_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    max_tokens=900,
    groq_api_key=settings.groq_api_key,
    temperature=0.7,
    streaming=True,
)

CHAT_SYSTEM_PROMPT = """You are a friendly, conversational Music Knowledge Agent.
Your role is to handle small talk, greetings, and general music chit-chat.
- Be warm and welcoming.
- You can mention famous artists, genres, or basic music trivia using your general knowledge.
- Keep responses concise (1-3 sentences) unless asked otherwise.
- If the user asks for deep factual data, real-time info, or complex analysis, gently guide them so another specialized agent can handle it.
"""

async def chat_agent_node(state: AgentState) -> dict:
    """
    Chat sub-agent node: handles small talk and general conversational queries
    without strictly enforcing RAG context checks.
    """
    messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT)] + state["messages"][-6:]

    response = await _llm.ainvoke(messages)

    return {
        "final_answer": response.content,
        "current_agent": "FINISH",
        "messages": [AIMessage(content=response.content)],
    }
