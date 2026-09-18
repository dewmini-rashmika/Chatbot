from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage

from app.agents.state import AgentState
from app.core.config import settings

_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    max_tokens=900,
    groq_api_key=settings.groq_api_key,
    temperature=0.8,
    streaming=True,
)

CREATIVE_SYSTEM_PROMPT = """You are a highly creative Music Poet, Songwriting Assistant, and Aesthetic Guide.
Your role is to write original lyrics, compose poetry about music, and describe musical aesthetics in a highly evocative, artistic way.
- You create ORIGINAL content. Do not reproduce real copyrighted song lyrics.
- Be highly expressive, poetic, and atmospheric in your descriptions of music.
- If the user asks for a song, write full verses, choruses, and describe the instrumentation/vibe.
- FORMATTING RULE: When writing lyrics, ALWAYS use explicit line breaks (`\n`) for each line of the song. Do not write lyrics as a single continuous paragraph.
- Embrace artistic freedom and write beautiful prose.
"""

async def creative_agent_node(state: AgentState) -> dict:
    """
    Creative sub-agent node: handles original poetry, songwriting, and aesthetic descriptions.
    Routes directly to END to bypass strict factual reviewer checks.
    """
    messages = [SystemMessage(content=CREATIVE_SYSTEM_PROMPT)] + state["messages"][-6:]

    response = await _llm.ainvoke(messages)

    return {
        "final_answer": response.content,
        "current_agent": "FINISH",
        "messages": [AIMessage(content=response.content)],
    }
