"""
Creative Sub-Agent — original songwriting, lyrics, and music poetry.

Handles requests for original creative writing about music.
Routes directly to END, bypassing the strict factual reviewer checks
(creative output cannot be "fact-checked" against a knowledge base).

LLM choice: groq/llama-3.3-70b-versatile (via LiteLLM)
  Swap via CREATIVE_MODEL env var.
"""

import structlog
from langchain_core.messages import AIMessage
from app.agents.state import AgentState
from app.core.llm_factory import call_llm, get_model_for_role

logger = structlog.get_logger(__name__)

CREATIVE_SYSTEM_PROMPT = """You are a Master Songwriter, Lyricist, and Music Poet with encyclopedic genre knowledge.
Your role is to write original lyrics, compose poetry about music, and describe musical aesthetics in a highly evocative, artistic way.

# CORE RULES
1. **Be Original:** NEVER reproduce real copyrighted song lyrics. Only write NEW, original content.
2. **Be Specific:** Use genre-accurate language, instruments, moods, and imagery.
3. **Be Artistic:** Prioritize emotional resonance, vivid imagery, and musicality of language.
4. **Respect Intent:** Match the user's requested genre, mood, era, and style as closely as possible.

# FORMAT
- For lyrics: include clear section labels (Verse 1, Chorus, Bridge, etc.)
- For poetry: use appropriate line breaks and stanza structure.
- For aesthetic descriptions: write in flowing, evocative prose.
"""


async def creative_agent_node(state: AgentState) -> dict:
    """
    Creative sub-agent node: handles original poetry, songwriting, and aesthetic descriptions.
    Routes directly to END to bypass strict factual reviewer checks.
    """
    messages         = state["messages"]
    task_description = state.get("task_description", "")
    user_memories    = state.get("user_memories", [])

    # Prefer task_description (set by supervisor) over raw message
    prompt = task_description or (messages[-1].content if messages else "")

    model = get_model_for_role("creative")
    logger.info("creative_agent.start", prompt=prompt[:80], model=model)

    # Inject any relevant user preferences
    memories_str = ""
    if user_memories:
        memories_str = "\n\nUSER MEMORIES:\n" + "\n".join(f"- {m}" for m in user_memories)

    llm_messages = [
        {"role": "system", "content": CREATIVE_SYSTEM_PROMPT},
        {"role": "user",   "content": f"{prompt}{memories_str}"},
    ]

    response = await call_llm(
        role="creative",
        messages=llm_messages,
        temperature=0.9,   # high creativity
    )
    answer = response.choices[0].message.content

    return {
        "messages":     [AIMessage(content=answer)],
        "final_answer": answer,
    }
