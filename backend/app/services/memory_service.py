"""
Long-Term Memory Service.
Extracts user preference facts from conversations and persists them
to PostgreSQL. Recalled at the start of each new session to personalize responses.

Why separate long-term memory from the conversation checkpointer?
- The checkpointer stores the full message history (short-term).
- Long-term memory stores EXTRACTED FACTS — a much smaller, distilled form
  that can be loaded quickly into any new conversation's system prompt.
- This mirrors how humans work: you don't replay every conversation,
  you remember key facts about a person.
"""
import uuid
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.config import settings
from app.models.models import LongTermMemory

_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    groq_api_key=settings.groq_api_key,
    temperature=0,
)

EXTRACTION_PROMPT = """You are a memory extractor for a music chatbot.
Analyze the conversation and extract any NEW, durable facts about the user's music preferences.

Examples of good facts to extract:
- "User loves 1950s bebop jazz"
- "User's favourite artist is Miles Davis"
- "User dislikes heavy metal"
- "User plays guitar and is interested in music theory"
- "User is learning counterpoint"

Rules:
- Only extract facts that are clearly stated or strongly implied.
- Do NOT extract temporary context ("user asked about X" — that's short-term).
- Return a JSON list of strings. Empty list [] if nothing durable was found.
- Maximum 5 new facts per conversation.
"""


class LongTermMemoryService:
    """Manages extraction and retrieval of per-user persistent memory facts."""

    async def extract_and_store(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        messages: list,
        db: AsyncSession,
    ) -> list[str]:
        """
        Run at end of conversation to extract and persist new user facts.
        """
        import json

        # Format recent messages for extraction
        formatted = "\n".join(
            f"{m.type.upper()}: {m.content}"
            for m in messages[-20:]  # Last 20 messages only
            if hasattr(m, "content")
        )

        response = await _llm.ainvoke([
            SystemMessage(content=EXTRACTION_PROMPT),
            HumanMessage(content=f"Conversation:\n{formatted}"),
        ])

        try:
            raw = response.content.strip().strip("```json").strip("```").strip()
            facts = json.loads(raw)
            if not isinstance(facts, list):
                facts = []
        except Exception:
            facts = []

        # Persist new facts to PostgreSQL
        new_memory_objects = []
        for fact in facts[:5]:
            mem = LongTermMemory(
                user_id=user_id,
                fact=str(fact),
                source_conversation_id=conversation_id,
            )
            db.add(mem)
            new_memory_objects.append(str(fact))

        await db.flush()
        return new_memory_objects

    async def recall(self, user_id: uuid.UUID, db: AsyncSession) -> list[str]:
        """
        Retrieve all stored facts for a user to inject into a new session.
        """
        result = await db.execute(
            select(LongTermMemory)
            .where(LongTermMemory.user_id == user_id)
            .order_by(LongTermMemory.created_at.desc())
            .limit(20)
        )
        memories = result.scalars().all()
        return [m.fact for m in memories]
