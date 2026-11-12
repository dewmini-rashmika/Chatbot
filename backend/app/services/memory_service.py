"""
Long-Term Memory Service.
Extracts user preference facts from conversations and persists them
to PostgreSQL. Recalled at the start of each new session to personalize responses.

Why separate long-term memory from the conversation checkpointer?
- The checkpointer stores the full message history (short-term, per-session).
- Long-term memory stores distilled preference FACTS that survive across sessions.
- Keeping them separate avoids bloating the LangGraph state with ever-growing history.

LLM choice: groq/llama-3.1-8b-instant (via LiteLLM)
  - Simple JSON extraction task — no frontier model needed.
  - Runs asynchronously in the background after the user leaves.
  - Ultra-cheap and fast: ~50ms on Groq, fraction of a cent per call.
  - Swap via MEMORY_MODEL env var without touching this file.
"""

import json
import uuid
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc
from app.models.models import LongTermMemory
from app.core.llm_factory import call_llm, get_model_for_role

logger = structlog.get_logger(__name__)

EXTRACTION_PROMPT = """You are a memory extractor for a music chatbot.
Analyze the conversation and extract any NEW, durable facts about the user's music preferences.

Examples of good facts to extract:
- "User loves 1950s bebop jazz"
- "User's favourite artist is Miles Davis"
- "User dislikes heavy metal"
- "User plays guitar"
- "User is learning music theory"

Rules:
- Extract ONLY facts that will still be relevant in future conversations.
- Do NOT extract facts about the current query (e.g. "User asked about Beatles").
- Do NOT extract facts already well-known from the conversation context.
- Return a JSON array of strings. If no new facts found, return: []
- Output ONLY valid JSON. No preamble, no markdown fences.

Example output:
["User loves 1960s British Invasion rock", "User plays bass guitar"]
"""


class LongTermMemoryService:
    """Manages extraction and retrieval of per-user persistent memory facts."""

    async def extract_and_store(
        self,
        conversation_id: str,
        session: AsyncSession,
    ) -> None:
        """
        Run at end of conversation to extract and persist new user facts.
        Called asynchronously — does not block the response stream.
        """
        model = get_model_for_role("memory")
        logger.info("memory.extracting", conversation_id=conversation_id, model=model)

        try:
            # Fetch the full conversation messages from the DB
            # (This relies on the LangGraph checkpoint messages being accessible
            #  via the conversation_id — adapt to your actual message fetch logic)
            from app.models.models import Conversation
            result = await session.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
            conversation = result.scalar_one_or_none()
            if not conversation:
                return

            # Format the conversation history for the extractor
            # messages field stores the LangGraph messages as JSON
            messages_data = conversation.messages if hasattr(conversation, "messages") else []
            if not messages_data:
                return

            formatted = "\n".join(
                f"{m.get('type', m.get('role', 'unknown')).upper()}: {m.get('content', '')}"
                for m in messages_data
                if isinstance(m, dict)
            )

            if not formatted.strip():
                return

            llm_messages = [
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user",   "content": f"Conversation:\n{formatted}"},
            ]

            response = await call_llm(
                role="memory",
                messages=llm_messages,
                temperature=0.0,
                max_tokens=512,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            facts: list[str] = json.loads(raw)
            if not isinstance(facts, list):
                facts = []

            # Persist each new fact to the long_term_memory table
            new_memory_objects = [
                LongTermMemory(
                    id=uuid.uuid4(),
                    user_id=conversation.user_id,
                    fact=fact,
                )
                for fact in facts
                if isinstance(fact, str) and fact.strip()
            ]

            if new_memory_objects:
                session.add_all(new_memory_objects)
                await session.flush()
                logger.info(
                    "memory.stored",
                    count=len(new_memory_objects),
                    conversation_id=conversation_id,
                )

        except Exception as exc:
            logger.warning("memory.extraction_failed", error=str(exc))

    async def recall(
        self,
        user_id: str,
        session: AsyncSession,
        limit: int = 20,
    ) -> list[str]:
        """
        Retrieve all stored facts for a user to inject into a new session.
        Returns the most recent `limit` facts, newest first.
        """
        try:
            result = await session.execute(
                select(LongTermMemory)
                .where(LongTermMemory.user_id == user_id)
                .order_by(desc(LongTermMemory.created_at))
                .limit(limit)
            )
            memories = result.scalars().all()
            return [mem.fact for mem in memories if mem.fact]
        except Exception as exc:
            logger.warning("memory.recall_failed", error=str(exc))
            return []
