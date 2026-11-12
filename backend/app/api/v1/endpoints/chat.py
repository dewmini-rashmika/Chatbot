"""
Chat SSE (Server-Sent Events) streaming endpoint.
This is the core endpoint that powers the real-time chat interface.

Why SSE over WebSockets?
- SSE is simpler and perfect for one-directional streaming (server -> client).
- Native browser support -- no extra library needed in React.
- Works through standard HTTP proxies and load balancers.
"""

import json
import time
import uuid
import structlog
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.future import select
from langchain_core.messages import HumanMessage
from app.agents.graph import get_graph
from app.core.dependencies import get_current_user
from app.core.logging_config import log_evaluation
from app.db.database import get_db, AsyncSessionLocal
from app.guardrails.guardrails import check_input_guardrails, check_output_guardrails
from app.models.models import Conversation, Message, User
from app.schemas.schemas import ChatRequest, HITLDecisionRequest
from app.services.memory_service import LongTermMemoryService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


async def _stream_agent_response(
    graph,
    initial_state: dict,
    config: dict,
):
    """
    Generator that streams agent execution events as SSE.
    Each event is a JSON string with an 'event' type and 'data' payload.
    """
    async def send_event(event_type: str, data: dict) -> str:
        return f"data: {json.dumps({'event': event_type, 'data': data})}\n\n"

    final_answer_parts: list[str] = []
    sources_captured:   list[str] = []

    yield await send_event("agent_start", {"message": "Routing your question to the right specialist..."})

    async for event in graph.astream_events(initial_state, config=config, version="v1"):
        kind = event.get("event", "")
        name = event.get("name", "")

        # Stream LLM tokens as they arrive
        if kind == "on_llm_stream":
            chunk = event.get("data", {}).get("chunk", {})
            token = ""
            if hasattr(chunk, "content"):
                token = chunk.content
            elif isinstance(chunk, dict):
                token = chunk.get("content", "")
            if token:
                final_answer_parts.append(token)
                yield await send_event("token", {"token": token})

        # Announce tool calls to the frontend
        elif kind == "on_tool_start":
            tool_name = event.get("data", {}).get("input", {})
            yield await send_event("tool_start", {"tool": name, "input": str(tool_name)[:200]})

        # Capture sources from chain outputs
        elif kind == "on_chain_end":
            d = event.get("data", {}).get("output", {})
            if isinstance(d, dict):
                for src in d.get("retrieved_context", []):
                    source = src.get("source", src.get("_source", ""))
                    if source and source not in sources_captured:
                        sources_captured.append(source)

    # Emit the done event with the full answer and sources
    final_answer = "".join(final_answer_parts)
    yield await send_event("done", {
        "answer":  final_answer,
        "sources": sources_captured[:5],
    })


@router.post("")
async def stream_chat(
    payload: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Main streaming chat endpoint.
    Returns a Server-Sent Events stream of agent execution events.

    Message persistence strategy:
    - User message is saved + committed BEFORE streaming starts (safe, synchronous).
    - Assistant message is saved AFTER streaming ends via a BackgroundTask that
      opens its own DB session (the request session will be closed by then).
    """
    # ── 1. Validate conversation belongs to user ──────────────────────────────
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == payload.conversation_id,
            Conversation.user_id == current_user.id,
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # ── 2. Input guardrails ───────────────────────────────────────────────────
    guard = check_input_guardrails(payload.message)
    if not guard.passed:
        async def _blocked_stream():
            yield f"data: {json.dumps({'event': 'token', 'data': {'token': guard.reason}})}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'data': {'answer': guard.reason, 'sources': []}})}\n\n"
        return StreamingResponse(_blocked_stream(), media_type="text/event-stream")

    # ── 3. Save user message ──────────────────────────────────────────────────
    user_msg = Message(
        id=uuid.uuid4(),
        conversation_id=conv.id,
        role="user",
        content=payload.message,
    )
    db.add(user_msg)
    await db.commit()

    # ── 4. Recall long-term memories for personalization ──────────────────────
    memory_service = LongTermMemoryService()
    user_memories = await memory_service.recall(str(current_user.id), db)

    # ── 5. Build initial graph state ──────────────────────────────────────────
    graph        = get_graph()
    thread_id    = conv.thread_id
    config       = {"configurable": {"thread_id": thread_id}}
    start_time   = time.time()
    initial_state = {
        "messages":          [HumanMessage(content=payload.message)],
        "user_memories":     user_memories,
        "hitl_required":     False,
        "hitl_approved":     False,
        "guardrail_triggered": False,
    }

    async def _stream_and_save():
        """Wraps the SSE generator and persists the assistant reply when done."""
        final_parts: list[str] = []
        final_sources: list[str] = []

        async for chunk in _stream_agent_response(graph, initial_state, config):
            yield chunk
            # Parse token/done events to capture the full answer
            try:
                parsed = json.loads(chunk.removeprefix("data: ").strip())
                if parsed.get("event") == "token":
                    final_parts.append(parsed["data"].get("token", ""))
                elif parsed.get("event") == "done":
                    final_sources = parsed["data"].get("sources", [])
            except Exception:
                pass

        # Save the assistant reply after streaming
        full_answer = "".join(final_parts)
        latency = round((time.time() - start_time) * 1000, 2)

        # Output guardrail check
        out_guard = check_output_guardrails(full_answer)
        if not out_guard.passed:
            full_answer = out_guard.reason

        background_tasks.add_task(
            _save_assistant_message,
            conversation_id=str(conv.id),
            content=full_answer,
            sources=final_sources,
            latency_ms=latency,
            user_query=payload.message,
        )

    return StreamingResponse(_stream_and_save(), media_type="text/event-stream")


async def _save_assistant_message(
    conversation_id: str,
    content: str,
    sources: list[str],
    latency_ms: float,
    user_query: str,
) -> None:
    """
    Background task: persist the assistant reply to the DB after streaming ends.
    Opens its own session so it runs safely outside the original request scope.
    """
    async with AsyncSessionLocal() as session:
        try:
            msg = Message(
                id=uuid.uuid4(),
                conversation_id=uuid.UUID(conversation_id),
                role="assistant",
                content=content,
                extra_data={"sources": [{"name": s} for s in sources], "latency_ms": latency_ms},
            )
            session.add(msg)
            await session.commit()

            log_evaluation(
                query=user_query,
                response=content,
                sources=sources,
                latency_ms=latency_ms,
                thread_id=conversation_id,
            )
        except Exception as exc:
            logger.warning("chat.save_failed", error=str(exc))
            await session.rollback()


@router.post("/hitl-decision")
async def hitl_decision(
    payload: HITLDecisionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Resume a paused HITL conversation after human approval/rejection.
    Called by the frontend's Approve/Reject buttons.
    """
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == payload.conversation_id,
            Conversation.user_id == current_user.id,
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    graph  = get_graph()
    config = {"configurable": {"thread_id": conv.thread_id}}

    if payload.approved:
        await graph.aupdate_state(
            config,
            {"hitl_approved": True, "hitl_required": False},
        )
        # Resume graph execution
        async for _ in graph.astream(None, config=config):
            pass
        return {"status": "approved", "message": "Action approved and executed."}
    else:
        await graph.aupdate_state(
            config,
            {"hitl_approved": False, "hitl_required": False,
             "final_answer": "Action cancelled based on your decision."},
        )
        return {"status": "rejected", "message": "Action cancelled."}
