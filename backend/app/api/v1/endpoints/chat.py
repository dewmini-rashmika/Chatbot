"""
Chat SSE (Server-Sent Events) streaming endpoint.
This is the core endpoint that powers the real-time chat interface.

Why SSE over WebSockets?
- SSE is simpler and perfect for one-directional streaming (server -> client).
- Native browser support -- no extra library needed in React.
- Works through proxies and load balancers more reliably than WebSockets.
- WebSockets would be used for true bidirectional comms (e.g., voice).
"""
import json
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

import time
from app.agents.graph import get_graph
from app.core.dependencies import get_current_user
from app.db.database import AsyncSessionLocal, get_db
from app.guardrails.guardrails import check_input_guardrails, check_output_guardrails
from app.models.models import Conversation, Message, User
from app.schemas.schemas import ChatRequest, HITLDecisionRequest
from app.services.memory_service import LongTermMemoryService
from app.core.logging_config import log_evaluation

router = APIRouter(prefix="/chat", tags=["Chat"])
memory_service = LongTermMemoryService()


async def _save_assistant_message(conversation_id: uuid.UUID, content: str, sources: list):
    """
    Background task: persist the assistant reply to the DB after streaming ends.
    Opens its own session so it runs safely outside the original request scope.
    """
    async with AsyncSessionLocal() as db:
        try:
            msg = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=content,
                extra_data={"sources": sources} if sources else None,
            )
            db.add(msg)
            await db.commit()
        except Exception:
            await db.rollback()


async def _stream_agent_response(
    user_message: str,
    thread_id: str,
    user_id: str,
    user_memories: list[str],
):
    """
    Generator that streams agent execution events as SSE.
    Each event is a JSON string with an 'event' type and 'data' payload.
    """
    start_time = time.time()

    async def send_event(event: str, data: str | dict) -> str:
        payload = json.dumps({"event": event, "data": data})
        return f"data: {payload}\n\n"

    # -- Input guardrail check -----------------------------------------------
    guard = check_input_guardrails(user_message)
    if not guard.passed:
        yield await send_event("guardrail_blocked", guard.reason)
        yield await send_event("done", "")
        latency = (time.time() - start_time) * 1000
        log_evaluation(thread_id, user_message, guard.reason, [], latency)
        return

    yield await send_event("thinking", "Routing your question to the right specialist...")

    graph = await get_graph()

    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "user_id": user_id,
        "thread_id": thread_id,
        "user_memories": user_memories,
        "current_agent": "",
        "task_description": user_message,
        "retrieved_context": [],
        "tool_results": [],
        "confidence_score": 0.0,
        "guardrail_triggered": False,
        "guardrail_reason": "",
        "hitl_required": False,
        "hitl_action": "",
        "hitl_approved": None,
        "final_answer": "",
        "sources": [],
    }

    config = {"configurable": {"thread_id": thread_id}}

    # Stream events from LangGraph
    async for event in graph.astream_events(initial_state, config=config, version="v2"):
        kind = event.get("event", "")
        name = event.get("name", "")

        if kind == "on_chain_start" and name in ("rag_agent", "tool_agent", "supervisor"):
            yield await send_event("agent_start", f"🤖 {name.replace('_', ' ').title()} is working...")

        elif kind == "on_llm_stream":
            chunk = event.get("data", {}).get("chunk", {})
            if hasattr(chunk, "content") and chunk.content:
                yield await send_event("token", chunk.content)

        elif kind == "on_tool_start":
            tool_name = event.get("data", {}).get("input", {})
            yield await send_event("tool_start", f"🔧 Using tool: {name}")

        elif kind == "on_tool_end":
            yield await send_event("tool_end", f"✅ Tool completed: {name}")

        elif kind == "on_chain_end" and name in ("reviewer_agent", "tool_agent"):
            state = event.get("data", {}).get("output", {})
            if state and state.get("hitl_required"):
                yield await send_event("hitl_required", {
                    "action": state.get("hitl_action", ""),
                    "thread_id": thread_id,
                })
                latency = (time.time() - start_time) * 1000
                log_evaluation(thread_id, user_message, "[HITL PAUSED]", [], latency)
                return  # Pause -- frontend must call /hitl-decision

    # Get the final state
    final_state = await graph.aget_state(config)
    final_answer = final_state.values.get("final_answer", "")
    sources = final_state.values.get("sources", [])

    # -- Output guardrail check -----------------------------------------------
    out_guard = check_output_guardrails(final_answer)
    if not out_guard.passed:
        yield await send_event("guardrail_blocked", out_guard.reason)
        latency = (time.time() - start_time) * 1000
        log_evaluation(thread_id, user_message, out_guard.reason, [], latency)
    else:
        yield await send_event("final_answer", {
            "content": final_answer,
            "sources": sources,
        })
        latency = (time.time() - start_time) * 1000
        log_evaluation(thread_id, user_message, final_answer, sources, latency)

    yield await send_event("done", "")


@router.post("/stream")
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
      opens its own DB session, avoiding session-scope conflicts with SSE.
    """
    # Verify conversation belongs to user
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == payload.conversation_id,
            Conversation.user_id == current_user.id,
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Persist user message and commit immediately (before streaming begins)
    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=payload.message,
    )
    db.add(user_msg)
    await db.commit()

    # Load long-term memories for personalization
    user_memories = await memory_service.recall(current_user.id, db)

    # Schedule background task to save assistant reply once streaming ends.
    # We pass a coroutine factory; FastAPI's BackgroundTasks will call it after response.
    # We capture conv.id and thread_id now (before session closes).
    conv_id = conv.id
    thread_id = conv.thread_id

    async def _save_reply_after_stream():
        """
        Runs the full stream and then persists the assistant message.
        This is NOT called as a BackgroundTask directly -- instead we wrap
        the generator so the save happens naturally at stream end.
        """
        pass  # handled inline in the wrapped generator below

    async def _stream_and_save():
        """Wraps the SSE generator and persists the assistant reply when done."""
        final_answer_parts: list[str] = []
        sources_captured: list = []

        async for chunk in _stream_agent_response(
            user_message=payload.message,
            thread_id=thread_id,
            user_id=str(current_user.id),
            user_memories=user_memories,
        ):
            # Intercept final_answer event to capture content + sources for DB
            try:
                line = chunk
                if line.startswith("data: "):
                    parsed = json.loads(line[6:])
                    if parsed.get("event") == "final_answer":
                        d = parsed.get("data", {})
                        final_answer_parts.append(d.get("content", ""))
                        sources_captured = d.get("sources", [])
                    elif parsed.get("event") == "guardrail_blocked":
                        final_answer_parts.append("⚠️ " + str(parsed.get("data", "")))
            except Exception:
                pass
            yield chunk

        # After stream ends, persist assistant message directly
        if final_answer_parts:
            await _save_assistant_message(
                conv_id,
                "".join(final_answer_parts),
                sources_captured,
            )

    return StreamingResponse(
        _stream_and_save(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering for SSE
        },
    )


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
    graph = await get_graph()
    config = {"configurable": {"thread_id": payload.thread_id}}

    # Inject the human's decision into the state and resume
    await graph.aupdate_state(
        config,
        {"hitl_approved": payload.approved},
        as_node="hitl_node",
    )

    # Run the graph until it finishes (it will be quick since it's just tool/reviewer)
    async for event in graph.astream_events(None, config=config, version="v2"):
        pass

    # Get the final state
    final_state = await graph.aget_state(config)
    final_answer = final_state.values.get("final_answer", "")
    sources = final_state.values.get("sources", [])

    from app.guardrails.guardrails import check_output_guardrails
    out_guard = check_output_guardrails(final_answer)
    
    if not out_guard.passed:
        final_answer = "⚠️ " + out_guard.reason
        sources = []
        
    # Get conversation to save the message
    result = await db.execute(
        select(Conversation).where(
            Conversation.thread_id == payload.thread_id,
            Conversation.user_id == current_user.id,
        )
    )
    conv = result.scalar_one_or_none()
    
    if conv:
        await _save_assistant_message(conv.id, final_answer, sources)
    
    return {"content": final_answer, "sources": sources}
