"""
Chat SSE (Server-Sent Events) streaming endpoint.
This is the core endpoint that powers the real-time chat interface.

Why SSE over WebSockets?
- SSE is simpler and perfect for one-directional streaming (server → client).
- Native browser support — no extra library needed in React.
- Works through proxies and load balancers more reliably than WebSockets.
- WebSockets would be used for true bidirectional comms (e.g., voice).
"""
import json
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

import time
from app.agents.graph import get_graph
from app.core.dependencies import get_current_user
from app.db.database import get_db
from app.guardrails.guardrails import check_input_guardrails, check_output_guardrails
from app.models.models import Conversation, Message, User
from app.schemas.schemas import ChatRequest, HITLDecisionRequest
from app.services.memory_service import LongTermMemoryService
from app.core.logging_config import log_evaluation

router = APIRouter(prefix="/chat", tags=["Chat"])
memory_service = LongTermMemoryService()


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

    # ── Input guardrail check ──────────────────────────────────────────────────
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
                return  # Pause — frontend must call /hitl-decision

    # Get the final state
    final_state = await graph.aget_state(config)
    final_answer = final_state.values.get("final_answer", "")
    sources = final_state.values.get("sources", [])

    # ── Output guardrail check ─────────────────────────────────────────────────
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
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Main streaming chat endpoint.
    Returns a Server-Sent Events stream of agent execution events.
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

    # Persist user message
    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=payload.message,
    )
    db.add(user_msg)
    await db.flush()

    # Load long-term memories for personalization
    user_memories = await memory_service.recall(current_user.id, db)

    return StreamingResponse(
        _stream_agent_response(
            user_message=payload.message,
            thread_id=conv.thread_id,
            user_id=str(current_user.id),
            user_memories=user_memories,
        ),
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
        return {"content": "⚠️ " + out_guard.reason, "sources": []}
    
    return {"content": final_answer, "sources": sources}
