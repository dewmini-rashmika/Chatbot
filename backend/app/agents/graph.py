"""
LangGraph Multi-Agent Graph Assembly.

This module wires all agents into a compiled LangGraph StateGraph.
The graph handles:
- Node execution order via edges
- Conditional routing via edge functions
- Postgres-backed conversation checkpointing (for HITL resume)

Graph topology:
  supervisor → (conditional) → rag_agent | tool_agent | chat_agent | creative_agent | hitl_node
  rag_agent  → reviewer_agent
  tool_agent → (conditional) → hitl_node | reviewer_agent
  reviewer_agent → (conditional) → END | hitl_node
  chat_agent     → END  (bypasses reviewer)
  creative_agent → END  (bypasses reviewer)
"""

import contextlib
import structlog
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.agents.state import AgentState
from app.agents.supervisor import supervisor_node, route_after_supervisor
from app.agents.rag_agent import rag_agent_node
from app.agents.tool_agent import tool_agent_node
from app.agents.chat_agent import chat_agent_node
from app.agents.creative_agent import creative_agent_node
from app.agents.reviewer_agent import reviewer_agent_node
from app.core.config import settings

logger = structlog.get_logger(__name__)

# Singleton compiled graph (initialised once at startup)
_graph = None


async def hitl_node(state: AgentState) -> dict:
    """
    HITL pause node.
    LangGraph will interrupt_before this node, giving the frontend
    a chance to call /conversations/hitl-decision with approve/reject.
    """
    return {}


def build_graph(checkpointer) -> StateGraph:
    """
    Assemble and compile the multi-agent LangGraph.

    Graph topology:
    supervisor → (conditional) → rag_agent | tool_agent | chat_agent | creative_agent | hitl_node
    rag_agent  → reviewer_agent
    tool_agent → (conditional) → hitl_node | reviewer_agent
    reviewer_agent → (conditional) → END | hitl_node
    chat_agent     → END
    creative_agent → END
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("supervisor",      supervisor_node)
    builder.add_node("rag_agent",       rag_agent_node)
    builder.add_node("tool_agent",      tool_agent_node)
    builder.add_node("chat_agent",      chat_agent_node)
    builder.add_node("creative_agent",  creative_agent_node)
    builder.add_node("reviewer_agent",  reviewer_agent_node)
    builder.add_node("hitl_node",       hitl_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_entry_point("supervisor")

    # ── Supervisor → specialist agents (conditional) ──────────────────────────
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "rag_agent":      "rag_agent",
            "tool_agent":     "tool_agent",
            "chat_agent":     "chat_agent",
            "creative_agent": "creative_agent",
            "hitl_node":      "hitl_node",
        },
    )

    # ── RAG agent → reviewer ──────────────────────────────────────────────────
    builder.add_edge("rag_agent", "reviewer_agent")

    # ── Tool agent → hitl or reviewer (conditional) ───────────────────────────
    def route_after_tool(state: AgentState) -> str:
        return "hitl_node" if state.get("hitl_required") else "reviewer_agent"

    builder.add_conditional_edges(
        "tool_agent",
        route_after_tool,
        {
            "hitl_node":     "hitl_node",
            "reviewer_agent": "reviewer_agent",
        },
    )

    # ── Chat / Creative → END (bypass reviewer) ───────────────────────────────
    builder.add_edge("chat_agent",     END)
    builder.add_edge("creative_agent", END)

    # ── Reviewer → END or HITL ────────────────────────────────────────────────
    def route_after_reviewer(state: AgentState) -> str:
        """Conditional edge: where to go after the Reviewer Agent runs."""
        return "hitl_node" if state.get("hitl_required") else END

    builder.add_conditional_edges(
        "reviewer_agent",
        route_after_reviewer,
        {
            "hitl_node": "hitl_node",
            END:         END,
        },
    )

    # ── HITL → tool_agent (on resume after approval) ──────────────────────────
    builder.add_edge("hitl_node", "tool_agent")

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["hitl_node"],
    )


async def init_graph() -> None:
    """Initialize the graph with a Postgres checkpointer at app startup."""
    global _graph
    stack = contextlib.AsyncExitStack()
    checkpointer = await stack.enter_async_context(
        AsyncPostgresSaver.from_conn_string(settings.database_url)
    )
    await checkpointer.setup()
    _graph = build_graph(checkpointer)
    logger.info("graph.initialized", supervisor_model=settings.supervisor_model,
                rag_model=settings.rag_model, reviewer_model=settings.reviewer_model)


def get_graph():
    """Returns the singleton compiled graph (initialized once at startup)."""
    if _graph is None:
        raise RuntimeError("Graph not initialized. Call init_graph() at startup.")
    return _graph
