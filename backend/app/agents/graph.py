"""
LangGraph Multi-Agent Graph Assembly.

This module wires all agents into a compiled LangGraph StateGraph.
The graph handles:
- Node execution order via edges
- Conditional routing via edge functions
- HITL breakpoints (interrupt_before)
- Postgres checkpointing for conversation persistence + resumption
"""
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


async def hitl_node(state: AgentState) -> dict:
    """
    HITL pause node.
    LangGraph will interrupt_before this node, giving the frontend
    a chance to call /conversations/hitl-decision with approve/reject.
    """
    if state.get("hitl_approved") is True:
        # If it was a write action from tool_agent, return there to execute.
        # Otherwise, the human approved a low-confidence answer, so we finish.
        action = state.get("hitl_action", "")
        next_agent = "tool_agent" if "write action" in action else "FINISH"
        return {"hitl_required": False, "current_agent": next_agent}
    elif state.get("hitl_approved") is False:
        # Human rejected — return a safe message
        return {
            "hitl_required": False,
            "final_answer": "Action cancelled based on your decision.",
            "current_agent": "FINISH",
        }
    # Still waiting — stay in this state
    return {}


def route_after_reviewer(state: AgentState) -> str:
    """Conditional edge: where to go after the Reviewer Agent runs."""
    if state.get("guardrail_triggered"):
        return END
    if state.get("hitl_required") and state.get("hitl_approved") is None:
        return "hitl_node"
    return END


async def build_graph(checkpointer: AsyncPostgresSaver) -> StateGraph:
    """
    Assemble and compile the multi-agent LangGraph.
    
    Graph topology:
    supervisor → (conditional) → rag_agent | tool_agent | chat_agent | creative_agent | hitl_node
    rag_agent  → reviewer_agent → (conditional) → END | hitl_node
    tool_agent → reviewer_agent → (conditional) → END | hitl_node
    chat_agent → END
    creative_agent → END
    hitl_node  → (conditional) → tool_agent | END
    """
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("rag_agent", rag_agent_node)
    builder.add_node("tool_agent", tool_agent_node)
    builder.add_node("chat_agent", chat_agent_node)
    builder.add_node("creative_agent", creative_agent_node)
    builder.add_node("reviewer_agent", reviewer_agent_node)
    builder.add_node("hitl_node", hitl_node)

    # Entry point
    builder.set_entry_point("supervisor")

    # Supervisor routes to sub-agents
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "rag_agent": "rag_agent",
            "tool_agent": "tool_agent",
            "chat_agent": "chat_agent",
            "creative_agent": "creative_agent",
            "hitl_node": "hitl_node",
            "reviewer_agent": "reviewer_agent",
        },
    )
    # Sub-agents flow to reviewer
    builder.add_edge("rag_agent", "reviewer_agent")
    
    def route_after_tool(state: AgentState) -> str:
        if state.get("hitl_required") and state.get("hitl_approved") is None:
            return "hitl_node"
        return "reviewer_agent"
        
    builder.add_conditional_edges(
        "tool_agent",
        route_after_tool,
        {"hitl_node": "hitl_node", "reviewer_agent": "reviewer_agent"}
    )
    
    # Non-factual agents flow directly to END to bypass strict factual reviewer
    builder.add_edge("chat_agent", END)
    builder.add_edge("creative_agent", END)

    # Reviewer conditionally ends or escalates to HITL
    builder.add_conditional_edges(
        "reviewer_agent",
        route_after_reviewer,
        {END: END, "hitl_node": "hitl_node"},
    )

    # HITL can resume into tool_agent or end
    builder.add_conditional_edges(
        "hitl_node",
        lambda s: s.get("current_agent") if s.get("current_agent") in ["tool_agent", "FINISH"] else END,
        {"tool_agent": "tool_agent", "FINISH": END, END: END},
    )

    # Compile with Postgres checkpointer + HITL interrupt
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["hitl_node"],  # Pause before HITL node — frontend must resume
    )

    return graph


# Singleton graph instance (initialized at startup)
_graph = None


async def get_graph() -> StateGraph:
    """Returns the singleton compiled graph (initialized once at startup)."""
    global _graph
    if _graph is None:
        raise RuntimeError("Graph not initialized. Call init_graph() at startup.")
    return _graph


from contextlib import AsyncExitStack

async def init_graph(database_url: str, stack: AsyncExitStack) -> None:
    """Initialize the graph with a Postgres checkpointer at app startup."""
    global _graph
    
    # from_conn_string is an async context manager in langgraph-checkpoint-postgres 2.x
    checkpointer = await stack.enter_async_context(
        AsyncPostgresSaver.from_conn_string(database_url)
    )
    await checkpointer.setup()  # Creates the checkpointing tables
    _graph = await build_graph(checkpointer)
