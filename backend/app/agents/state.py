"""
LangGraph Agent State Definition.
Using TypedDict for type-safe state passing between all nodes in the graph.
This is the single source of truth for what data flows through the multi-agent system.
"""

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The shared state object passed between every node in the LangGraph.

    Why use a TypedDict state?
    - It makes the data contract between agents explicit and type-safe.
    - LangGraph serialises/deserialises it via the Postgres checkpointer.
    - Every node receives the full state and returns a partial update dict.
    """

    # ── Core conversation ─────────────────────────────────────────────────────
    # Annotated with add_messages so LangGraph appends rather than overwrites
    messages: Annotated[list, add_messages]

    # ── Routing ───────────────────────────────────────────────────────────────
    current_agent:    str   # which sub-agent is active ("rag_agent", etc.)
    task_description: str   # natural-language description of the delegated task

    # ── RAG context ───────────────────────────────────────────────────────────
    retrieved_context: list[dict]   # fused vector + SQL docs
    tool_results:      list[dict]   # raw tool call observations

    # ── Generated answer ──────────────────────────────────────────────────────
    final_answer: str

    # ── Quality gate (reviewer) ───────────────────────────────────────────────
    confidence_score: float

    # ── Guardrails ────────────────────────────────────────────────────────────
    guardrail_triggered: bool
    guardrail_reason:    str

    # ── Human-in-the-loop (HITL) ──────────────────────────────────────────────
    hitl_required: bool   # True → pause graph, ask human
    hitl_approved: bool   # True → human approved the pending action
    hitl_action:   str    # description of the action awaiting approval

    # ── Long-term memory ──────────────────────────────────────────────────────
    user_memories: list[str]   # recalled facts from previous sessions
