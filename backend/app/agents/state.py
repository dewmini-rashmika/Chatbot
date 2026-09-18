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
    - LangGraph serializes this to JSON for the Postgres checkpointer,
      enabling conversation resumption across sessions.
    """
    # Core chat messages (uses add_messages reducer to append, not overwrite)
    messages: Annotated[list, add_messages]
    
    # User context
    user_id: str
    thread_id: str
    
    # Long-term memory facts injected at session start
    user_memories: list[str]
    
    # Routing & delegation
    current_agent: str          # which sub-agent is active
    task_description: str       # what the supervisor asked the sub-agent to do
    
    # Tool & retrieval results
    retrieved_context: list[dict[str, Any]]   # RAG results (vector + SQL)
    tool_results: list[dict[str, Any]]         # external tool call results
    
    # Quality & safety signals
    confidence_score: float     # Reviewer agent's confidence in the answer
    guardrail_triggered: bool   # Did a safety check fire?
    guardrail_reason: str       # Why it fired
    
    # HITL state
    hitl_required: bool         # Should we pause for human approval?
    hitl_action: str            # What action needs approval?
    hitl_approved: bool | None  # Human's decision (None = pending)
    
    # Final output
    final_answer: str
    sources: list[dict[str, Any]]  # RAG sources for citations
