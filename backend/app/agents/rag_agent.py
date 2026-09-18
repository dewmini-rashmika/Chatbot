"""
RAG Sub-Agent — Hybrid Retrieval (Vector + Vectorless).

This agent handles all knowledge-retrieval tasks:
1. Vector RAG via ChromaDB — semantic similarity search for unstructured text.
2. Vectorless RAG via PostgreSQL Text-to-SQL — exact structured fact lookup.
3. Hybrid fusion via Reciprocal Rank Fusion (RRF) — combines both results.

Interview talking point:
  "We use RRF instead of simply concatenating results because it's a 
   rank-aware fusion algorithm — it handles the score distribution 
   differences between embedding cosine similarity and SQL relevance."
"""
import asyncio
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from app.agents.state import AgentState
from app.core.config import settings
from app.services.vector_store import VectorStoreService
from app.services.sql_rag import SQLRAGService

_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    max_tokens=900,
    groq_api_key=settings.groq_api_key,
    temperature=0.3,
    streaming=True,
)

RAG_SYSTEM_PROMPT = """You are an expert Music Knowledge Agent. You have access to:
- A comprehensive library of music theory textbooks and historical essays.
- A structured database of artist discographies, chart positions, and awards.
- Synthesized music reviews styled after various critical voices.
- Real-time web search results (Tavily/DDG) for current news and general artist info.

CRITICAL RULES:
1. Never reproduce song lyrics beyond 2-3 words (copyright protection).
2. Always cite the source document, database record, or web URL when answering.
3. If retrieval confidence is low, say so explicitly — do not hallucinate.
4. Distinguish clearly between your knowledge and what documents say.

When answering, use the retrieved context provided. If context is insufficient, 
say the confidence is low rather than guessing.
"""


def reciprocal_rank_fusion(
    vector_results: list[dict],
    sql_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) to merge vector and SQL rankings.
    
    Formula: score(d) = Σ 1/(k + rank(d))
    
    Why RRF:
    - Vector scores (cosine similarity 0.0-1.0) and SQL scores are not 
      comparable on the same scale.
    - RRF only uses rank positions, making it scale-invariant.
    - Consistently outperforms simple score combination in benchmarks.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
        doc_map[doc_id] = doc

    for rank, doc in enumerate(sql_results, start=1):
        doc_id = doc["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
        doc_map[doc_id] = {**doc_map.get(doc_id, {}), **doc}

    # Sort by fused score descending
    ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in ranked[:8]]


async def rag_agent_node(state: AgentState) -> dict:
    """
    RAG sub-agent node: retrieves relevant context using hybrid retrieval
    and generates an answer grounded in that context.
    """
    from app.tools.music_tools import search_web
    
    user_query = state["messages"][-1].content
    user_memories = state.get("user_memories", [])

    # Run vector, SQL retrieval, and web search in parallel for performance
    vector_service = VectorStoreService()
    sql_service = SQLRAGService()

    vector_results, sql_results, web_results = await asyncio.gather(
        vector_service.search(user_query, n_results=3),
        sql_service.search(user_query),
        search_web.ainvoke({"query": user_query}),
        return_exceptions=True
    )
    
    # Handle possible tool exception gracefully
    if isinstance(web_results, Exception):
        web_results = f"Web search failed: {web_results}"

    # Filter out low-relevance vector results so we don't clutter sources with irrelevant PDFs
    valid_vector = []
    if not isinstance(vector_results, Exception):
        valid_vector = [d for d in vector_results if d.get("score", 1.0) > 0.25]

    # Fuse internal results using RRF
    fused_context = reciprocal_rank_fusion(
        valid_vector, 
        sql_results if not isinstance(sql_results, Exception) else []
    )
    
    # Inject real-time web context alongside the internal database context
    if web_results and web_results != "No results found." and not isinstance(web_results, Exception):
        fused_context.append({
            "id": "web_search",
            "source": "Web Search (Tavily/DDG)",
            "content": str(web_results)
        })

    # Build context string for the LLM
    context_str = "\n\n---\n\n".join(
        [f"[Source: {doc.get('source', 'Unknown')}]\n{doc.get('content', '')}"
         for doc in fused_context]
    )
    
    # Truncate context to fit in token limits
    if len(context_str) > 10000:
        context_str = context_str[:10000] + '... [TRUNCATED]'

    # Inject long-term user memories as personalization
    memory_str = ""
    if user_memories:
        memory_str = "\n\nUser preferences from previous sessions:\n" + "\n".join(
            f"- {m}" for m in user_memories
        )

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT + memory_str),
        HumanMessage(content=f"Context:\n{context_str}\n\nQuestion: {user_query}"),
    ]

    response = await _llm.ainvoke(messages)

    return {
        "retrieved_context": fused_context,
        "final_answer": response.content,
        "sources": [{"source": d.get("source"), "id": d.get("id")} for d in fused_context],
        "current_agent": "reviewer_agent",
        "messages": [AIMessage(content=response.content)],
    }
