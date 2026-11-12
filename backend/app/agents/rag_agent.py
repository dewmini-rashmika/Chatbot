"""
RAG Sub-Agent — Hybrid Retrieval (Vector + Vectorless).

This agent handles all knowledge-retrieval tasks:
1. Vector RAG via ChromaDB — semantic similarity search for unstructured text.
2. Vectorless RAG via PostgreSQL Text-to-SQL — exact structured fact lookup.
3. Hybrid fusion via Reciprocal Rank Fusion (RRF).
4. Answer generation grounded in the fused context.

LLM choice: gemini/gemini-1.5-flash (via LiteLLM)
  - 1M-token context window to hold all fused documents.
  - Fast and cheap for synthesis tasks.
  - Swap to "anthropic/claude-3-5-sonnet-20241022" via RAG_MODEL env var
    for higher synthesis quality.
"""

import re
import asyncio
import urllib.parse
import structlog
from langchain_core.messages import AIMessage
from app.agents.state import AgentState
from app.core.llm_factory import call_llm, get_model_for_role
from app.services.sql_rag import SQLRAGService
from app.services.vector_store import VectorStoreService
from app.tools.music_tools import search_web

logger = structlog.get_logger(__name__)

RAG_SYSTEM_PROMPT = """You are an expert Music Knowledge Agent. You have access to:
- A comprehensive library of music theory textbooks and historical essays.
- A structured database of artist discographies, chart positions, and awards.
- Synthesized music reviews styled after various critical voices.
- Real-time web search results for the most current information.

Instructions:
- Answer the user's question using ONLY the context provided below.
- Be detailed, accurate, and cite specific documents where relevant.
- If the context does not contain enough information to answer confidently, say so clearly.
- Do NOT invent facts, artist names, dates, or chart positions.
- Structure longer answers with clear paragraphs; use bullet points for lists.
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
    - k=60 is the standard default from the original RRF paper.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, doc in enumerate(vector_results):
        doc_id = doc.get("id") or f"vec_{rank}"
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        doc_map[doc_id] = doc

    for rank, doc in enumerate(sql_results):
        # Build a stable ID from the row content hash
        row_key = str(sorted(doc.items()))
        doc_id = f"sql_{hash(row_key)}"
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    ranked = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    return [doc_map[d] for d in ranked]


def _clean_source_name(raw_source: str) -> str:
    """Convert a file path or URL into a readable citation label."""
    if raw_source.startswith("http"):
        try:
            parts = urllib.parse.urlparse(raw_source)
            netloc = parts.netloc.replace("www.", "")
            return f"Web Search ({netloc})"
        except Exception:
            return "Web Search"

    name = raw_source.replace("\\", "/").split("/")[-1]
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\.\w+$", "", name)  # strip extension

    if name.startswith("theory "):
        return f"Music Theory: {name[7:].title()}"
    if name.startswith("synthetic review "):
        return f"Music Review: {name[17:].title()}"
    return name.title()


async def rag_agent_node(state: AgentState) -> dict:
    """
    RAG sub-agent node: retrieves relevant context using hybrid retrieval
    and generates an answer grounded in that context.
    """
    messages  = state["messages"]
    user_query = messages[-1].content if messages else ""
    user_memories: list[str] = state.get("user_memories", [])

    model = get_model_for_role("rag")
    logger.info("rag_agent.start", query=user_query[:80], model=model)

    # ── 1. Parallel hybrid retrieval ─────────────────────────────────────────
    vector_service = VectorStoreService()
    sql_service    = SQLRAGService()

    vector_results, sql_results = await asyncio.gather(
        vector_service.search(user_query),
        sql_service.search(user_query),
    )

    # ── 2. Optional web search (if vector+SQL both return nothing) ────────────
    web_results: list[dict] = []
    if not vector_results and not sql_results:
        try:
            raw_web = await asyncio.get_event_loop().run_in_executor(
                None, lambda: search_web.invoke({"query": user_query})
            )
            # search_web returns a formatted string; wrap it as a single doc
            web_results = [{"content": raw_web, "_source": "Web Search Result"}]
        except Exception as exc:
            logger.warning("rag_agent.web_search_failed", error=str(exc))

    # ── 3. Fuse rankings via RRF ──────────────────────────────────────────────
    fused_context = reciprocal_rank_fusion(vector_results, sql_results)

    # ── 4. Build context string ───────────────────────────────────────────────
    context_parts: list[str] = []
    seen_domains: set[str] = set()
    unique_domains: list[str] = []

    for doc in fused_context[:10]:  # top-10 fused docs
        raw_source  = doc.get("source", doc.get("_source", "Internal Database"))
        source_name = _clean_source_name(str(raw_source))
        content     = doc.get("content", doc.get("document", str(doc)))

        # Truncate very long chunks to keep context manageable
        if len(content) > 1500:
            content = content[:1500] + "... [TRUNCATED]"

        context_parts.append(f"[{source_name}]\n{content}")

        # Collect unique source domains for the footer
        domain = source_name.split(":")[0].strip()
        if domain not in seen_domains:
            seen_domains.add(domain)
            unique_domains.append(domain)

    for doc in web_results:
        source_name = _clean_source_name(doc.get("_source", "Web Search Result"))
        content = doc.get("content", "")
        if len(content) > 1500:
            content = content[:1500] + "... [TRUNCATED]"
        context_parts.append(f"[{source_name}]\n{content}")

    context_str = "\n\n---\n\n".join(context_parts) if context_parts else "No results found."

    # ── 5. Inject user memories ───────────────────────────────────────────────
    memory_str = ""
    if user_memories:
        memory_str = "\n\nUser preferences from previous sessions:\n" + "\n".join(
            f"- {m}" for m in user_memories
        )

    # ── 6. Build LLM prompt and generate answer ───────────────────────────────
    domain_str = ", ".join(unique_domains) if unique_domains else "knowledge base"

    llm_messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{context_str}"
                f"{memory_str}"
                f"\n\nQuestion: {user_query}"
            ),
        },
    ]

    response = await call_llm(
        role="rag",
        messages=llm_messages,
        temperature=0.3,   # low but not zero — allows natural phrasing
    )
    answer = response.choices[0].message.content

    # Append source attribution footer
    if unique_domains:
        answer += f"\n\n*Sources: {domain_str}*"

    logger.info("rag_agent.done", sources=domain_str, model=model)

    return {
        "messages":          [AIMessage(content=answer)],
        "final_answer":      answer,
        "retrieved_context": fused_context,
        "current_agent":     "reviewer_agent",
    }
