"""
Tool/Action Sub-Agent — handles live data and external API calls.

Responsible for:
- Real-time Spotify data (top charts, now playing, playlist management)
- MusicBrainz lookups (open, free music metadata)
- Web search for current news/events
- Any write-action (modifying playlists) — these ALWAYS require HITL approval.

LLM choice: groq/llama-3.3-70b-versatile (via LiteLLM)
  - Strong function-calling support needed for tool use.
  - Swap via TOOL_MODEL env var.

Note: This agent uses langchain's AgentExecutor for the tool-calling loop,
but the underlying LLM is routed via LiteLLM through a ChatLiteLLM wrapper.
"""

import re
import urllib.parse
import structlog
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatLiteLLM
from app.agents.state import AgentState
from app.core.config import settings
from app.core.llm_factory import get_model_for_role
from app.tools.music_tools import (
    search_musicbrainz,
    get_artist_info,
    get_spotify_top_tracks,
    search_web,
    read_url,
)

logger = structlog.get_logger(__name__)

# Write-action keywords that require HITL approval before execution
WRITE_ACTIONS = {
    "create_playlist",
    "add_to_playlist",
    "remove_from_playlist",
    "follow_artist",
    "unfollow_artist",
    "save_track",
    "delete_playlist",
}

TOOL_AGENT_PROMPT_STR = """You are a Music Tool Agent with access to live music data APIs.
Your role is to fetch real-time, accurate music data to answer the user's question.

Available tools:
- search_musicbrainz: Search MusicBrainz for artists, albums, recordings.
- get_artist_info: Get detailed artist information from MusicBrainz.
- get_spotify_top_tracks: Get an artist's current top tracks from Spotify.
- search_web: Search the web for current music news and events.
- read_url: Fetch and read the content of a specific URL.

Rules:
- Always prefer structured tool results over general knowledge.
- For ANY write action (playlist creation, track saving), you MUST flag it for human approval.
- Be concise and factual. Cite your sources.
"""

_tools = [
    search_musicbrainz,
    get_artist_info,
    get_spotify_top_tracks,
    search_web,
    read_url,
]


def _build_executor() -> AgentExecutor:
    """Build a LangChain AgentExecutor backed by ChatLiteLLM."""
    model_name = get_model_for_role("tool")

    llm = ChatLiteLLM(
        model=model_name,
        temperature=0.1,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", TOOL_AGENT_PROMPT_STR),
        MessagesPlaceholder("messages"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, _tools, prompt)
    return AgentExecutor(agent=agent, tools=_tools, verbose=False, handle_parsing_errors=True)


async def tool_agent_node(state: AgentState) -> dict:
    """
    Tool sub-agent: executes external API calls and returns results.
    Flags write-actions for HITL before execution.
    """
    messages   = state["messages"]
    user_query = messages[-1].content if messages else ""

    model = get_model_for_role("tool")
    logger.info("tool_agent.start", query=user_query[:80], model=model)

    # ── Check if this is a write action requiring HITL ────────────────────────
    is_write_action = any(kw in user_query.lower() for kw in WRITE_ACTIONS)
    if is_write_action:
        action = next((kw for kw in WRITE_ACTIONS if kw in user_query.lower()), "write_action")
        logger.info("tool_agent.hitl_required", action=action)
        return {
            "hitl_required": True,
            "hitl_action":   f"Tool agent wants to execute a write action: '{action}' — please approve.",
            "current_agent": "hitl_node",
        }

    # ── Execute tool-calling agent ────────────────────────────────────────────
    try:
        executor = _build_executor()
        result   = await executor.ainvoke({"messages": messages})
        output   = result.get("output", "I couldn't retrieve that information right now.")

        # Collect source URLs from intermediate steps
        intermediate = result.get("intermediate_steps", [])
        urls: list[str] = []
        seen_domains: set[str] = set()
        unique_domains: list[str] = []

        for _, observation in intermediate:
            found_urls = re.findall(r"URL: (https?://[^\s]+)", str(observation))
            urls.extend(found_urls)

        for u in urls:
            try:
                parts  = urllib.parse.urlparse(u)
                domain = parts.netloc.replace("www.", "")
                if domain not in seen_domains:
                    seen_domains.add(domain)
                    unique_domains.append(domain)
            except Exception:
                pass

        # Append source footer
        if unique_domains:
            domain_str = ", ".join(unique_domains)
            output += f"\n\n*Sources: {domain_str}*"

        tool_results = [
            {"step": i, "observation": str(obs)[:500]}
            for i, (_, obs) in enumerate(intermediate)
        ]

    except Exception as exc:
        logger.warning("tool_agent.execution_failed", error=str(exc))
        output       = "I couldn't retrieve that information right now."
        tool_results = []

    logger.info("tool_agent.done", model=model)

    return {
        "messages":      [AIMessage(content=output)],
        "final_answer":  output,
        "tool_results":  tool_results,
        "current_agent": "reviewer_agent",
    }
