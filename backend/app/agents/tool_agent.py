"""
Tool/Action Sub-Agent — handles live data and external API calls.

Responsible for:
- Real-time Spotify data (top charts, now playing, playlist management)
- MusicBrainz lookups (open, free music metadata)
- Web search for current news/events
- Any write-action (modifying playlists) — these ALWAYS go through HITL
"""
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.agents.state import AgentState
from app.core.config import settings
from app.tools.music_tools import (
    search_musicbrainz,
    get_spotify_top_tracks,
    search_web,
    get_artist_info,
    read_url,
)

# Write actions that require human approval before execution
WRITE_ACTIONS = {"add_to_playlist", "create_playlist", "follow_artist"}

_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    max_tokens=900,
    groq_api_key=settings.groq_api_key,
    temperature=0,
)

TOOL_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Music Action Agent. You have access to real-time music tools.
    
Available tools:
- search_musicbrainz: Look up factual music metadata (IDs, release dates, etc.)
- get_spotify_top_tracks: Get current top charts
- search_web: Search for current music news
- get_artist_info: Get comprehensive artist information
- read_url: Extract text from a specific webpage URL provided by the user

RULES:
- If the user provides a link/URL, YOU MUST use the read_url tool to read it. Do not say you cannot read links.
- For any write action (modifying playlists), always flag hitl_required=True
- Never fabricate API responses — use the actual tool results
- If a tool fails, report the failure clearly"""),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

_tools = [search_musicbrainz, get_spotify_top_tracks, search_web, get_artist_info, read_url]


async def tool_agent_node(state: AgentState) -> dict:
    """
    Tool sub-agent: executes external API calls and returns results.
    Flags write-actions for HITL before execution.
    """
    user_query = state["messages"][-6:][-1].content

    # Check if this is a write action requiring HITL
    write_keywords = ["add to playlist", "create playlist", "follow", "save", "modify"]
    is_write_action = any(kw in user_query.lower() for kw in write_keywords)

    if is_write_action and not state.get("hitl_approved"):
        return {
            "hitl_required": True,
            "hitl_action": f"Tool agent wants to execute a write action: '{user_query}'",
            "hitl_approved": None,
            "current_agent": "hitl_node",
        }

    # Build and run the tool-calling agent
    agent = create_tool_calling_agent(_llm, _tools, TOOL_AGENT_PROMPT)
    executor = AgentExecutor(
        agent=agent, 
        tools=_tools, 
        verbose=False, 
        max_iterations=5, 
        return_intermediate_steps=True
    )

    result = await executor.ainvoke({"messages": state["messages"][-6:]})
    output = result.get("output", "I couldn't retrieve that information right now.")

    intermediate_steps = result.get("intermediate_steps", [])
    
    # Extract tools used as sources
    sources = []
    for action, observation in intermediate_steps:
        if action.tool == "search_web":
            import re
            from urllib.parse import urlparse
            urls = re.findall(r'URL: (https?://[^\s]+)', str(observation))
            domains = []
            for u in urls:
                try:
                    netloc = urlparse(u).netloc.replace('www.', '')
                    parts = netloc.split('.')
                    name = parts[-2] if len(parts) > 1 else parts[0]
                    domains.append(name.capitalize())
                except Exception:
                    pass
            seen = set()
            unique_domains = [x for x in domains if not (x in seen or seen.add(x))]
            domain_str = ", ".join(unique_domains)
            source_name = f"Web Search ({domain_str})" if domain_str else "Web Search"
            sources.append({"source": source_name, "id": action.tool})
        elif action.tool == "read_url":
            import re
            from urllib.parse import urlparse
            url = action.tool_input.get('url', str(action.tool_input))
            try:
                netloc = urlparse(url).netloc.replace('www.', '')
                parts = netloc.split('.')
                name = parts[-2] if len(parts) > 1 else parts[0]
                domain_str = name.capitalize()
            except Exception:
                domain_str = "Url"
            sources.append({"source": f"Web Search ({domain_str})", "id": action.tool})
        
    return {
        "tool_results": intermediate_steps,
        "final_answer": output,
        "sources": sources,
        "current_agent": "reviewer_agent",
        "messages": [AIMessage(content=output)],
    }
