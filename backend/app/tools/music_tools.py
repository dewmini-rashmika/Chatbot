"""
Music-specific LangChain tools for the Tool Sub-Agent.
These are the callable tools the LLM can invoke via function calling.

Each tool is decorated with @tool and has a clear docstring —
this docstring IS the tool description the LLM reads to decide when to use it.
"""

import os
import json
import requests
import structlog
import musicbrainzngs
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from app.core.config import settings

logger = structlog.get_logger(__name__)

# Configure MusicBrainz user-agent (required by their API TOS)
musicbrainzngs.set_useragent(
    settings.musicbrainz_app_name,
    settings.musicbrainz_version,
    settings.musicbrainz_contact,
)

# Note frequencies for music theory calculations (equal temperament, A4 = 440 Hz)
NOTE_FREQUENCIES = {
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13,
    "E4": 329.63, "F4": 349.23, "F#4": 369.99, "G4": 392.00,
    "G#4": 415.30, "A4": 440.00, "A#4": 466.16, "B4": 493.88,
    "C5": 523.25, "C3": 130.81, "A3": 220.00, "A5": 880.00,
}


@tool
def search_musicbrainz(query: str) -> str:
    """
    Search MusicBrainz for factual music metadata: artists, albums, release dates,
    recording IDs. Use this for exact factual lookups about music releases.
    MusicBrainz is a free, open music encyclopedia — no API key required.

    Args:
        query: Artist name, album title, or track name to search for.
    Returns:
        JSON string with matching artists and their release counts.
    """
    try:
        result = musicbrainzngs.search_artists(query=query, limit=5)
        artists = result.get("artist-list", [])
        output = []
        for artist in artists:
            releases = musicbrainzngs.browse_releases(
                artist=artist["id"], limit=5
            ).get("release-list", [])
            output.append({
                "name":          artist.get("name"),
                "id":            artist.get("id"),
                "disambiguation": artist.get("disambiguation", ""),
                "recent_releases": [r.get("title") for r in releases],
            })
        return json.dumps(output, indent=2) if output else "No results found."
    except Exception as e:
        logger.warning("search_musicbrainz.failed", error=str(e))
        return f"No results found."


@tool
def get_artist_info(artist_name: str) -> str:
    """
    Get comprehensive information about a music artist including biography,
    associated acts, and release count from MusicBrainz.

    Args:
        artist_name: The name of the artist to look up.
    Returns:
        JSON string with artist details.
    """
    try:
        result = musicbrainzngs.search_artists(query=artist_name, limit=1)
        artists = result.get("artist-list", [])
        if not artists:
            return f"No artist found for '{artist_name}'."

        artist = artists[0]
        artist_id = artist["id"]

        release_list = musicbrainzngs.browse_releases(
            artist=artist_id, limit=10
        ).get("release-list", [])

        output = {
            "name":          artist.get("name"),
            "id":            artist_id,
            "disambiguation": artist.get("disambiguation", ""),
            "type":          artist.get("type", ""),
            "country":       artist.get("area", {}).get("name", ""),
            "releases":      [r.get("title") for r in release_list],
            "release_count": len(release_list),
        }
        return json.dumps(output, indent=2)
    except Exception as e:
        logger.warning("get_artist_info.failed", error=str(e))
        return f"Could not retrieve artist info for '{artist_name}'."


@tool
def get_spotify_top_tracks(market: str = "US") -> str:
    """
    Get information about current popular music. Since Spotify's featured
    playlists API requires OAuth, this tool searches for current trending
    music news via web search as a free-tier alternative.

    Args:
        market: Market/region code (e.g., 'US', 'GB', 'IN').
    Returns:
        A summary of current top music charts and trends.
    """
    try:
        query = f"top music charts {market} 2024 site:billboard.com OR site:spotify.com"
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "Could not retrieve current charts."
        chunks = [
            f"{r.get('title', '')}: {r.get('body', '')}"
            for r in results
        ]
        return "\n\n".join(chunks)
    except Exception as e:
        logger.warning("get_spotify_top_tracks.failed", error=str(e))
        return f"Error fetching charts: {e}"


@tool
def search_web(query: str) -> str:
    """
    Search the web for current music news, recent events, or any music-related
    question that requires up-to-date information not in the knowledge base.
    Use for questions about recent releases, tours, awards shows, or news.

    Args:
        query: The search query string.
    Returns:
        Formatted search results with titles and snippets.
    """
    # Try Tavily first (richer results), fall back to DuckDuckGo
    if settings.tavily_api_key:
        try:
            os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
            tavily = TavilySearchResults(max_results=5)
            results = tavily.invoke({"query": query})
            if results:
                return "\n\n".join(
                    f"**{r.get('title', 'Result')}**\n{r.get('content', r.get('snippet', ''))}"
                    for r in results
                )
        except Exception as e:
            logger.warning("search_web.tavily_failed", error=str(e))

    # DuckDuckGo fallback
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        return "\n\n".join(
            f"**{r.get('title', '')}**\n{r.get('body', '')}"
            for r in results
        )
    except Exception as e:
        logger.warning("search_web.ddgs_failed", error=str(e))
        return "No results found."


@tool
def read_url(url: str) -> str:
    """
    Read and extract the main text content from a specific webpage URL.
    Use this when the user asks you to read, summarize, or analyze a specific link.

    Args:
        url: The full HTTP URL to read.
    Returns:
        The extracted text content of the webpage.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script/style noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.extract()

        lines = [line.strip() for line in soup.get_text().splitlines() if line.strip()]
        # Deduplicate consecutive repeated lines
        chunks: list[str] = []
        for line in lines:
            if not chunks or line != chunks[-1]:
                chunks.append(line)

        text = "\n".join(chunks)
        # Cap at 3000 chars to avoid blowing the context window
        if len(text) > 3000:
            text = text[:3000] + "... [truncated]"
        return text

    except Exception as e:
        logger.warning("read_url.failed", url=url, error=str(e))
        return f"Failed to read URL: {e}"
