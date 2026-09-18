"""
Music-specific LangChain tools for the Tool Sub-Agent.
These are the callable tools the LLM can invoke via function calling.

Each tool is decorated with @tool and has a clear docstring —
this docstring IS the tool description the LLM reads to decide when to use it.
"""
import json
import musicbrainzngs
from langchain_core.tools import tool
from duckduckgo_search import DDGS

from app.core.config import settings

# Configure MusicBrainz user agent (required by their API terms)
musicbrainzngs.set_useragent(
    settings.musicbrainz_app_name,
    settings.musicbrainz_version,
    settings.musicbrainz_contact,
)


@tool
def search_musicbrainz(query: str) -> str:
    """
    Search MusicBrainz for factual music metadata: artists, albums, release dates, 
    recording IDs. Use this for exact factual lookups about music releases.
    MusicBrainz is a free, open music encyclopedia — no API key required.
    
    Args:
        query: Artist name, album title, or track name to look up.
    Returns:
        JSON string with matching results.
    """
    try:
        result = musicbrainzngs.search_artists(artist=query, limit=5)
        artists = result.get("artist-list", [])
        output = []
        for a in artists[:3]:
            output.append({
                "name": a.get("name"),
                "type": a.get("type"),
                "country": a.get("country"),
                "disambiguation": a.get("disambiguation", ""),
                "id": a.get("id"),
            })
        return json.dumps(output, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


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
        result = musicbrainzngs.search_artists(artist=artist_name, limit=1)
        artists = result.get("artist-list", [])
        if not artists:
            return json.dumps({"error": f"No artist found for '{artist_name}'"})

        artist = artists[0]
        artist_id = artist.get("id")

        # Get releases for this artist
        releases = musicbrainzngs.browse_releases(artist=artist_id, limit=10)
        release_list = releases.get("release-list", [])

        return json.dumps({
            "name": artist.get("name"),
            "type": artist.get("type"),
            "country": artist.get("country"),
            "life_span": artist.get("life-span", {}),
            "recent_releases": [r.get("title") for r in release_list[:5]],
            "musicbrainz_id": artist_id,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_spotify_top_tracks(market: str = "US") -> str:
    """
    Get information about current popular music. Since Spotify's featured 
    playlists API requires OAuth, this tool searches for current trending 
    music news via web search as a free-tier alternative.
    
    Args:
        market: Market/region code (e.g., 'US', 'GB', 'IN').
    Returns:
        Current trending music information as a string.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                f"top music charts {market} 2024 site:billboard.com OR site:spotify.com",
                max_results=5,
            ))
        if not results:
            return "Could not retrieve current charts."

        output = []
        for r in results:
            output.append(f"- {r.get('title', '')}: {r.get('body', '')[:200]}")
        return "\n".join(output)
    except Exception as e:
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
        A summary of web search results.
    """
    import os
    if os.environ.get("TAVILY_API_KEY"):
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            tavily = TavilySearchResults(max_results=5)
            results = tavily.invoke({"query": query + " music"})
            
            if not results:
                return "No results found."

            output = []
            for r in results:
                output.append(
                    f"Summary: {r.get('content', '')}\n"
                    f"URL: {r.get('url', '')}"
                )
            return "\n\n---\n\n".join(output)
        except Exception as e:
            return f"Tavily Search failed: {e}"
    else:
        # 2. If Tavily fails or is missing, use DDGS
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query + " music", max_results=5))

            if not results:
                return "No results found."

            output = []
            for r in results:
                output.append(
                    f"Title: {r.get('title', '')}\n"
                    f"Summary: {r.get('body', '')[:300]}\n"
                    f"URL: {r.get('href', '')}"
                )
            return "\n\n---\n\n".join(output)
        except Exception as e:
            return f"Search failed: {e}"


@tool
def calculate_music_theory(expression: str) -> str:
    """
    Perform basic music theory calculations: interval counting, 
    time signature calculations, BPM conversions, frequency calculations.
    
    Args:
        expression: A music theory calculation query like 
                    "frequency of A4" or "beats per measure in 7/8 at 120 BPM"
    Returns:
        The calculation result as a string.
    """
    # A simple frequency calculator for notes
    NOTE_FREQUENCIES = {
        "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23,
        "G4": 392.00, "A4": 440.00, "B4": 493.88, "C5": 523.25,
    }
    expr_upper = expression.upper()
    for note, freq in NOTE_FREQUENCIES.items():
        if note in expr_upper:
            return f"The frequency of {note} is {freq} Hz (A4=440Hz equal temperament)"
    return f"Could not parse: '{expression}'. Try asking about note frequencies like 'frequency of A4'."

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
    import requests
    from bs4 import BeautifulSoup
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove scripts and styles
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
            
        text = soup.get_text(separator=' ')
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Truncate to ~15,000 characters to comfortably fit in the LLM context window
        return text[:15000]
    except Exception as e:
        return f"Failed to read URL: {str(e)}"
