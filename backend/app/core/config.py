"""
Application configuration loaded from environment variables.
Using pydantic-settings for type-safe, validated config.
"""
import functools
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_name: str = "Music Knowledge & Discovery Agent"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    secret_key: str = "change-me-in-production"

    # ── JWT ───────────────────────────────────────────────────────────────────
    jwt_secret_key: str = "change-jwt-secret"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "music_agent"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/music_agent"
    sync_database_url: str = "postgresql://postgres:postgres@localhost:5432/music_agent"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_persist_dir: str = "./data/chroma_store"

    # ── LLM API Keys (used by LiteLLM gateway) ───────────────────────────────
    groq_api_key: str = ""
    google_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    deepseek_api_key: str = ""

    # ── Per-Role Model Selection (swap any model without touching agent code) ─
    # 1. Supervisor  — pure routing classification, ultra-low latency
    supervisor_model: str = "groq/llama-3.3-70b-versatile"

    # 2. Text-to-SQL — high code accuracy, strict SQL schema adherence
    sql_model: str = "groq/llama-3.3-70b-versatile"

    # 3. RAG Generator — large context window, natural writing style
    rag_model: str = "gemini/gemini-1.5-flash"

    # 4. Reviewer / Judge — deep logical reasoning, skeptical quality gate
    reviewer_model: str = "groq/llama-3.3-70b-versatile"

    # 5. Memory Extractor — simple JSON extraction, background async task
    memory_model: str = "groq/llama-3.1-8b-instant"

    # 6. Chat Agent — casual conversation, general music chit-chat
    chat_model: str = "groq/llama-3.3-70b-versatile"

    # 7. Creative Agent — songwriting, lyrics, music poetry
    creative_model: str = "groq/llama-3.3-70b-versatile"

    # 8. Tool Agent — live API calls, function calling
    tool_model: str = "groq/llama-3.3-70b-versatile"

    # ── Legacy Gemini fields (kept for backward compat) ───────────────────────
    gemini_model: str = "gemini-1.5-flash"
    gemini_pro_model: str = "gemini-1.5-flash"

    # ── Spotify ───────────────────────────────────────────────────────────────
    spotify_client_id: str = ""
    spotify_client_secret: str = ""
    spotify_redirect_uri: str = "http://localhost:8000/auth/spotify/callback"

    # ── MusicBrainz ───────────────────────────────────────────────────────────
    musicbrainz_app_name: str = "MusicKnowledgeAgent"
    musicbrainz_version: str = "1.0"
    musicbrainz_contact: str = ""

    # ── Guardrails ────────────────────────────────────────────────────────────
    max_lyric_chars: int = 50
    confidence_threshold: float = 0.6

    # ── Tavily ────────────────────────────────────────────────────────────────
    tavily_api_key: str = ""

    # ── LangSmith Observability ───────────────────────────────────────────────
    langchain_tracing_v2: str = "false"
    langchain_api_key: str = ""
    langchain_project: str = "music-agent"
    langchain_endpoint: str = "https://api.smith.langchain.com"


@functools.lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — avoids re-parsing .env on every call."""
    return Settings()


settings = get_settings()
