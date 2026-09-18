"""
Application configuration loaded from environment variables.
Using pydantic-settings for type-safe, validated config.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────────
    app_name: str = "Music Knowledge & Discovery Agent"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    secret_key: str = "change-me-in-production"

    # ── JWT ────────────────────────────────────────────────────────────────────
    jwt_secret_key: str = "change-jwt-secret"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # ── PostgreSQL ─────────────────────────────────────────────────────────────
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

    # ── LLM ───────────────────────────────────────────────────────────────────
    google_api_key: str | None = None
    groq_api_key: str | None = None
    gemini_model: str = "gemini-3.6-flash"
    gemini_pro_model: str = "gemini-3.6-flash"

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


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton — avoids re-parsing .env on every call."""
    return Settings()


settings = get_settings()
