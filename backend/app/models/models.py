"""
SQLAlchemy ORM models for Users, Conversations, Messages, and Long-term Memory.
All models use UUID primary keys for distributed-system compatibility.
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class User(Base):
    """
    Core user account model.
    Password is stored as a bcrypt hash — never plaintext.
    """
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    conversations: Mapped[list["Conversation"]] = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )
    long_term_memories: Mapped[list["LongTermMemory"]] = relationship(
        "LongTermMemory", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username}>"


class Conversation(Base):
    """
    Represents a single chat thread (maps to a LangGraph thread_id).
    Each conversation belongs to one user.
    The thread_id links to LangGraph's Postgres checkpointer.
    """
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    thread_id: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )  # LangGraph thread identifier
    title: Mapped[str] = mapped_column(String(500), nullable=False, default="New Conversation")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Conversation id={self.id} thread_id={self.thread_id}>"


class Message(Base):
    """
    Individual messages within a conversation.
    Stores the full message payload + metadata (tool calls, agent thoughts).
    """
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(50), nullable=False)  # user | assistant | tool
    content: Mapped[str] = mapped_column(Text, nullable=False)
    extra_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # tool_calls, sources, etc.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")


class LongTermMemory(Base):
    """
    Stores extracted user preferences/facts across sessions.
    e.g., "User loves jazz from the 1950s", "User's favourite artist is Miles Davis"
    These are retrieved at the start of new conversations to personalize responses.
    """
    __tablename__ = "long_term_memories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    fact: Mapped[str] = mapped_column(Text, nullable=False)  # The extracted fact
    category: Mapped[str] = mapped_column(String(100), nullable=True)  # genre, artist, mood, etc.
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    source_conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="long_term_memories")


# ── Music Knowledge Structured DB (Vectorless RAG) ─────────────────────────────

class Artist(Base):
    """Structured artist facts for exact SQL lookups."""
    __tablename__ = "artists"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    birth_year: Mapped[int | None] = mapped_column(nullable=True)
    death_year: Mapped[int | None] = mapped_column(nullable=True)
    nationality: Mapped[str | None] = mapped_column(String(100), nullable=True)
    genres: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    musicbrainz_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    extra_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    albums: Mapped[list["Album"]] = relationship("Album", back_populates="artist")


class Album(Base):
    """Structured album facts — release dates, labels, chart positions."""
    __tablename__ = "albums"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    artist_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("artists.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    release_year: Mapped[int | None] = mapped_column(nullable=True)
    release_date: Mapped[str | None] = mapped_column(String(50), nullable=True)
    label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    genre: Mapped[str | None] = mapped_column(String(255), nullable=True)
    chart_positions: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    awards: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    musicbrainz_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    extra_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    artist: Mapped["Artist"] = relationship("Artist", back_populates="albums")
    tracks: Mapped[list["Track"]] = relationship("Track", back_populates="album")


class Track(Base):
    """Individual track data."""
    __tablename__ = "tracks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    album_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("albums.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    track_number: Mapped[int | None] = mapped_column(nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(nullable=True)
    musicbrainz_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    album: Mapped["Album"] = relationship("Album", back_populates="tracks")
