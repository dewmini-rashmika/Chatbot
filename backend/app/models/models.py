"""
SQLAlchemy ORM models for Users, Conversations, Messages, and Long-term Memory.
All models use UUID primary keys for distributed-system compatibility.
"""

import uuid
from datetime import datetime
from sqlalchemy import String, Boolean, Text, Float, DateTime, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import func
from app.db.database import Base


class User(Base):
    """
    Core user account model.
    Password is stored as a bcrypt hash — never plaintext.
    """
    __tablename__ = "users"

    id:              Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username:        Mapped[str]       = mapped_column(String(50), unique=True, nullable=False, index=True)
    email:           Mapped[str]       = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str]       = mapped_column(String(255), nullable=False)
    is_active:       Mapped[bool]      = mapped_column(Boolean, default=True)
    is_verified:     Mapped[bool]      = mapped_column(Boolean, default=False)
    created_at:      Mapped[datetime]  = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at:      Mapped[datetime]  = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    conversations:   Mapped[list["Conversation"]]  = relationship("Conversation", back_populates="user")
    long_term_memories: Mapped[list["LongTermMemory"]] = relationship("LongTermMemory", back_populates="user")


class Conversation(Base):
    """
    Represents a single chat thread (maps to a LangGraph thread_id).
    Each conversation belongs to one user.
    The thread_id links to LangGraph's Postgres checkpointer.
    """
    __tablename__ = "conversations"

    id:         Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id:    Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    thread_id:  Mapped[str]       = mapped_column(String(255), unique=True, nullable=False, index=True)
    title:      Mapped[str]       = mapped_column(String(255), nullable=True)
    is_active:  Mapped[bool]      = mapped_column(Boolean, default=True)
    messages:   Mapped[list]      = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime]  = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime]  = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user:     Mapped["User"]           = relationship("User", back_populates="conversations")
    msgs:     Mapped[list["Message"]]  = relationship("Message", back_populates="conversation", order_by="Message.created_at")

    def __repr__(self) -> str:
        return f"<Conversation id={self.id} thread_id={self.thread_id}>"


class Message(Base):
    """
    Individual messages within a conversation.
    Stores the full message payload + metadata (tool calls, agent thoughts).
    """
    __tablename__ = "messages"

    id:              Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True)
    role:            Mapped[str]       = mapped_column(String(20), nullable=False)   # "user" | "assistant" | "tool"
    content:         Mapped[str]       = mapped_column(Text, nullable=False)
    extra_data:      Mapped[dict]      = mapped_column(JSONB, default=dict)          # sources, confidence, agent, etc.
    created_at:      Mapped[datetime]  = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="msgs")


class LongTermMemory(Base):
    """
    Stores extracted user preferences/facts across sessions.
    e.g., "User loves jazz from the 1950s", "User's favourite artist is Miles Davis"
    These are retrieved at the start of new conversations to personalize responses.
    """
    __tablename__ = "long_term_memories"

    id:                     Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id:                Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    fact:                   Mapped[str]       = mapped_column(Text, nullable=False)
    source_conversation_id: Mapped[str]       = mapped_column(String(255), nullable=True)
    created_at:             Mapped[datetime]  = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship("User", back_populates="long_term_memories")


# ── Music knowledge tables (read-only, populated via ingest scripts) ───────────

class Artist(Base):
    """Structured artist facts for exact SQL lookups."""
    __tablename__ = "artists"

    id:          Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name:        Mapped[str]       = mapped_column(String(255), nullable=False, index=True)
    birth_year:  Mapped[int]       = mapped_column(Integer, nullable=True)
    death_year:  Mapped[int]       = mapped_column(Integer, nullable=True)
    nationality: Mapped[str]       = mapped_column(String(100), nullable=True)
    genres:      Mapped[list]      = mapped_column(JSONB, default=list)
    awards:      Mapped[dict]      = mapped_column(JSONB, default=dict)
    musicbrainz_id: Mapped[str]   = mapped_column(String(255), nullable=True)

    albums: Mapped[list["Album"]] = relationship("Album", back_populates="artist")


class Album(Base):
    """Structured album facts — release dates, labels, chart positions."""
    __tablename__ = "albums"

    id:           Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    artist_id:    Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("artists.id"), nullable=False)
    title:        Mapped[str]       = mapped_column(String(255), nullable=False)
    release_year: Mapped[int]       = mapped_column(Integer, nullable=True)
    release_date: Mapped[datetime]  = mapped_column(DateTime, nullable=True)
    genre:        Mapped[str]       = mapped_column(String(100), nullable=True)
    label:        Mapped[str]       = mapped_column(String(255), nullable=True)

    artist: Mapped["Artist"]      = relationship("Artist", back_populates="albums")
    tracks: Mapped[list["Track"]] = relationship("Track", back_populates="album")


class Track(Base):
    """Individual track data."""
    __tablename__ = "tracks"

    id:           Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    album_id:     Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("albums.id"), nullable=False)
    artist_id:    Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("artists.id"), nullable=False)
    title:        Mapped[str]       = mapped_column(String(255), nullable=False)
    duration_ms:  Mapped[int]       = mapped_column(Integer, nullable=True)
    track_number: Mapped[int]       = mapped_column(Integer, nullable=True)
    category:     Mapped[str]       = mapped_column(String(100), nullable=True)

    album:  Mapped["Album"]  = relationship("Album",  back_populates="tracks")
