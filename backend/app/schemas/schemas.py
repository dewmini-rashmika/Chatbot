"""
Pydantic schemas for request validation and response serialization.
Separating schemas from ORM models is a production best practice —
it gives you full control over what data enters and leaves the API.
"""
import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator


# ── Auth Schemas ───────────────────────────────────────────────────────────────

class UserRegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=100)

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username must be alphanumeric (underscores/hyphens allowed)")
        return v.lower()


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshTokenRequest(BaseModel):
    refresh_token: str


# ── User Schemas ───────────────────────────────────────────────────────────────

class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    username: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Conversation Schemas ───────────────────────────────────────────────────────

class ConversationCreateRequest(BaseModel):
    title: str = Field(default="New Conversation", max_length=500)


class ConversationResponse(BaseModel):
    id: uuid.UUID
    thread_id: str
    title: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationListResponse(BaseModel):
    conversations: list[ConversationResponse]
    total: int


# ── Chat / Message Schemas ─────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: uuid.UUID


class SourceItem(BaseModel):
    source: str
    id: str = ""


class MessageResponse(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    sources: list[SourceItem] = []
    created_at: datetime

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm_message(cls, msg) -> "MessageResponse":
        """Build from ORM Message, pulling sources out of extra_data."""
        sources: list[SourceItem] = []
        if msg.extra_data and isinstance(msg.extra_data.get("sources"), list):
            for s in msg.extra_data["sources"]:
                if isinstance(s, dict):
                    sources.append(SourceItem(source=s.get("source", ""), id=s.get("id", "")))
                elif isinstance(s, str):
                    sources.append(SourceItem(source=s))
        return cls(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            sources=sources,
            created_at=msg.created_at,
        )


class ConversationMessagesResponse(BaseModel):
    conversation_id: uuid.UUID
    messages: list[MessageResponse]


# ── Streaming Schemas ──────────────────────────────────────────────────────────

class StreamEvent(BaseModel):
    """Schema for Server-Sent Events streamed to the frontend."""
    event: str  # token | tool_start | tool_end | hitl_required | done | error
    data: str | dict
    conversation_id: str | None = None


# ── HITL Schemas ───────────────────────────────────────────────────────────────

class HITLDecisionRequest(BaseModel):
    thread_id: str
    approved: bool
    reason: str | None = None
