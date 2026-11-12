"""
Pydantic schemas for request validation and response serialization.
Separating schemas from ORM models is a production best practice —
it gives you full control over what data enters and leaves the API.
"""

import uuid
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, field_validator, ConfigDict


# ── Auth schemas ───────────────────────────────────────────────────────────────

class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email:    EmailStr
    password: str = Field(..., min_length=8)

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        clean = v.replace("_", "").replace("-", "")
        if not clean.isalnum():
            raise ValueError("Username must be alphanumeric (underscores/hyphens allowed)")
        return v.lower()


class UserLoginRequest(BaseModel):
    email:    EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:         uuid.UUID
    username:   str
    email:      str
    is_active:  bool
    created_at: datetime


# ── Conversation schemas ───────────────────────────────────────────────────────

class ConversationCreateRequest(BaseModel):
    title: str | None = None


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:         uuid.UUID
    thread_id:  str
    title:      str | None
    is_active:  bool
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    conversations: list[ConversationResponse]
    total:         int


class SourceItem(BaseModel):
    name:  str
    score: float = 0.0


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:              uuid.UUID
    role:            str
    content:         str
    sources:         list[SourceItem] = []
    confidence:      float | None     = None
    created_at:      datetime

    @classmethod
    def from_orm_message(cls, message) -> "MessageResponse":
        """Build from ORM Message, pulling sources out of extra_data."""
        extra = message.extra_data or {}
        sources = [
            SourceItem(name=s.get("name", ""), score=s.get("score", 0.0))
            for s in extra.get("sources", [])
        ]
        return cls(
            id=message.id,
            role=message.role,
            content=message.content,
            sources=sources,
            confidence=extra.get("confidence"),
            created_at=message.created_at,
        )


class ConversationMessagesResponse(BaseModel):
    conversation_id: uuid.UUID
    messages:        list[MessageResponse]


# ── Chat schemas ───────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    conversation_id: uuid.UUID
    message:         str = Field(..., min_length=1, max_length=4000)


class HITLDecisionRequest(BaseModel):
    conversation_id: uuid.UUID
    approved:        bool
    reason:          str | None = None


# ── SSE schema ─────────────────────────────────────────────────────────────────

class StreamEvent(BaseModel):
    """Schema for Server-Sent Events streamed to the frontend."""
    event: str    # "token" | "tool_start" | "agent_start" | "done" | "error"
    data:  dict
