"""Pydantic models for API requests and responses — OpenAI-compatible schema."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, field_validator


# --- Enums ---

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"


class ModelStatus(str, Enum):
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    DEPRECATED = "deprecated"


# --- Request Models ---

class ChatMessage(BaseModel):
    role: MessageRole
    content: str = Field(..., min_length=1, max_length=100_000)
    name: Optional[str] = Field(None, max_length=64)


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=100)
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=1000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    stream: bool = False
    stop: Optional[Union[str, list[str]]] = None
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = Field(None, max_length=256)
    seed: Optional[int] = None

    @field_validator("stop")
    @classmethod
    def validate_stop_sequences(cls, v: Optional[Union[str, list[str]]]) -> Optional[Union[str, list[str]]]:
        if v is None:
            return v
        if isinstance(v, str):
            return v
        if len(v) > 4:
            raise ValueError("Maximum 4 stop sequences allowed")
        return v


class EmbeddingRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=100)
    input: Union[str, list[str]] = Field(..., min_length=1)
    encoding_format: str = Field(default="float", pattern="^(float|base64)$")


class ConversationCreateRequest(BaseModel):
    model_id: str = Field(..., min_length=1, max_length=100)
    title: Optional[str] = Field(None, max_length=500)
    system_prompt: Optional[str] = Field(None, max_length=16000)


# --- Response Models ---

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[FinishReason] = FinishReason.STOP


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[MessageRole] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[FinishReason] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int = 0


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: UsageInfo


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "neuronlm"
    context_window: int = 32768
    status: ModelStatus = ModelStatus.READY


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: Optional[str] = None
    param: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: int = 0
    uptime_seconds: float = 0.0


class ConversationResponse(BaseModel):
    id: str
    user_id: str
    title: Optional[str]
    model_id: str
    system_prompt: Optional[str]
    message_count: int = 0
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    token_count: int = 0
    created_at: str
