"""Conversation management API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from neuronlm.core.exceptions import ConversationNotFoundError
from neuronlm.models.schemas import ConversationCreateRequest, MessageRole
from neuronlm.services.conversation import get_conversation_store

router = APIRouter(prefix="/v1/conversations", tags=["Conversations"])


def _get_user_id(request: Request) -> str:
    return getattr(request.state, "user_id", "anonymous")


@router.post("")
async def create_conversation(body: ConversationCreateRequest, request: Request) -> Any:
    """Create a new conversation."""
    user_id = _get_user_id(request)
    store = get_conversation_store()
    conv = store.create_conversation(
        user_id=user_id,
        model_id=body.model_id,
        title=body.title,
        system_prompt=body.system_prompt,
    )
    return JSONResponse(status_code=201, content=conv.model_dump())


@router.get("")
async def list_conversations(
    request: Request,
    limit: int = 20,
    offset: int = 0,
) -> Any:
    """List conversations for the authenticated user."""
    user_id = _get_user_id(request)
    store = get_conversation_store()
    conversations = store.list_conversations(user_id, limit=limit, offset=offset)
    return {"conversations": [c.model_dump() for c in conversations], "total": len(conversations)}


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str, request: Request) -> Any:
    """Get a conversation by ID."""
    user_id = _get_user_id(request)
    store = get_conversation_store()
    try:
        conv = store.get_conversation(conversation_id, user_id)
        return conv.model_dump()
    except ConversationNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "Conversation not found", "type": "not_found_error"}},
        )


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str, request: Request) -> Any:
    """Delete a conversation."""
    user_id = _get_user_id(request)
    store = get_conversation_store()
    try:
        store.delete_conversation(conversation_id, user_id)
        return JSONResponse(status_code=204, content=None)
    except ConversationNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "Conversation not found", "type": "not_found_error"}},
        )


@router.get("/{conversation_id}/messages")
async def get_messages(conversation_id: str, request: Request, limit: int = 100) -> Any:
    """Get messages for a conversation."""
    user_id = _get_user_id(request)
    store = get_conversation_store()
    try:
        # Verify ownership
        store.get_conversation(conversation_id, user_id)
        messages = store.get_messages(conversation_id, limit=limit)
        return {"messages": [m.model_dump() for m in messages]}
    except ConversationNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "Conversation not found", "type": "not_found_error"}},
        )
