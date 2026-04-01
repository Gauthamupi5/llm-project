"""API route handlers for chat completions, embeddings, and models."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from neuronlm.core.exceptions import (
    ContentModerationError,
    InferenceError,
    ModelNotFoundError,
    ModelNotReadyError,
)
from neuronlm.core.moderation import get_moderator
from neuronlm.models.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
    ErrorDetail,
    ModelInfo,
    ModelListResponse,
    UsageInfo,
)
from neuronlm.services.inference_engine import get_inference_engine
from neuronlm.services.usage_tracker import get_usage_tracker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["LLM API"])


def _get_user_id(request: Request) -> str:
    """Extract user_id from request state (set by auth middleware)."""
    return getattr(request.state, "user_id", "anonymous")


# --- Chat Completions ---

@router.post("/chat/completions")
async def create_chat_completion(
    body: ChatCompletionRequest,
    request: Request,
) -> Any:
    """Create a chat completion (streaming or non-streaming)."""
    user_id = _get_user_id(request)
    engine = get_inference_engine()
    moderator = get_moderator()
    tracker = get_usage_tracker()

    # Content moderation on user messages
    for msg in body.messages:
        if msg.role.value == "user":
            mod_result = moderator.check_input(msg.content)
            if mod_result.flagged:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": mod_result.message or "Content policy violation",
                            "type": "invalid_request_error",
                            "code": "content_policy_violation",
                        }
                    },
                )

    # Normalize stop sequences
    stop = body.stop
    if isinstance(stop, str):
        stop = [stop]

    start_time = time.monotonic()

    try:
        if body.stream:
            # Streaming response
            async def event_generator():
                async for chunk in engine.stream_complete(
                    model_id=body.model,
                    messages=body.messages,
                    temperature=body.temperature,
                    top_p=body.top_p,
                    max_tokens=body.max_tokens,
                    stop=stop,
                    frequency_penalty=body.frequency_penalty,
                    presence_penalty=body.presence_penalty,
                    seed=body.seed,
                ):
                    yield chunk

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Non-streaming response
            response = await engine.complete(
                model_id=body.model,
                messages=body.messages,
                temperature=body.temperature,
                top_p=body.top_p,
                max_tokens=body.max_tokens,
                stop=stop,
                frequency_penalty=body.frequency_penalty,
                presence_penalty=body.presence_penalty,
                seed=body.seed,
            )

            # Track usage
            latency_ms = int((time.monotonic() - start_time) * 1000)
            tracker.record_usage(
                user_id=user_id,
                model_id=body.model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                latency_ms=latency_ms,
            )

            return response

    except ModelNotFoundError as e:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": e.message, "type": "not_found_error", "code": e.code}},
        )
    except ModelNotReadyError as e:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": e.message, "type": "service_unavailable", "code": e.code}},
        )
    except Exception as e:
        logger.exception("Inference error")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error", "type": "server_error", "code": "internal_error"}},
        )


# --- Embeddings ---

@router.post("/embeddings")
async def create_embedding(
    body: EmbeddingRequest,
    request: Request,
) -> Any:
    """Generate embeddings for the given input."""
    user_id = _get_user_id(request)
    engine = get_inference_engine()
    tracker = get_usage_tracker()

    texts = body.input if isinstance(body.input, list) else [body.input]

    try:
        embeddings, total_tokens = await engine.generate_embedding(
            model_id=body.model,
            texts=texts,
        )

        tracker.record_usage(
            user_id=user_id,
            model_id=body.model,
            prompt_tokens=total_tokens,
            completion_tokens=0,
        )

        return EmbeddingResponse(
            data=[
                EmbeddingData(embedding=emb, index=i)
                for i, emb in enumerate(embeddings)
            ],
            model=body.model,
            usage=UsageInfo(
                prompt_tokens=total_tokens,
                completion_tokens=0,
                total_tokens=total_tokens,
            ),
        )

    except ModelNotFoundError as e:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": e.message, "type": "not_found_error", "code": e.code}},
        )


# --- Models ---

@router.get("/models")
async def list_models() -> ModelListResponse:
    """List all available models."""
    engine = get_inference_engine()
    return ModelListResponse(data=engine.list_models())


@router.get("/models/{model_id}")
async def get_model(model_id: str) -> Any:
    """Get details for a specific model."""
    engine = get_inference_engine()
    try:
        models = engine.list_models()
        for model in models:
            if model.id == model_id:
                return model
        raise ModelNotFoundError(model_id)
    except ModelNotFoundError as e:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": e.message, "type": "not_found_error", "code": e.code}},
        )


# --- Usage ---

@router.get("/usage")
async def get_usage(request: Request) -> dict[str, Any]:
    """Get usage statistics for the authenticated user."""
    user_id = _get_user_id(request)
    tracker = get_usage_tracker()
    daily = tracker.get_daily_usage(user_id)
    return {"user_id": user_id, "daily_usage": daily}
