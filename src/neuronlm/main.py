"""FastAPI application factory — assembles the NeuronLM application."""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from neuronlm.api.middleware import (
    AuthenticationMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
)
from neuronlm.api.routes import router as llm_router
from neuronlm.api.conversations import router as conv_router
from neuronlm.config import get_settings
from neuronlm.models.schemas import HealthResponse
from neuronlm.services.inference_engine import get_inference_engine

logger = logging.getLogger(__name__)

_start_time = time.time()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Enterprise LLM Platform — OpenAI-compatible API",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # --- Middleware (applied in reverse order) ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthenticationMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    # --- Routers ---
    app.include_router(llm_router)
    app.include_router(conv_router)

    # --- Health Endpoints ---
    @app.get("/health", tags=["Health"])
    async def health_check() -> HealthResponse:
        engine = get_inference_engine()
        models = engine.list_models()
        return HealthResponse(
            status="healthy",
            version=settings.app_version,
            models_loaded=len(models),
            uptime_seconds=round(time.time() - _start_time, 2),
        )

    @app.get("/ready", tags=["Health"])
    async def readiness_check() -> dict[str, str]:
        engine = get_inference_engine()
        models = engine.list_models()
        if not models:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ready"}

    # --- Global Exception Handler ---
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "type": "server_error",
                    "code": "internal_error",
                }
            },
        )

    return app


# Application instance for uvicorn
app = create_app()
