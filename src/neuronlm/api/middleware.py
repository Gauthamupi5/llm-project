"""FastAPI middleware for authentication, rate limiting, logging, and CORS."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from neuronlm.core.auth import get_auth_service
from neuronlm.core.exceptions import AuthenticationError, RateLimitError
from neuronlm.core.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

# Endpoints that don't require authentication
PUBLIC_PATHS = {"/health", "/ready", "/docs", "/openapi.json", "/redoc"}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request with timing and correlation ID."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.monotonic()

        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
            },
        )

        response = await call_next(request)
        duration_ms = (time.monotonic() - start_time) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"

        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )

        return response


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Validates authentication for protected endpoints."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip auth for public paths
        if request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")

        try:
            auth_service = get_auth_service()
            auth_context = auth_service.authenticate(auth_header)
            request.state.user_id = auth_context["user_id"]
            request.state.auth_type = auth_context["auth_type"]
            request.state.scopes = auth_context["scopes"]
        except AuthenticationError as e:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": e.message,
                        "type": "authentication_error",
                        "code": e.code,
                    }
                },
            )

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforces per-key rate limits."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip for public paths or non-API calls
        if request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        # Rate limit by user_id if authenticated
        user_id = getattr(request.state, "user_id", None)
        if user_id is None:
            return await call_next(request)

        rate_limiter = get_rate_limiter()
        if not rate_limiter.check(user_id):
            remaining = rate_limiter.remaining(user_id)
            retry_after = rate_limiter.retry_after(user_id)

            from fastapi.responses import JSONResponse
            response = JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "Rate limit exceeded. Please retry after some time.",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                },
            )
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["Retry-After"] = str(int(retry_after) + 1)
            return response

        response = await call_next(request)

        # Add rate limit headers
        remaining = rate_limiter.remaining(user_id)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response
