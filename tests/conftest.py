"""Shared test fixtures for all NeuronLM tests."""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch

# Set test environment before importing app modules
os.environ["NEURONLM_ENVIRONMENT"] = "development"
os.environ["NEURONLM_DEBUG"] = "true"
os.environ["NEURONLM_SECRET_KEY"] = "test-secret-key-for-testing-only"
os.environ["NEURONLM_JWT_SECRET_KEY"] = "test-jwt-secret-key-for-testing-only"

from httpx import ASGITransport, AsyncClient

from neuronlm.config import Settings, get_settings
from neuronlm.core.auth import AuthService, get_auth_service
from neuronlm.core.rate_limiter import RateLimiter, get_rate_limiter
from neuronlm.core.moderation import ContentModerator
from neuronlm.core.tokenizer import TokenizerService
from neuronlm.main import create_app
from neuronlm.services.inference_engine import InferenceEngine
from neuronlm.services.conversation import ConversationStore
from neuronlm.services.usage_tracker import UsageTracker


@pytest.fixture
def settings() -> Settings:
    """Test settings."""
    return Settings(
        environment="development",
        debug=True,
        secret_key="test-secret",
        jwt_secret_key="test-jwt-secret",
        rate_limit_requests_per_minute=60,
    )


@pytest.fixture
def tokenizer() -> TokenizerService:
    """Fresh tokenizer instance."""
    return TokenizerService()


@pytest.fixture
def auth_service() -> AuthService:
    """Fresh auth service instance."""
    return AuthService()


@pytest.fixture
def rate_limiter() -> RateLimiter:
    """Fresh rate limiter instance."""
    return RateLimiter(default_rpm=60)


@pytest.fixture
def moderator() -> ContentModerator:
    """Content moderator with moderation enabled."""
    return ContentModerator(enabled=True)


@pytest.fixture
def moderator_disabled() -> ContentModerator:
    """Content moderator with moderation disabled."""
    return ContentModerator(enabled=False)


@pytest.fixture
def inference_engine() -> InferenceEngine:
    """Fresh inference engine instance."""
    return InferenceEngine()


@pytest.fixture
def conversation_store() -> ConversationStore:
    """Fresh conversation store instance."""
    return ConversationStore()


@pytest.fixture
def usage_tracker() -> UsageTracker:
    """Fresh usage tracker instance."""
    return UsageTracker()


@pytest.fixture
def api_key(auth_service: AuthService) -> str:
    """Generate a valid API key for testing."""
    raw_key, _ = auth_service.generate_api_key(user_id="test-user-001", name="test-key")
    return raw_key


@pytest.fixture
def auth_headers(api_key: str) -> dict[str, str]:
    """Authorization headers with a valid API key."""
    return {"Authorization": f"Bearer {api_key}"}


@pytest.fixture
def app():
    """Fresh FastAPI application instance."""
    return create_app()


@pytest.fixture
async def client(app, auth_service: AuthService) -> AsyncClient:
    """Async HTTP client for API testing.

    Patches the auth service singleton so test API keys work.
    """
    import neuronlm.core.auth as auth_module
    import neuronlm.core.rate_limiter as rl_module

    original_auth = auth_module._auth_service
    original_rl = rl_module._rate_limiter

    auth_module._auth_service = auth_service
    rl_module._rate_limiter = RateLimiter(default_rpm=1000)  # High limit for tests

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    auth_module._auth_service = original_auth
    rl_module._rate_limiter = original_rl
