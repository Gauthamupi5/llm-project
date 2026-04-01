"""NeuronLM configuration module using pydantic-settings."""

from __future__ import annotations

import secrets
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "NeuronLM"
    app_version: str = "1.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Database
    database_url: str = "postgresql+asyncpg://neuronlm:neuronlm@localhost:5432/neuronlm"
    db_pool_size: int = 20
    db_max_overflow: int = 10
    db_pool_timeout: int = 30

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_seconds: int = 3600

    # Auth
    jwt_secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60
    api_key_prefix: str = "nlm-"

    # Rate Limiting
    rate_limit_requests_per_minute: int = 60
    rate_limit_tokens_per_minute: int = 500_000
    rate_limit_tokens_per_day: int = 10_000_000

    # Model Defaults
    default_model: str = "neuronlm-7b"
    default_temperature: float = 0.7
    default_max_tokens: int = 2048
    max_context_length: int = 32768
    max_stop_sequences: int = 4

    # Inference
    inference_timeout_seconds: int = 120
    stream_idle_timeout_seconds: int = 30
    max_batch_size: int = 64
    inference_backend: str = "simulated"  # simulated | bigram | transformers
    bigram_checkpoint_path: Optional[str] = None
    transformers_model_name: str = "distilgpt2"

    # Content Moderation
    enable_content_moderation: bool = True
    max_input_length: int = 100_000

    # Observability
    log_level: str = "INFO"
    enable_tracing: bool = False
    prometheus_port: int = 9090

    # CORS
    cors_origins: list[str] = ["*"]

    model_config = {"env_prefix": "NEURONLM_", "env_file": ".env", "case_sensitive": False}

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v.startswith(("postgresql", "sqlite")):
            raise ValueError("Only PostgreSQL and SQLite database URLs are supported")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
