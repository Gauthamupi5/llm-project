"""Authentication and authorization service."""

from __future__ import annotations

import hashlib
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from jose import JWTError, jwt

from neuronlm.config import get_settings
from neuronlm.core.exceptions import AuthenticationError


class AuthService:
    """Handles API key and JWT authentication."""

    API_KEY_PREFIX = "nlm-"
    API_KEY_LENGTH = 32

    def __init__(self) -> None:
        self._settings = get_settings()
        # In-memory store for demo; production uses database
        self._api_keys: dict[str, dict[str, Any]] = {}

    # --- API Key Management ---

    def generate_api_key(self, user_id: str, name: str = "default") -> tuple[str, str]:
        """Generate a new API key. Returns (raw_key, key_hash)."""
        raw_key = self.API_KEY_PREFIX + secrets.token_urlsafe(self.API_KEY_LENGTH)
        key_hash = self.hash_api_key(raw_key)
        key_prefix = raw_key[: len(self.API_KEY_PREFIX) + 8]

        self._api_keys[key_hash] = {
            "user_id": user_id,
            "name": name,
            "key_prefix": key_prefix,
            "key_hash": key_hash,
            "scopes": ["chat", "embeddings", "models"],
            "rate_limit_rpm": self._settings.rate_limit_requests_per_minute,
            "is_active": True,
            "expires_at": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        return raw_key, key_hash

    @staticmethod
    def hash_api_key(raw_key: str) -> str:
        """Hash an API key with SHA-256."""
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def validate_api_key(self, raw_key: str) -> dict[str, Any]:
        """Validate an API key and return the associated metadata."""
        if not raw_key.startswith(self.API_KEY_PREFIX):
            raise AuthenticationError("Invalid API key format")

        if len(raw_key) < len(self.API_KEY_PREFIX) + 10:
            raise AuthenticationError("Invalid API key format")

        key_hash = self.hash_api_key(raw_key)
        key_data = self._api_keys.get(key_hash)

        if key_data is None:
            raise AuthenticationError("Invalid API key")

        if not key_data["is_active"]:
            raise AuthenticationError("API key has been deactivated")

        if key_data["expires_at"]:
            expires = datetime.fromisoformat(key_data["expires_at"])
            if expires < datetime.now(timezone.utc):
                raise AuthenticationError("API key has expired")

        return key_data

    def deactivate_api_key(self, key_hash: str) -> bool:
        """Deactivate an API key."""
        if key_hash in self._api_keys:
            self._api_keys[key_hash]["is_active"] = False
            return True
        return False

    # --- JWT Management ---

    def create_jwt_token(self, user_id: str, extra_claims: Optional[dict[str, Any]] = None) -> str:
        """Create a signed JWT token."""
        now = datetime.now(timezone.utc)
        payload: dict[str, Any] = {
            "sub": user_id,
            "iat": now,
            "exp": now + timedelta(minutes=self._settings.jwt_expiry_minutes),
            "iss": "neuronlm",
        }
        if extra_claims:
            payload.update(extra_claims)

        return jwt.encode(
            payload,
            self._settings.jwt_secret_key,
            algorithm=self._settings.jwt_algorithm,
        )

    def validate_jwt_token(self, token: str) -> dict[str, Any]:
        """Validate a JWT token and return its claims."""
        try:
            payload = jwt.decode(
                token,
                self._settings.jwt_secret_key,
                algorithms=[self._settings.jwt_algorithm],
            )
        except JWTError as e:
            raise AuthenticationError(f"Invalid JWT token: {e}")

        if "sub" not in payload:
            raise AuthenticationError("JWT token missing 'sub' claim")

        return payload

    # --- Unified Authentication ---

    def authenticate(self, authorization: str) -> dict[str, Any]:
        """Authenticate a request using the Authorization header value.

        Supports:
        - Bearer <api-key> (starts with nlm-)
        - Bearer <jwt-token>
        """
        if not authorization:
            raise AuthenticationError("Missing Authorization header")

        parts = authorization.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise AuthenticationError("Invalid Authorization header format. Use 'Bearer <token>'")

        token = parts[1].strip()

        if token.startswith(self.API_KEY_PREFIX):
            key_data = self.validate_api_key(token)
            return {
                "user_id": key_data["user_id"],
                "auth_type": "api_key",
                "scopes": key_data["scopes"],
            }
        else:
            claims = self.validate_jwt_token(token)
            return {
                "user_id": claims["sub"],
                "auth_type": "jwt",
                "scopes": claims.get("scopes", ["chat", "embeddings", "models"]),
            }


# Module-level singleton
_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """Return the global auth service instance."""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service
