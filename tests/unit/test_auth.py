"""Unit tests for authentication and authorization."""

import time
import pytest

from neuronlm.core.auth import AuthService
from neuronlm.core.exceptions import AuthenticationError


class TestApiKeyGeneration:
    """API key generation and validation tests."""

    def test_generate_api_key_format(self, auth_service: AuthService):
        """UT-AUTH-001: Generated key has correct format."""
        raw_key, key_hash = auth_service.generate_api_key("user-1", "test")
        assert raw_key.startswith("nlm-")
        assert len(raw_key) > 10
        assert len(key_hash) == 64  # SHA-256 hex length

    def test_invalid_key_prefix(self, auth_service: AuthService):
        """UT-AUTH-002: Invalid prefix is rejected."""
        with pytest.raises(AuthenticationError, match="Invalid API key format"):
            auth_service.validate_api_key("xxx-a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6")

    def test_key_too_short(self, auth_service: AuthService):
        """UT-AUTH-003: Short key is rejected."""
        with pytest.raises(AuthenticationError, match="Invalid API key format"):
            auth_service.validate_api_key("nlm-abc")

    def test_key_hash_verification(self, auth_service: AuthService):
        """UT-AUTH-004: Hash verification works correctly."""
        raw_key, key_hash = auth_service.generate_api_key("user-1")
        computed_hash = auth_service.hash_api_key(raw_key)
        assert computed_hash == key_hash

    def test_valid_key_accepted(self, auth_service: AuthService):
        """Valid API key is accepted and returns metadata."""
        raw_key, _ = auth_service.generate_api_key("user-1", "my-key")
        metadata = auth_service.validate_api_key(raw_key)
        assert metadata["user_id"] == "user-1"
        assert metadata["name"] == "my-key"
        assert metadata["is_active"] is True

    def test_unknown_key_rejected(self, auth_service: AuthService):
        """Unknown API key is rejected."""
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            auth_service.validate_api_key("nlm-" + "x" * 43)  # Valid format, unknown key

    def test_deactivated_key_rejected(self, auth_service: AuthService):
        """UT-AUTH-006: Deactivated key is rejected."""
        raw_key, key_hash = auth_service.generate_api_key("user-1")
        auth_service.deactivate_api_key(key_hash)

        with pytest.raises(AuthenticationError, match="deactivated"):
            auth_service.validate_api_key(raw_key)


class TestJWTTokens:
    """JWT token creation and validation tests."""

    def test_create_and_validate_jwt(self, auth_service: AuthService):
        """UT-AUTH-007: Valid JWT is created and validated."""
        token = auth_service.create_jwt_token("user-1")
        claims = auth_service.validate_jwt_token(token)
        assert claims["sub"] == "user-1"
        assert claims["iss"] == "neuronlm"

    def test_jwt_with_extra_claims(self, auth_service: AuthService):
        """JWT with extra claims preserves them."""
        token = auth_service.create_jwt_token(
            "user-1",
            extra_claims={"role": "admin"},
        )
        claims = auth_service.validate_jwt_token(token)
        assert claims["role"] == "admin"

    def test_invalid_jwt_rejected(self, auth_service: AuthService):
        """UT-AUTH-009: Tampered JWT is rejected."""
        token = auth_service.create_jwt_token("user-1")
        tampered = token[:-5] + "XXXXX"  # Corrupt signature
        with pytest.raises(AuthenticationError, match="Invalid JWT"):
            auth_service.validate_jwt_token(tampered)

    def test_garbage_jwt_rejected(self, auth_service: AuthService):
        """Complete garbage as JWT is rejected."""
        with pytest.raises(AuthenticationError, match="Invalid JWT"):
            auth_service.validate_jwt_token("not.a.jwt")


class TestUnifiedAuthentication:
    """Unified authenticate() method tests."""

    def test_authenticate_with_api_key(self, auth_service: AuthService, api_key: str):
        """API key authentication via unified method."""
        result = auth_service.authenticate(f"Bearer {api_key}")
        assert result["user_id"] == "test-user-001"
        assert result["auth_type"] == "api_key"

    def test_authenticate_with_jwt(self, auth_service: AuthService):
        """JWT authentication via unified method."""
        token = auth_service.create_jwt_token("user-jwt-1")
        result = auth_service.authenticate(f"Bearer {token}")
        assert result["user_id"] == "user-jwt-1"
        assert result["auth_type"] == "jwt"

    def test_authenticate_missing_header(self, auth_service: AuthService):
        """Missing Authorization header is rejected."""
        with pytest.raises(AuthenticationError, match="Missing"):
            auth_service.authenticate("")

    def test_authenticate_wrong_scheme(self, auth_service: AuthService):
        """Non-Bearer scheme is rejected."""
        with pytest.raises(AuthenticationError, match="Invalid Authorization"):
            auth_service.authenticate("Basic dXNlcjpwYXNz")

    def test_authenticate_no_token(self, auth_service: AuthService):
        """Bearer without token is rejected."""
        with pytest.raises(AuthenticationError):
            auth_service.authenticate("Bearer ")
