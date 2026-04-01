"""Custom exception classes for NeuronLM."""

from __future__ import annotations

from typing import Optional


class NeuronLMError(Exception):
    """Base exception for all NeuronLM errors."""

    def __init__(self, message: str, code: Optional[str] = None) -> None:
        self.message = message
        self.code = code
        super().__init__(message)


class AuthenticationError(NeuronLMError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Invalid or missing authentication credentials") -> None:
        super().__init__(message, code="authentication_error")


class PermissionError(NeuronLMError):
    """Raised when user lacks permissions."""

    def __init__(self, message: str = "Insufficient permissions") -> None:
        super().__init__(message, code="permission_error")


class RateLimitError(NeuronLMError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded. Please retry after some time.") -> None:
        super().__init__(message, code="rate_limit_exceeded")


class ModelNotFoundError(NeuronLMError):
    """Raised when requested model is not found."""

    def __init__(self, model_id: str) -> None:
        super().__init__(f"Model '{model_id}' not found", code="model_not_found")
        self.model_id = model_id


class ModelNotReadyError(NeuronLMError):
    """Raised when model is not ready for inference."""

    def __init__(self, model_id: str) -> None:
        super().__init__(f"Model '{model_id}' is not ready", code="model_not_ready")
        self.model_id = model_id


class ValidationError(NeuronLMError):
    """Raised for request validation errors."""

    def __init__(self, message: str, param: Optional[str] = None) -> None:
        super().__init__(message, code="invalid_request_error")
        self.param = param


class InferenceError(NeuronLMError):
    """Raised when inference fails."""

    def __init__(self, message: str = "Inference failed") -> None:
        super().__init__(message, code="inference_error")


class ConversationNotFoundError(NeuronLMError):
    """Raised when conversation is not found."""

    def __init__(self, conversation_id: str) -> None:
        super().__init__(f"Conversation '{conversation_id}' not found", code="not_found")
        self.conversation_id = conversation_id


class ContentModerationError(NeuronLMError):
    """Raised when content violates moderation policies."""

    def __init__(self, message: str = "Content violates usage policies") -> None:
        super().__init__(message, code="content_policy_violation")


class QuotaExceededError(NeuronLMError):
    """Raised when usage quota is exceeded."""

    def __init__(self, message: str = "Usage quota exceeded") -> None:
        super().__init__(message, code="quota_exceeded")
