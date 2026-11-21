"""Domain exceptions shared across layers.

Each error carries a stable machine-readable ``code`` that the API layer maps onto an
RFC 7807 problem type and HTTP status, so lower layers never import HTTP concepts.
"""

from __future__ import annotations


class ResearchRAGError(Exception):
    """Base class for all expected, user-facing errors."""

    code = "internal_error"

    def __init__(self, message: str, *, detail: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class ValidationFailedError(ResearchRAGError):
    """Input failed validation (bad config, bad parameters)."""

    code = "validation_failed"


class UploadRejectedError(ResearchRAGError):
    """An uploaded file was rejected by the upload validator or the PDF parser."""

    code = "upload_rejected"


class NotFoundError(ResearchRAGError):
    """A requested resource does not exist."""

    code = "not_found"


class ConflictError(ResearchRAGError):
    """The request conflicts with existing state (e.g. idempotency key reuse)."""

    code = "conflict"


class ProviderError(ResearchRAGError):
    """An upstream model provider failed in a non-retryable way."""

    code = "provider_error"


class RetryableProviderError(ProviderError):
    """An upstream provider failed in a way that may succeed on retry (timeouts, 429, 5xx)."""

    code = "provider_unavailable"

    def __init__(
        self, message: str, *, retry_after: float | None = None, detail: dict[str, object] | None = None
    ) -> None:
        super().__init__(message, detail=detail)
        self.retry_after = retry_after


class ProviderTimeoutError(RetryableProviderError):
    """An upstream provider call timed out."""

    code = "provider_timeout"


class RateLimitedError(RetryableProviderError):
    """An upstream provider returned HTTP 429 or equivalent."""

    code = "provider_rate_limited"


class ConfigurationError(ResearchRAGError):
    """The service is misconfigured (missing optional dependency, missing key)."""

    code = "configuration_error"
