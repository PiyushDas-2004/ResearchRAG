"""Map OpenAI SDK exceptions onto ResearchRAG domain errors."""

from __future__ import annotations

from researchrag.errors import (
    ProviderError,
    ProviderTimeoutError,
    RateLimitedError,
    ResearchRAGError,
    RetryableProviderError,
)


def map_openai_error(exc: Exception) -> ResearchRAGError:
    """Translate an ``openai`` exception (by class name, so the SDK stays optional)."""
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if name == "APITimeoutError" or isinstance(exc, TimeoutError):
        return ProviderTimeoutError("OpenAI request timed out.")
    if name == "RateLimitError" or status == 429:
        retry_after = None
        response = getattr(exc, "response", None)
        header = response.headers.get("retry-after") if response is not None else None
        if header:
            try:
                retry_after = float(header)
            except ValueError:
                retry_after = None
        return RateLimitedError("OpenAI rate limit exceeded.", retry_after=retry_after)
    if name in {"APIConnectionError", "InternalServerError"} or (isinstance(status, int) and status >= 500):
        return RetryableProviderError(f"OpenAI temporarily unavailable ({name}).")
    return ProviderError(f"OpenAI request failed ({name}).", detail={"status": status})
