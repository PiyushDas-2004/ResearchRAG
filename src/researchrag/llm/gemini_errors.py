"""Map google-genai SDK exceptions onto ResearchRAG domain errors."""

from __future__ import annotations

from researchrag.errors import (
    ProviderError,
    ProviderTimeoutError,
    RateLimitedError,
    ResearchRAGError,
    RetryableProviderError,
)


def map_gemini_error(exc: Exception) -> ResearchRAGError:
    """Translate a ``google.genai`` exception (by attributes, so the SDK stays optional)."""
    code = getattr(exc, "code", None)
    name = type(exc).__name__
    if isinstance(exc, TimeoutError) or "Timeout" in name:
        return ProviderTimeoutError("Gemini request timed out.")
    if code == 429:
        return RateLimitedError("Gemini rate limit exceeded.")
    if name == "ServerError" or (isinstance(code, int) and code >= 500):
        return RetryableProviderError(f"Gemini temporarily unavailable ({code}).")
    return ProviderError(f"Gemini request failed ({name}).", detail={"code": code})
