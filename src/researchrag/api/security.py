"""API-key authentication and per-key rate limiting."""

from __future__ import annotations

import hashlib
import hmac
import time
from collections import deque
from dataclasses import dataclass

from fastapi import HTTPException, Request

from researchrag.config.settings import Settings


@dataclass(frozen=True)
class Principal:
    """The authenticated caller (only a hash of the key is ever stored or logged)."""

    key_id: str
    anonymous: bool = False


def key_id(raw_key: str) -> str:
    """Stable, non-reversible identifier for an API key."""
    return hashlib.sha256(raw_key.encode()).hexdigest()[:16]


class ApiKeyAuth:
    """Validates ``X-API-Key`` or ``Authorization: Bearer`` against configured keys (constant time)."""

    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.auth_enabled
        self._digests = [hashlib.sha256(k.get_secret_value().encode()).digest() for k in settings.api_keys]

    def authenticate(self, request: Request) -> Principal:
        """Return the caller or raise 401."""
        if not self.enabled:
            return Principal("anonymous", anonymous=True)
        raw = request.headers.get("x-api-key")
        auth = request.headers.get("authorization", "")
        if not raw and auth.lower().startswith("bearer "):
            raw = auth[7:].strip()
        if not raw:
            raise HTTPException(
                401, "Missing API key (X-API-Key header).", headers={"WWW-Authenticate": "Bearer"}
            )
        digest = hashlib.sha256(raw.encode()).digest()
        if not any(hmac.compare_digest(digest, known) for known in self._digests):
            raise HTTPException(401, "Invalid API key.", headers={"WWW-Authenticate": "Bearer"})
        return Principal(key_id(raw))


class RateLimiter:
    """Sliding-window limit per principal (per process; use a shared store for multi-instance)."""

    def __init__(self, per_minute: int, *, clock: object = time.monotonic) -> None:
        self.per_minute = per_minute
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def check(self, principal: Principal) -> None:
        """Record a hit or raise 429 with Retry-After."""
        if self.per_minute <= 0:
            return
        now = self._clock()  # type: ignore[operator]
        window = self._hits.setdefault(principal.key_id, deque())
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= self.per_minute:
            retry = int(60 - (now - window[0])) + 1
            raise HTTPException(
                429,
                f"Rate limit of {self.per_minute} requests/minute exceeded.",
                headers={"Retry-After": str(retry)},
            )
        window.append(now)
