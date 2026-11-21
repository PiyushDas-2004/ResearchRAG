"""Retry with exponential backoff and an async token-bucket rate limiter."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from researchrag.errors import RetryableProviderError

T = TypeVar("T")
Sleep = Callable[[float], Awaitable[None]]


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    base_delay: float = 0.5,
    max_delay: float = 20.0,
    sleep: Sleep | None = None,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    """Call ``fn`` and retry :class:`RetryableProviderError` with jittered exponential backoff.

    A provider-supplied ``retry_after`` takes precedence over the computed delay.
    Non-retryable errors propagate immediately.
    """
    attempt = 0
    while True:
        try:
            return await fn()
        except RetryableProviderError as exc:
            if attempt >= max_retries:
                raise
            delay = min(max_delay, base_delay * (2**attempt)) * (0.5 + random.random() / 2)  # retry jitter  # noqa: S311  # nosec B311
            if exc.retry_after is not None:
                delay = min(max_delay, exc.retry_after)
            if on_retry is not None:
                on_retry(attempt + 1, exc)
            await (sleep or asyncio.sleep)(delay)
            attempt += 1


class AsyncRateLimiter:
    """Token bucket allowing ``per_minute`` acquisitions per minute (burst = one second's worth)."""

    def __init__(
        self, per_minute: int, *, clock: Callable[[], float] = time.monotonic, sleep: Sleep = asyncio.sleep
    ):
        self.rate = per_minute / 60.0
        self.capacity = max(1.0, self.rate)
        self.tokens = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        self._lock = asyncio.Lock()

    async def acquire(self, cost: float = 1.0) -> None:
        """Wait until ``cost`` tokens are available, then consume them."""
        async with self._lock:
            while True:
                now = self._clock()
                self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.rate)
                self._last = now
                if self.tokens >= cost or self.tokens >= self.capacity:
                    self.tokens -= cost
                    return
                await self._sleep((cost - self.tokens) / self.rate)
