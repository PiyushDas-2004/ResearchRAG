"""Provider-agnostic LLM interface.

Core pipeline code talks only to :class:`LLMClient`; provider SDKs are confined to adapter
modules. ``LLMRequest.task`` names the pipeline step (``answer``, ``rewrite``, ``verify``...);
real providers ignore it, while the offline mock uses it to pick a deterministic behaviour.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, Protocol

from researchrag.ingestion.tokenizer import SimpleTokenizer
from researchrag.resilience import Sleep, retry_async

Role = Literal["system", "user", "assistant"]
_TOK = SimpleTokenizer()


@dataclass(frozen=True)
class Message:
    """One chat message."""

    role: Role
    content: str


@dataclass(frozen=True)
class LLMRequest:
    """A completion request."""

    messages: list[Message]
    task: str
    max_tokens: int = 900
    temperature: float | None = 0.0
    json_mode: bool = True
    timeout_s: float = 60.0


@dataclass(frozen=True)
class LLMResponse:
    """A completed response with token usage."""

    text: str
    input_tokens: int
    output_tokens: int
    model_id: str


@dataclass(frozen=True)
class StreamEvent:
    """A streamed text delta; the final event carries usage."""

    delta: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class UsageTotals:
    """Accumulated usage across calls (for cost accounting)."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    by_task: dict[str, int] = field(default_factory=dict)


class LLMClient(Protocol):
    """A chat-completion model."""

    provider: str
    model: str

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return the full response."""
        ...

    def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Yield text deltas, then a final event with usage."""
        ...


def estimate_tokens(messages: list[Message]) -> int:
    """Approximate prompt tokens when a provider does not report usage."""
    return sum(_TOK.count(m.content) + 4 for m in messages)


class RetryingLLM:
    """Wraps an :class:`LLMClient` with bounded retries on retryable provider errors.

    Streaming is retried only if the failure happens before the first delta (otherwise the
    caller would receive duplicated text).
    """

    def __init__(self, inner: LLMClient, *, max_retries: int, sleep: Sleep | None = None) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self.max_retries = max_retries
        self._sleep = sleep
        self.usage = UsageTotals()

    @property
    def model_id(self) -> str:
        """``provider/model``."""
        return f"{self.provider}/{self.model}"

    def _record(self, task: str, input_tokens: int, output_tokens: int) -> None:
        self.usage.input_tokens += input_tokens
        self.usage.output_tokens += output_tokens
        self.usage.calls += 1
        self.usage.by_task[task] = self.usage.by_task.get(task, 0) + input_tokens + output_tokens

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete with retries."""
        response = await retry_async(
            lambda: self.inner.complete(request), max_retries=self.max_retries, sleep=self._sleep
        )
        self._record(request.task, response.input_tokens, response.output_tokens)
        return response

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Stream with retries before the first delta."""
        attempt = 0
        while True:
            started = False
            try:
                async for event in self.inner.stream(request):
                    if event.delta:
                        started = True
                    if event.input_tokens is not None:
                        self._record(request.task, event.input_tokens, event.output_tokens or 0)
                    yield event
                return
            except Exception as exc:
                from researchrag.errors import RetryableProviderError  # noqa: PLC0415

                if started or not isinstance(exc, RetryableProviderError) or attempt >= self.max_retries:
                    raise
                attempt += 1
                delay = exc.retry_after if exc.retry_after is not None else 0.5 * 2**attempt
                if self._sleep is not None:
                    await self._sleep(delay)
                else:
                    import asyncio  # noqa: PLC0415

                    await asyncio.sleep(min(delay, 20.0))
