"""Per-query token and cost accounting.

A :class:`~researchrag.config.pricing.CostLedger` is bound to the current query through a
context variable; :class:`MeteredLLM` and :class:`MeteredEmbeddings` record every call into
whichever ledger is active, so long-lived shared clients still attribute usage to the right
query (and to the right stage via the request's task name).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from researchrag.config.pricing import CostLedger
from researchrag.embeddings.base import EmbedKind
from researchrag.embeddings.service import EmbeddingService
from researchrag.llm.base import LLMClient, LLMRequest, LLMResponse, StreamEvent
from researchrag.observability.metrics import COST, TOKENS

_LEDGER: ContextVar[CostLedger | None] = ContextVar("researchrag_ledger", default=None)


@contextmanager
def metered(ledger: CostLedger) -> Iterator[CostLedger]:
    """Bind ``ledger`` as the active ledger for the enclosed code."""
    token = _LEDGER.set(ledger)
    try:
        yield ledger
    finally:
        _LEDGER.reset(token)


def _record(stage: str, model_id: str, input_tokens: int, output_tokens: int) -> None:
    TOKENS.labels(model=model_id, kind="input").inc(input_tokens)
    TOKENS.labels(model=model_id, kind="output").inc(output_tokens)
    ledger = _LEDGER.get()
    if ledger is not None:
        usd = ledger.record(stage, model_id, input_tokens, output_tokens)
        COST.labels(stage=stage).inc(usd)


class MeteredLLM:
    """Wraps an :class:`LLMClient` and records usage into the active ledger."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model

    @property
    def model_id(self) -> str:
        """``provider/model``."""
        return f"{self.provider}/{self.model}"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete and record usage."""
        response = await self.inner.complete(request)
        _record(request.task, self.model_id, response.input_tokens, response.output_tokens)
        return response

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Stream and record the final usage event."""
        async for event in self.inner.stream(request):
            if event.input_tokens is not None:
                _record(request.task, self.model_id, event.input_tokens, event.output_tokens or 0)
            yield event


class MeteredEmbeddings(EmbeddingService):
    """An :class:`EmbeddingService` that records embedding tokens into the active ledger."""

    async def embed(self, texts: list[str], kind: EmbedKind = "document") -> list[list[float]]:
        """Embed and record the tokens billed for cache misses."""
        before = self.tokens_used
        vectors = await super().embed(texts, kind)
        if self.tokens_used > before:
            _record("embed", self.model_id, self.tokens_used - before, 0)
        return vectors
