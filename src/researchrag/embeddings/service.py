"""Batching, rate limiting, retries and caching around any :class:`Embedder`."""

from __future__ import annotations

import hashlib
from collections import OrderedDict

from researchrag.config.pipeline import EmbeddingConfig
from researchrag.embeddings.base import Embedder, EmbedKind
from researchrag.errors import ConfigurationError, ProviderError
from researchrag.resilience import AsyncRateLimiter, Sleep, retry_async


class EmbeddingService:
    """Embeds texts in batches with an LRU cache, a rate limiter and bounded retries."""

    def __init__(self, embedder: Embedder, config: EmbeddingConfig, *, sleep: Sleep | None = None) -> None:
        self.embedder = embedder
        self.config = config
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._limiter = AsyncRateLimiter(config.requests_per_minute)
        self._sleep = sleep
        self.cache_hits = 0
        self.tokens_used = 0

    @property
    def model_id(self) -> str:
        """``provider/model`` identifier stored with embeddings."""
        return f"{self.embedder.provider}/{self.embedder.model}"

    @property
    def dimension(self) -> int:
        """Vector dimension."""
        return self.embedder.dimension

    def _key(self, text: str, kind: EmbedKind) -> str:
        return hashlib.sha256(f"{self.model_id}|{self.dimension}|{kind}|{text}".encode()).hexdigest()

    async def embed(self, texts: list[str], kind: EmbedKind = "document") -> list[list[float]]:
        """Embed texts, serving repeats from the cache; output order matches input order."""
        result: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for i, text in enumerate(texts):
            cached = self._cache.get(self._key(text, kind))
            if cached is not None:
                self._cache.move_to_end(self._key(text, kind))
                result[i] = cached
                self.cache_hits += 1
            else:
                missing.append(i)
        size = self.config.batch_size
        for start in range(0, len(missing), size):
            idxs = missing[start : start + size]
            batch_texts = [texts[i] for i in idxs]
            await self._limiter.acquire()
            batch = await retry_async(
                lambda bt=batch_texts: self.embedder.embed(bt, kind),  # type: ignore[misc]
                max_retries=self.config.max_retries,
                sleep=self._sleep,
            )
            if len(batch.vectors) != len(idxs) or any(len(v) != self.dimension for v in batch.vectors):
                raise ProviderError(
                    "Embedding provider returned the wrong number or size of vectors.",
                    detail={"expected_dim": self.dimension},
                )
            self.tokens_used += batch.input_tokens
            for i, vec in zip(idxs, batch.vectors, strict=True):
                result[i] = vec
                self._remember(self._key(texts[i], kind), vec)
        return [v for v in result if v is not None]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return (await self.embed([text], "query"))[0]

    def _remember(self, key: str, vec: list[float]) -> None:
        if self.config.cache_size <= 0:
            return
        self._cache[key] = vec
        if len(self._cache) > self.config.cache_size:
            self._cache.popitem(last=False)


def build_embedder(config: EmbeddingConfig) -> Embedder:
    """Instantiate the configured embedding provider."""
    if config.provider == "fake":
        from researchrag.embeddings.fake import HashingEmbedder  # noqa: PLC0415

        return HashingEmbedder(config.dimension, config.model)
    if config.provider == "openai":
        from researchrag.embeddings.openai_embedder import OpenAIEmbedder  # noqa: PLC0415

        return OpenAIEmbedder(config.model, config.dimension)
    if config.provider == "gemini":
        from researchrag.embeddings.gemini_embedder import GeminiEmbedder  # noqa: PLC0415

        return GeminiEmbedder(config.model, config.dimension)
    if config.provider == "local":
        from researchrag.embeddings.local_embedder import LocalEmbedder  # noqa: PLC0415

        return LocalEmbedder(config.model, config.dimension)
    raise ConfigurationError(f"Unknown embedding provider {config.provider}")  # pragma: no cover
