"""OpenAI embeddings adapter (``openai`` SDK, ``embeddings.create``)."""

from __future__ import annotations

from typing import Any

from researchrag.embeddings.base import EmbeddingBatch, EmbedKind
from researchrag.errors import ConfigurationError
from researchrag.llm.openai_errors import map_openai_error


class OpenAIEmbedder:
    """Embeddings via the OpenAI API; ``dimension`` is passed as ``dimensions`` (text-embedding-3+)."""

    provider = "openai"

    def __init__(
        self, model: str, dimension: int, *, client: Any | None = None, timeout_s: float = 60.0
    ) -> None:
        self.model = model
        self.dimension = dimension
        self._timeout = timeout_s
        if client is None:
            try:
                from openai import AsyncOpenAI  # noqa: PLC0415, TID251 - adapter module
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ConfigurationError(
                    "pip install 'researchrag[openai]' to use OpenAI embeddings"
                ) from exc
            client = AsyncOpenAI(max_retries=0)
        self._client: Any = client

    async def embed(self, texts: list[str], kind: EmbedKind) -> EmbeddingBatch:
        """Embed a batch of texts."""
        try:
            response = await self._client.embeddings.create(
                model=self.model, input=texts, dimensions=self.dimension, timeout=self._timeout
            )
        except Exception as exc:
            raise map_openai_error(exc) from exc
        vectors = [list(item.embedding) for item in sorted(response.data, key=lambda d: d.index)]
        tokens = int(getattr(response.usage, "prompt_tokens", 0) or 0)
        return EmbeddingBatch(vectors, tokens)
