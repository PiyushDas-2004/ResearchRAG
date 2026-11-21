"""Local sentence-transformers embeddings (optional ``[local]`` extra)."""

from __future__ import annotations

import asyncio
from typing import Any

from researchrag.embeddings.base import EmbeddingBatch, EmbedKind
from researchrag.errors import ConfigurationError


class LocalEmbedder:
    """Runs a sentence-transformers model in a worker thread."""

    provider = "local"

    def __init__(self, model: str, dimension: int, *, encoder: Any | None = None) -> None:
        self.model = model
        self.dimension = dimension
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - optional
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ConfigurationError("pip install 'researchrag[local]' to use local embeddings") from exc
            encoder = SentenceTransformer(model)
        self._encoder: Any = encoder

    async def embed(self, texts: list[str], kind: EmbedKind) -> EmbeddingBatch:
        """Embed texts locally (normalised)."""
        vectors = await asyncio.to_thread(self._encoder.encode, texts, normalize_embeddings=True)
        out = [list(map(float, v))[: self.dimension] for v in vectors]
        return EmbeddingBatch(out, sum(len(t.split()) for t in texts))
