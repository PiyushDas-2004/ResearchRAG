"""Embedding provider interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

EmbedKind = Literal["document", "query"]


@dataclass(frozen=True)
class EmbeddingBatch:
    """Vectors for a batch of texts plus the number of input tokens billed."""

    vectors: list[list[float]]
    input_tokens: int


class Embedder(Protocol):
    """Turns texts into fixed-dimension vectors."""

    provider: str
    model: str
    dimension: int

    async def embed(self, texts: list[str], kind: EmbedKind) -> EmbeddingBatch:
        """Embed ``texts``; ``kind`` lets providers use asymmetric query/document modes."""
        ...
