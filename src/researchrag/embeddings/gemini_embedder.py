"""Gemini embeddings adapter (``google-genai`` SDK, ``models.embed_content``)."""

from __future__ import annotations

from typing import Any

from researchrag.embeddings.base import EmbeddingBatch, EmbedKind
from researchrag.errors import ConfigurationError
from researchrag.llm.gemini_errors import map_gemini_error


class GeminiEmbedder:
    """Embeddings via the Gemini API with asymmetric task types and ``output_dimensionality``."""

    provider = "gemini"

    def __init__(self, model: str, dimension: int, *, client: Any | None = None) -> None:
        self.model = model
        self.dimension = dimension
        if client is None:
            try:
                from google import genai  # noqa: PLC0415, TID251 - adapter module
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ConfigurationError(
                    "pip install 'researchrag[gemini]' to use Gemini embeddings"
                ) from exc
            client = genai.Client()
        self._client: Any = client

    async def embed(self, texts: list[str], kind: EmbedKind) -> EmbeddingBatch:
        """Embed a batch of texts (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY task types)."""
        config = {
            "task_type": "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT",
            "output_dimensionality": self.dimension,
        }
        try:
            response = await self._client.aio.models.embed_content(
                model=self.model, contents=texts, config=config
            )
        except Exception as exc:
            raise map_gemini_error(exc) from exc
        vectors = [_normalise(list(e.values)) for e in response.embeddings]
        usage = getattr(response, "usage_metadata", None)
        tokens = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        return EmbeddingBatch(vectors, tokens)


def _normalise(vec: list[float]) -> list[float]:
    # Truncated (output_dimensionality < native) Gemini embeddings are not unit-length.
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm > 0 else vec
