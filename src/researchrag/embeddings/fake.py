"""Deterministic hashing embedder for offline use and tests.

Feature hashing of stemmed unigrams, bigrams and character 4-grams into a signed,
L2-normalised vector. Cosine similarity therefore approximates weighted lexical overlap with
some tolerance for morphology. It is *not* a semantic model; it exists so the whole pipeline
(including pgvector and HNSW) runs and is testable with zero API keys.
"""

from __future__ import annotations

import hashlib
import itertools
import math

from researchrag.embeddings.base import EmbeddingBatch, EmbedKind
from researchrag.ingestion.tokenizer import SimpleTokenizer
from researchrag.textutils import content_terms


class HashingEmbedder:
    """``fake/fake-hash-v1``: reproducible bag-of-features embeddings."""

    provider = "fake"

    def __init__(self, dimension: int = 384, model: str = "fake-hash-v1") -> None:
        self.dimension = dimension
        self.model = model
        self._tok = SimpleTokenizer()

    def _features(self, text: str) -> dict[str, float]:
        terms = content_terms(text)
        feats: dict[str, float] = {}
        for term in terms:
            feats["u:" + term] = feats.get("u:" + term, 0.0) + 1.0
            padded = f"#{term}#"
            for i in range(max(1, len(padded) - 3)):
                gram = "c:" + padded[i : i + 4]
                feats[gram] = feats.get(gram, 0.0) + 0.25
        for a, b in itertools.pairwise(terms):
            key = f"b:{a}_{b}"
            feats[key] = feats.get(key, 0.0) + 0.5
        return feats

    def vector(self, text: str) -> list[float]:
        """Embed one text."""
        vec = [0.0] * self.dimension
        for feat, weight in self._features(text).items():
            digest = hashlib.blake2b(feat.encode(), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(weight)) if weight >= 1 else sign * weight
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm > 0 else vec

    async def embed(self, texts: list[str], kind: EmbedKind) -> EmbeddingBatch:
        """Embed a batch (``kind`` is ignored: the fake model is symmetric)."""
        return EmbeddingBatch([self.vector(t) for t in texts], sum(self._tok.count(t) for t in texts))
