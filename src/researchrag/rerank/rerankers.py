"""Second-stage rerankers: none, lexical (offline), cross-encoder (local) and LLM-based."""

from __future__ import annotations

import asyncio
import itertools
import json
from collections import OrderedDict
from typing import Any, Protocol

from researchrag.errors import ConfigurationError, ResearchRAGError
from researchrag.generation.json_repair import parse_json_object
from researchrag.generation.prompts import PromptPassage, rerank_messages
from researchrag.llm.base import LLMClient, LLMRequest
from researchrag.retrieval.types import Candidate
from researchrag.textutils import content_terms, coverage, numbers


class Reranker(Protocol):
    """Reorders candidates for a query."""

    name: str

    async def scores(self, query: str, candidates: list[Candidate]) -> list[float]:
        """Return one relevance score per candidate (higher is better)."""
        ...


class LexicalReranker:
    """Offline reranker: query-term coverage, bigram overlap and number agreement."""

    name = "lexical"

    async def scores(self, query: str, candidates: list[Candidate]) -> list[float]:
        """Score candidates deterministically."""
        q_terms = content_terms(query)
        q_set = set(q_terms)
        q_bigrams = set(itertools.pairwise(q_terms))
        q_nums = numbers(query)
        out = []
        for cand in candidates:
            terms = content_terms(cand.text)
            cov = coverage(q_set, set(terms))
            bigrams = set(itertools.pairwise(terms))
            bi = len(q_bigrams & bigrams) / len(q_bigrams) if q_bigrams else 0.0
            num = len(q_nums & numbers(cand.text)) / len(q_nums) if q_nums else 0.0
            out.append(round(0.7 * cov + 0.2 * bi + 0.1 * num, 6))
        return out


class CrossEncoderReranker:
    """Local cross-encoder (sentence-transformers ``CrossEncoder``; optional ``[local]`` extra)."""

    name = "cross_encoder"

    def __init__(self, model: str, *, encoder: Any | None = None) -> None:
        if encoder is None:
            try:
                from sentence_transformers import CrossEncoder  # noqa: PLC0415 - optional
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ConfigurationError(
                    "pip install 'researchrag[local]' for cross-encoder reranking"
                ) from exc
            encoder = CrossEncoder(model)
        self._encoder: Any = encoder

    async def scores(self, query: str, candidates: list[Candidate]) -> list[float]:
        """Score (query, passage) pairs with the cross-encoder."""
        pairs = [(query, c.text) for c in candidates]
        raw = await asyncio.to_thread(self._encoder.predict, pairs)
        return [float(x) for x in raw]


class LLMReranker:
    """Listwise LLM reranking; falls back to the incoming order if the model output is unusable."""

    name = "llm"

    def __init__(self, llm: LLMClient, max_tokens: int = 600) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    async def scores(self, query: str, candidates: list[Candidate]) -> list[float]:
        """Ask the LLM to score each passage 0-10."""
        passages = [
            PromptPassage(i + 1, c.text, c.title, c.page_start, c.section) for i, c in enumerate(candidates)
        ]
        fallback = [float(len(candidates) - i) / len(candidates) for i in range(len(candidates))]
        try:
            response = await self.llm.complete(
                LLMRequest(rerank_messages(query, passages), task="rerank", max_tokens=self.max_tokens)
            )
            data = parse_json_object(response.text)
            by_id = {int(item["id"]): float(item["score"]) for item in data.get("scores", [])}
        except (ResearchRAGError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return fallback
        return [by_id.get(i + 1, 0.0) / 10.0 for i in range(len(candidates))]


class CachedReranker:
    """LRU cache of ``(query, chunk_id) -> score`` in front of any reranker."""

    def __init__(self, inner: Reranker, size: int = 4096) -> None:
        self.inner = inner
        self.name = inner.name
        self.size = size
        self._cache: OrderedDict[tuple[str, str], float] = OrderedDict()
        self.hits = 0

    async def scores(self, query: str, candidates: list[Candidate]) -> list[float]:
        """Serve cached scores; score only the misses."""
        found = {
            c.chunk_id: self._cache[(query, c.chunk_id)]
            for c in candidates
            if (query, c.chunk_id) in self._cache
        }
        missing = [c for c in candidates if c.chunk_id not in found]
        if missing:
            for cand, score in zip(missing, await self.inner.scores(query, missing), strict=True):
                found[cand.chunk_id] = score
                self._cache[(query, cand.chunk_id)] = score
                while len(self._cache) > self.size:
                    self._cache.popitem(last=False)
        self.hits += len(candidates) - len(missing)
        return [found[c.chunk_id] for c in candidates]


def build_reranker(provider: str, model: str, llm: LLMClient | None, cache_size: int) -> Reranker | None:
    """Create the configured reranker (``None`` for ``provider == 'none'``)."""
    inner: Reranker
    if provider == "none":
        return None
    if provider == "lexical":
        inner = LexicalReranker()
    elif provider == "cross_encoder":
        inner = CrossEncoderReranker(model)
    else:
        if llm is None:
            raise ConfigurationError("LLM reranking needs an LLM client")
        inner = LLMReranker(llm)
    return CachedReranker(inner, cache_size) if cache_size > 0 else inner
