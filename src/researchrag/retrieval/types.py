"""Retrieval data types and the search-backend interface."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol


@dataclass(frozen=True)
class SearchFilters:
    """Metadata filters applied inside the search query (not after retrieval)."""

    document_ids: tuple[str, ...] = ()
    collection: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    authors: tuple[str, ...] = ()
    include_references: bool = False

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly representation for traces."""
        return {
            "document_ids": list(self.document_ids),
            "collection": self.collection,
            "year_min": self.year_min,
            "year_max": self.year_max,
            "authors": list(self.authors),
            "include_references": self.include_references,
        }

    @property
    def active(self) -> bool:
        """True when any restrictive filter is set."""
        return bool(self.document_ids or self.collection or self.year_min or self.year_max or self.authors)


@dataclass(frozen=True)
class EmbeddingVersionRef:
    """Identifies which embedding version (and HNSW index) to search."""

    id: int
    dimension: int
    model_id: str


@dataclass(frozen=True)
class Candidate:
    """A retrieved chunk with per-method scores."""

    chunk_id: str
    document_id: str
    text: str
    page_start: int
    page_end: int
    section: str
    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    parent_id: str | None = None
    score: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    embedding: tuple[float, ...] | None = None

    def with_score(self, name: str, value: float, rank: int | None = None) -> Candidate:
        """Return a copy with ``score = value`` recorded under ``name``."""
        scores = {**self.scores, name: value}
        ranks = dict(self.ranks) if rank is None else {**self.ranks, name: rank}
        return replace(self, score=value, scores=scores, ranks=ranks)


@dataclass(frozen=True)
class ParentPassage:
    """A larger passage used as model context for a small retrieved chunk."""

    chunk_id: str
    document_id: str
    text: str
    page_start: int
    page_end: int
    section: str


class SearchBackend(Protocol):
    """Candidate generation over indexed chunks of one chunking configuration."""

    async def vector_search(
        self,
        query_vector: list[float],
        *,
        version: EmbeddingVersionRef,
        chunk_config_hash: str,
        k: int,
        filters: SearchFilters,
        ef_search: int,
    ) -> list[Candidate]:
        """Nearest neighbours by cosine distance (score = 1 - distance)."""
        ...

    async def lexical_search(
        self, query: str, *, chunk_config_hash: str, k: int, filters: SearchFilters
    ) -> list[Candidate]:
        """Full-text matches ranked by a BM25-style score."""
        ...

    async def get_parents(self, parent_ids: list[str]) -> dict[str, ParentPassage]:
        """Fetch parent passages by chunk id."""
        ...
